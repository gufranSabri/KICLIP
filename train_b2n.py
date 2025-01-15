import pprint
import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils.parser import load_config, parse_args
import utils.misc as misc
import utils.metrics as metrics
import utils.checkpoint as cu
from utils.meters import AVAMeter, EpochTimer, TrainMeter, ValMeter
from config.defaults import assert_and_infer_cfg
from models.temporalclip_video_model import TemporalClipVideo
from models.scar import SCAR
import models.losses as losses
from datasets import utils
from datasets.build import build_dataset
import models.optimizer as optim
from models.contrastive import (
    contrastive_forward,
    contrastive_parameter_surgery,
)
from logger import Logger
import datetime
import shutil

def construct_optimizer(model, cfg):
    bn_parameters = []
    non_bn_parameters = []
    zero_parameters = []
    no_grad_parameters = []
    skip = {}
    
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()

    for name_m, m in model.named_modules():
        is_bn = isinstance(m, torch.nn.modules.batchnorm._NormBase)
        for name_p, p in m.named_parameters(recurse=False):
            name = "{}.{}".format(name_m, name_p).strip(".")
            if not p.requires_grad:
                no_grad_parameters.append(p)
            elif is_bn:
                bn_parameters.append(p)
            elif any(k in name for k in skip):
                zero_parameters.append(p)
            elif cfg.SOLVER.ZERO_WD_1D_PARAM and (
                len(p.shape) == 1 or name.endswith(".bias")
            ):
                zero_parameters.append(p)
            else:
                non_bn_parameters.append(p)
    
    optim_params = [
        {
            "params": bn_parameters,
            "weight_decay": cfg.BN.WEIGHT_DECAY,
            "layer_decay": 1.0,
            "apply_LARS": False,
        },
        {
            "params": non_bn_parameters,
            "weight_decay": cfg.SOLVER.WEIGHT_DECAY,
            "layer_decay": 1.0,
            "apply_LARS": cfg.SOLVER.LARS_ON,
        },
        {
            "params": zero_parameters,
            "weight_decay": 0.0,
            "layer_decay": 1.0,
            "apply_LARS": cfg.SOLVER.LARS_ON,
        },
    ]
    optim_params = [x for x in optim_params if len(x["params"])]

    return torch.optim.AdamW(
        optim_params,
        lr=cfg.SOLVER.BASE_LR,
        betas=cfg.SOLVER.BETAS,
        eps=1e-08,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )


def construct_loader(cfg, split="train"):
    shuffle = False
    drop_last = False

    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
    elif split in ["test", "test_openset"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))

    dataset = build_dataset(dataset_name, cfg, split)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        worker_init_fn=utils.loader_worker_init_fn(dataset),
        collate_fn=None
    )

    return loader

def train_epoch(loader, model, optimizer, scaler, cur_epoch, cfg, train_meter, logger):
    model.train()
    train_meter.iter_tic()
    data_size = len(loader)

    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
    if cfg.MODEL.KEEP_RAW_MODEL:
        raw_clip_params = {}
        for n, p in model.named_parameters():
            if 'raw_model' in n:
                p.requires_grad = False
                raw_clip_params[n] = p

    total_top1_err, total_top5_err, total_samples = 0.0, 0.0, 0
    for cur_iter, (inputs, labels, index, time, meta) in tqdm(enumerate(loader), total=data_size, desc="Train"):
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].to(cfg.DEVICE)
                    else:
                        inputs[i] = inputs[i].to(cfg.DEVICE)
            else:
                inputs = inputs.to(cfg.DEVICE)
            if not isinstance(labels, list):
                labels = labels.to(cfg.DEVICE)
                index = index.to(cfg.DEVICE)

                if cfg.DEVICE == "mps":
                    time = time.float().to(cfg.DEVICE)    
                else:
                    time = time.to(cfg.DEVICE)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].to(cfg.DEVICE)
                else:
                    meta[key] = val.to(cfg.DEVICE)

        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )

        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)
        train_meter.data_toc()

        # optimizer.zero_grad()
        if cfg.DEVICE == 'mps':
            with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
                optimizer.zero_grad()
        else:
            with torch.amp.autocast(cfg.DEVICE, enabled=cfg.TRAIN.MIXED_PRECISION):
                optimizer.zero_grad()

        if cfg.MODEL.KEEP_RAW_MODEL and cfg.MODEL.RAW_MODEL_DISTILLATION:
            outputs = model(inputs)
            if len(outputs[0]) == 2:
                [preds, img_encode], [raw_pred, raw_img_encode] = outputs
            elif len(outputs[0]) == 3:
                [preds, img_encode, text_encode], [raw_pred, raw_img_encode, raw_text_encode] = outputs
        else:
            preds = model(inputs)
            if isinstance(preds, list):
                preds = preds[0]

        loss = loss_fun(preds, labels)
        if cfg.MODEL.KEEP_RAW_MODEL and cfg.MODEL.RAW_MODEL_DISTILLATION:
            distillation_loss_1 =  1 - F.cosine_similarity(img_encode, raw_img_encode, dim=-1).mean()
            distillation_loss_2 =  1 - F.cosine_similarity(text_encode, raw_text_encode, dim=-1).mean()
            distillation_loss = distillation_loss_1 + distillation_loss_2

            # if cur_iter % cfg.LOG_PERIOD == 0:
            #     logger('Distillation Loss: %.8f'%distillation_loss.item())
            #     logger('Distillation Loss Ratio: %f'%cfg.MODEL.DISTILLATION_RATIO)

            loss += cfg.MODEL.DISTILLATION_RATIO * distillation_loss

        loss_extra = None
        if isinstance(loss, (list, tuple)):
            loss, loss_extra = loss

        misc.check_nan_losses(loss)
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        # for name, param in model.model.visual.tempx_blocks.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name}: {param.grad.norm():.4f}")
        #     else:
        #         print(f"{name}: No gradient")

        if cfg.SOLVER.CLIP_GRAD_VAL:
            grad_norm = torch.nn.utils.clip_grad_value_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_VAL
            )
        elif cfg.SOLVER.CLIP_GRAD_L2NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.SOLVER.CLIP_GRAD_L2NORM
            )
        else:
            grad_norm = optim.get_grad_norm_(model.parameters())

        model, update_param = contrastive_parameter_surgery(
            model, 
            cfg, 
            epoch_exact, 
            cur_iter
        )

        if update_param: 
            scaler.step(optimizer)
        scaler.update()

        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 
            for x in num_topks_correct
        ]

        top1_err, top5_err = None, None

        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]

        loss, grad_norm, top1_err, top5_err = (
            loss.item(),
            grad_norm.item(),
            top1_err.item(),
            top5_err.item(),
        )

        total_top1_err += top1_err * preds.size(0)
        total_top5_err += top5_err * preds.size(0)
        total_samples += preds.size(0)

        train_meter.update_stats(
            top1_err,
            top5_err,
            loss,
            lr,
            grad_norm,
            batch_size
            * max(
                cfg.NUM_GPUS, 1
            ),
            loss_extra,
        )

        train_meter.iter_toc()
        train_meter.log_iter_stats(cur_epoch, cur_iter)

        if cfg.DEVICE == 'cuda':
            torch.cuda.synchronize()
        train_meter.iter_tic()

    del inputs
    if cfg.DEVICE == 'cuda':
        torch.cuda.empty_cache()
    
    epoch_top1_err = total_top1_err / total_samples
    epoch_top5_err = total_top5_err / total_samples
    epoch_top1_acc = 100.0 - epoch_top1_err
    epoch_top5_acc = 100.0 - epoch_top5_err

    logger("---------------------------------------------------------------------------")
    logger(f"Epoch {cur_epoch+1} Training Results:")
    logger(f"Top-1 Error: {epoch_top1_err:.2f}% | Top-5 Error: {epoch_top5_err:.2f}%")
    logger(f"Top-1 Accuracy: {epoch_top1_acc:.2f}% | Top-5 Accruacy: {epoch_top5_acc:.2f}%")
    logger("---------------------------------------------------------------------------")

    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

@torch.no_grad()
def eval_epoch(loader, model, cur_epoch, cfg, val_meter, logger):
    model.eval()
    val_meter.iter_tic()

    total_top1_correct, total_top5_correct, total_samples = 0, 0, 0
    for cur_iter, (inputs, labels, index, time, meta) in tqdm(enumerate(loader), total=len(loader), desc="Val"):
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(cfg.DEVICE)
            else:
                inputs = inputs.to(cfg.DEVICE)
            labels = labels.to(cfg.DEVICE)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].to(cfg.DEVICE)
                else:
                    meta[key] = val.to(cfg.DEVICE)
            index = index.to(cfg.DEVICE)
            if cfg.DEVICE == "mps":
                time = time.float().to(cfg.DEVICE)    
            else:
                time = time.to(cfg.DEVICE)

        batch_size = (
            inputs[0][0].size(0)
            if isinstance(inputs[0], list)
            else inputs[0].size(0)
        )
        val_meter.data_toc()

        if cfg.MODEL.KEEP_RAW_MODEL and cfg.MODEL.RAW_MODEL_DISTILLATION:
            outputs = model(inputs)
            if len(outputs[0]) == 2:
                [preds, img_encode], [raw_pred, raw_img_encode] = outputs
            elif len(outputs[0]) == 3:
                [preds, img_encode, text_encode], [raw_pred, raw_img_encode, raw_text_encode] = outputs
        else:
            preds = model(inputs)
            if isinstance(preds, list):
                preds = preds[0]

        if cfg.DATA.IN22k_VAL_IN1K != "":
            preds = preds[:, :1000]

        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0 for x in num_topks_correct
        ]
        top1_err, top5_err = top1_err.item(), top5_err.item()
        val_meter.iter_toc()

        top1_correct, top5_correct = num_topks_correct
        total_top1_correct += top1_correct
        total_top5_correct += top5_correct
        total_samples += preds.size(0)

        val_meter.update_stats(
            top1_err,
            top5_err,
            batch_size
            * max(
                cfg.NUM_GPUS, 1
            ),  # If running  on CPU (cfg.NUM_GPUS == 1), use 1 to represent 1 CPU.
        )

        val_meter.update_predictions(preds, labels)
        val_meter.log_iter_stats(cur_epoch, cur_iter)
        val_meter.iter_tic()

    top1_acc = (total_top1_correct / total_samples) * 100
    top5_acc = (total_top5_correct / total_samples) * 100
    top1_err = 100 - top1_acc
    top5_err = 100 - top5_acc

    logger("---------------------------------------------------------------------------")
    logger(f"Epoch {cur_epoch+1} Val Results:")
    logger(f"Top-1 Error: {top1_err:.2f}% | Top-5 Error: {top5_err:.2f}%")
    logger(f"Top-1 Accuracy: {top1_acc:.2f}% | Top-5 Accruacy: {top5_acc:.2f}%")
    logger("---------------------------------------------------------------------------")

    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def train():
    args = parse_args()
    date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # output_path = "_".join([args.opts[args.opts.index("OUTPUT_DIR")+1], date])
    # args.opts[args.opts.index("OUTPUT_DIR")+1] = output_path

    if os.path.exists(args.opts[args.opts.index("OUTPUT_DIR")+1]):
        shutil.rmtree(args.opts[args.opts.index("OUTPUT_DIR")+1])

    for path_to_config in args.cfg_files:
        cfg = load_config(args, path_to_config)
        cfg = assert_and_infer_cfg(cfg)
    cfg.DEVICE = "cuda" if torch.cuda.is_available() else "mps"

    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    if cfg.DEVICE == 'mps':
        torch.mps.manual_seed(seed)
        torch.backends.mps.deterministic=True
        torch.backends.mps.benchmark = False
    elif cfg.DEVICE == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

    # cfg.OUTPUT_DIR = output_path
    logger = Logger(os.path.join(cfg.OUTPUT_DIR, "log.txt"))

    logger("\nCONFIGS=============================================================")
    logger("MODEL.MODEL_NAME"+":"+ str(cfg.MODEL.MODEL_NAME))
    logger("MODEL.DISTILLATION_RATIO"+":"+ str(cfg.MODEL.DISTILLATION_RATIO))
    logger("CONFIGS=============================================================\n")

    model = None
    if cfg.MODEL.MODEL_NAME == "TemporalClipVideo":
        model = TemporalClipVideo(cfg).to(cfg.DEVICE)
    else:
        model_config = cfg.MODEL.MODEL_NAME.split("_")[1]
        cfg.MODEL.VIL = len(model_config) == 2
        cfg.MODEL.ADD_SPATIAL_MODEL = model_config[0] in ["X", "S"]
        cfg.MODEL.ADD_TEMPORAL_MODEL = model_config[0] in ["X", "T"]
        model = SCAR(cfg).to(cfg.DEVICE)

    optimizer = construct_optimizer(model, cfg)

    scaler = None
    if cfg.DEVICE == "mps":
        scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)    
    else:
        scaler = torch.amp.GradScaler(cfg.DEVICE, enabled=cfg.TRAIN.MIXED_PRECISION)

    start_epoch = 0
    last_checkpoint = cu.get_last_checkpoint("./pretrained", task=cfg.TASK)
    if last_checkpoint is not None:
        logger("\nLoading pretrained model==========================================\n")
        checkpoint_epoch = cu.load_checkpoint(
            last_checkpoint,
            model,
            cfg.NUM_GPUS > 1,
            optimizer,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
        )
        start_epoch = checkpoint_epoch + 1

    train_loader = construct_loader(cfg, "train")
    val_loader = construct_loader(cfg, "val")

    logger("DATASET SIZE TRAIN: " + str(len(train_loader)*cfg.TRAIN.BATCH_SIZE))
    logger("DATASET SIZE VAL: " + str(len(val_loader)*cfg.TRAIN.BATCH_SIZE))

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)
    epoch_timer = EpochTimer()

    flops, params = 0.0, 0.0
    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        epoch_timer.epoch_tic()

        logger(f"\nEpoch [{cur_epoch+1}/{cfg.SOLVER.MAX_EPOCH}]")
        train_epoch(train_loader, model, optimizer, scaler, cur_epoch, cfg, train_meter, logger)
        
        _ = misc.aggregate_sub_bn_stats(model)

        # if cur_epoch == 0:
        #     eval_epoch(val_loader, model, cur_epoch, cfg, val_meter, logger)

        epoch_timer.epoch_toc()
        cu.save_checkpoint(
            cfg.OUTPUT_DIR,
            model,
            optimizer,
            cur_epoch,
            cfg,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
        )
        logger("\n")

        logger(
            f"Epoch {cur_epoch+1} takes {epoch_timer.last_epoch_time():.2f}s. Epochs\n \
            from {start_epoch+1} to {cur_epoch+1} take {epoch_timer.avg_epoch_time():.2f}s in average and\n \
            {epoch_timer.median_epoch_time():.2f}s in median."
        )

        logger(
            f"For epoch {cur_epoch+1}, each iteraction takes\n \
            {epoch_timer.last_epoch_time()/len(train_loader):.2f}s in average\n \
            From epoch {start_epoch+1} to {cur_epoch+1}, each iteraction takes\n \
            {epoch_timer.avg_epoch_time()/len(train_loader):.2f}s in average."
        )
        logger("==============================================================================\n\n")

    logger("TOP1 ERR: " + str(val_meter.min_top1_err))
    logger("TOP5 ERR: " + str(val_meter.min_top5_err))
    logger("TOP1 ACC: " + str(100 - val_meter.min_top1_err))
    logger("TOP5 ACC: " + str(100 - val_meter.min_top5_err))
        

if __name__ == '__main__':
    train()