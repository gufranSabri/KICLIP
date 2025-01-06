import pprint
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
import models.losses as losses
from datasets import utils
from datasets.build import build_dataset
import models.optimizer as optim
from models.contrastive import (
    contrastive_forward,
    contrastive_parameter_surgery,
)

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

def train_epoch(loader, model, optimizer, scaler, cur_epoch, cfg, train_meter):
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

        optimizer.zero_grad()

        with torch.amp.autocast(cfg.DEVICE, enabled=cfg.TRAIN.MIXED_PRECISION):
            perform_backward = True
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
            #     print('Distillation Loss: %.8f'%distillation_loss.item())
            #     print('Distillation Loss Ratio: %f'%cfg.MODEL.DISTILLATION_RATIO)

            loss += cfg.MODEL.DISTILLATION_RATIO * distillation_loss

        loss_extra = None
        if isinstance(loss, (list, tuple)):
            loss, loss_extra = loss

        misc.check_nan_losses(loss)
        if perform_backward:
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

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
        torch.cuda.synchronize()
        train_meter.iter_tic()

    del inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    epoch_top1_err = total_top1_err / total_samples
    epoch_top5_err = total_top5_err / total_samples
    epoch_top1_acc = 100.0 - epoch_top1_err
    epoch_top5_acc = 100.0 - epoch_top5_err

    print("---------------------------------------------------------------------------")
    print(f"Epoch {cur_epoch} Training Results:")
    print(f"Top-1 Error: {epoch_top1_err:.2f}% | Top-5 Error: {epoch_top5_err:.2f}%")
    print(f"Top-1 Accuracy: {epoch_top1_acc:.2f}% | Top-5 Accruacy: {epoch_top5_acc:.2f}%")
    print("---------------------------------------------------------------------------")

    train_meter.log_epoch_stats(cur_epoch)
    train_meter.reset()

@torch.no_grad()
def eval_epoch(loader, model, cur_epoch, cfg, val_meter):
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

    print("---------------------------------------------------------------------------")
    print(f"Epoch {cur_epoch} Training Results:")
    print(f"Top-1 Error: {top1_err:.2f}% | Top-5 Error: {top5_err:.2f}%")
    print(f"Top-1 Accuracy: {top1_acc:.2f}% | Top-5 Accruacy: {top5_acc:.2f}%")
    print("---------------------------------------------------------------------------")

    val_meter.log_epoch_stats(cur_epoch)
    val_meter.reset()


def train():
    args = parse_args()
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

    print("\nCONFIGS=============================================================")
    print("MULTIGRID.SHORT_CYCLE", cfg.MULTIGRID.SHORT_CYCLE)
    print("MULTIGRID.LONG_CYCLE", cfg.MULTIGRID.LONG_CYCLE)
    print("BN.NORM_TYPE", cfg.BN.NORM_TYPE)
    print("NUM_GPUS", cfg.NUM_GPUS)
    print("TRAIN.CUSTOM_LOAD", cfg.TRAIN.CUSTOM_LOAD)
    print("SOLVER.LAYER_DECAY", cfg.SOLVER.LAYER_DECAY)
    print("MODEL.FINETUNE_FACTOR", cfg.MODEL.FINETUNE_FACTOR)
    print("MODEL.ADAPT_FINETUNE_FACTOR", cfg.MODEL.ADAPT_FINETUNE_FACTOR)
    print("MODEL.DEFAULT_FINETUNE_FACTOR", cfg.MODEL.DEFAULT_FINETUNE_FACTOR)
    print("MODEL.MLP_FINETUNE_FACTOR", cfg.MODEL.MLP_FINETUNE_FACTOR)
    print("MODEL.EXPERT_FINETUNE_FACTOR", cfg.MODEL.EXPERT_FINETUNE_FACTOR)
    print("SOLVER.OPTIMIZING_METHOD", cfg.SOLVER.OPTIMIZING_METHOD)
    print("SOLVER.LARS_ON", cfg.SOLVER.LARS_ON)
    print("TRAIN.MIXED_PRECISION", cfg.TRAIN.MIXED_PRECISION)
    print("TRAIN.AUTO_RESUME", cfg.TRAIN.AUTO_RESUME)
    print("DETECTION.ENABLE", cfg.DETECTION.ENABLE)
    print("AUG.NUM_SAMPLE", cfg.AUG.NUM_SAMPLE)
    print("DATA.TRAIN_CROP_NUM_TEMPORAL", cfg.DATA.TRAIN_CROP_NUM_TEMPORAL)
    print("DATA.TRAIN_CROP_NUM_SPATIAL", cfg.DATA.TRAIN_CROP_NUM_SPATIAL)
    print("MODEL.MODEL_NAME", cfg.MODEL.MODEL_NAME)
    print("MIXUP.ENABLE", cfg.MIXUP.ENABLE)
    print("BN.USE_PRECISE_STATS", cfg.BN.USE_PRECISE_STATS)
    print("TASK", cfg.TASK)
    print("CONTRASTIVE.KNN_ON", cfg.CONTRASTIVE.KNN_ON)
    print("TENSORBOARD.ENABLE", cfg.TENSORBOARD.ENABLE)
    print("NUM_SHARDS", cfg.NUM_SHARDS)
    print("VAL_MODE", cfg.VAL_MODE)
    print("TRAIN.EWC_SET", cfg.TRAIN.EWC_SET)
    print("DATA.LOADER_CHUNK_SIZE", cfg.DATA.LOADER_CHUNK_SIZE)
    print("TRAIN.ZS_CONS", cfg.TRAIN.ZS_CONS)
    print("TRAIN.CLIP_ORI_PATH", cfg.TRAIN.CLIP_ORI_PATH)
    print("MODEL.FROZEN_BN", cfg.MODEL.FROZEN_BN)
    print("MODEL.LOSS_FUNC", cfg.MODEL.LOSS_FUNC)
    print("TRAIN.LINEAR_CONNECT_CLIMB", cfg.TRAIN.LINEAR_CONNECT_CLIMB)
    print("MODEL.KEEP_RAW_MODEL", cfg.MODEL.KEEP_RAW_MODEL)
    print("MODEL.RAW_MODEL_DISTILLATION", cfg.MODEL.RAW_MODEL_DISTILLATION)
    print("MASK.ENABLE", cfg.MASK.ENABLE)
    print("MODEL.RECORD_ROUTING", cfg.MODEL.RECORD_ROUTING)
    print("SOLVER.CLIP_GRAD_VAL", cfg.SOLVER.CLIP_GRAD_VAL)
    print("SOLVER.CLIP_GRAD_L2NORM", cfg.SOLVER.CLIP_GRAD_L2NORM)
    print("DATA.MULTI_LABEL", cfg.DATA.MULTI_LABEL)
    print("DATA.IN22k_VAL_IN1K", cfg.DATA.IN22k_VAL_IN1K)
    print("CONFIGS=============================================================\n")

    model = TemporalClipVideo(cfg).to(cfg.DEVICE)
    optimizer = construct_optimizer(model, cfg)
    
    scaler = torch.amp.GradScaler(cfg.DEVICE, enabled=cfg.TRAIN.MIXED_PRECISION)
    last_checkpoint = cu.get_last_checkpoint("./pretrained", task=cfg.TASK)

    start_epoch = 0
    if last_checkpoint is not None:
        print("\nLoading pretrained model==========================================\n")
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

    raw_batch_size = cfg.TRAIN.BATCH_SIZE
    raw_mixup = cfg.MIXUP.ENABLE
    cfg.TRAIN.BATCH_SIZE = cfg.TRAIN.BATCH_SIZE // 2
    cfg.MIXUP.ENABLE = False
    fisher_loader = construct_loader(cfg, "train")
    cfg.TRAIN.BATCH_SIZE = raw_batch_size
    cfg.MIXUP.ENABLE = raw_mixup

    print("DATASET SIZE TRAIN", len(train_loader)*cfg.TRAIN.BATCH_SIZE)
    print("DATASET SIZE VAL", len(val_loader)*cfg.TRAIN.BATCH_SIZE)

    train_meter = TrainMeter(len(train_loader), cfg)
    val_meter = ValMeter(len(val_loader), cfg)
    # epoch_timer = EpochTimer()

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        print(f"\nEpoch [{cur_epoch+1}/{cfg.SOLVER.MAX_EPOCH}]")
        train_epoch(train_loader, model, optimizer, scaler, cur_epoch, cfg, train_meter)
        
        # epoch_timer.epoch_toc()
        _ = misc.aggregate_sub_bn_stats(model)

        eval_epoch(val_loader, model, cur_epoch, cfg, val_meter)

        cu.save_checkpoint(
            cfg.OUTPUT_DIR,
            model,
            optimizer,
            cur_epoch,
            cfg,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
        )
        print("==============================================================================\n\n")

    print("TOP1 ERR: ", val_meter.min_top1_err)
    print("TOP5 ERR: ", val_meter.min_top5_err)
    print("TOP1 ACC: ", 100 - val_meter.min_top1_err)
    print("TOP5 ACC: ", 100 - val_meter.min_top5_err)
        

if __name__ == '__main__':
    train()