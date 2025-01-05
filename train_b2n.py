from pprint import pprint
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from utils.parser import load_config, parse_args
import utils.misc as misc
import utils.metrics as metrics
import utils.checkpoint as cu
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
    )

    return loader

def train_epoch(loader, model, optimizer, scaler, cur_epoch, cfg):
    model.train()
    data_size = len(loader)

    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")
    if cfg.MODEL.KEEP_RAW_MODEL:
        raw_clip_params = {}
        for n, p in model.named_parameters():
            if 'raw_model' in n:
                p.requires_grad = False
                raw_clip_params[n] = p

    total_loss = 0.0
    total_grad_norm = 0.0
    total_top1_err = 0.0
    total_top5_err = 0.0
    
    for cur_iter, (inputs, labels, index, time, meta) in tqdm(enumerate(loader), total=data_size):
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
                time = time.float().to(cfg.DEVICE)

            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].to(cfg.DEVICE)
                else:
                    meta[key] = val.to(cfg.DEVICE)

        epoch_exact = cur_epoch + float(cur_iter) / data_size
        lr = optim.get_epoch_lr(epoch_exact, cfg)
        optim.set_lr(optimizer, lr)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=cfg.TRAIN.MIXED_PRECISION):
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

            loss += cfg.MODEL.DISTILLATION_RATIO * distillation_loss

        if isinstance(loss, (list, tuple)): 
            loss, _ = loss

        misc.check_nan_losses(loss)
        if perform_backward:
            scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        if cfg.SOLVER.CLIP_GRAD_L2NORM:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                cfg.SOLVER.CLIP_GRAD_L2NORM
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

        total_loss += loss.item()
        total_grad_norm += grad_norm.item()
        total_top1_err += top1_err
        total_top5_err += top5_err

    avg_loss = total_loss / data_size
    avg_grad_norm = total_grad_norm / data_size
    avg_top1_err = total_top1_err / data_size
    avg_top5_err = total_top5_err / data_size

    print(f"Epoch {cur_epoch+1} Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Average Gradient Norm: {avg_grad_norm:.4f}")
    print(f"  Top-1 Error: {avg_top1_err:.2f}%")
    print(f"  Top-5 Error: {avg_top5_err:.2f}%")

    del inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@torch.no_grad()
def eval_epoch(loader, model, cur_epoch, cfg):
    model.eval()
    loss_fun = losses.get_loss_func(cfg.MODEL.LOSS_FUNC)(reduction="mean")

    total_loss = 0.0
    total_top1_err = 0.0
    total_top5_err = 0.0
    num_samples = 0

    for cur_iter, (inputs, labels, index, time, meta) in tqdm(enumerate(loader), total=len(loader)):
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
            time = time.float().to(cfg.DEVICE)

        # Model inference
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

        # Compute loss
        loss = loss_fun(preds, labels)
        if isinstance(loss, (list, tuple)):
            loss = loss[0]
        total_loss += loss.item() * labels.size(0)

        # Top-k error calculation
        if cfg.DATA.IN22k_VAL_IN1K != "":
            preds = preds[:, :1000]

        num_topks_correct = metrics.topks_correct(preds, labels, (1, 5))
        top1_err, top5_err = [
            (1.0 - x / preds.size(0)) * 100.0
            for x in num_topks_correct
        ]
        total_top1_err += top1_err * labels.size(0)
        total_top5_err += top5_err * labels.size(0)

        num_samples += labels.size(0)

    avg_loss = total_loss / num_samples
    avg_top1_err = total_top1_err / num_samples
    avg_top5_err = total_top5_err / num_samples

    print(f"Evaluation Epoch {cur_epoch+1} Summary:")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Top-1 Error: {avg_top1_err:.2f}%")
    print(f"  Top-5 Error: {avg_top5_err:.2f}%")


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

    model = TemporalClipVideo(cfg).to(cfg.DEVICE)
    optimizer = construct_optimizer(model, cfg)
    
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    start_epoch = 0

    train_loader = construct_loader(cfg, "train")
    val_loader = construct_loader(cfg, "val")

    for cur_epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH):
        print(f"Epoch [{cur_epoch+1}/{cfg.SOLVER.MAX_EPOCH}]")
        train_epoch(train_loader, model, optimizer, scaler, cur_epoch, cfg)
        eval_epoch(val_loader, model, cur_epoch, cfg)
        print("====================================================================================\n")

        cu.save_checkpoint(
            cfg.OUTPUT_DIR,
            model,
            optimizer,
            cur_epoch,
            cfg,
            scaler if cfg.TRAIN.MIXED_PRECISION else None,
        )

if __name__ == '__main__':
    train()