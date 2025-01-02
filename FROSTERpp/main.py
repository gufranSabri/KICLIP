from pprint import pprint
from utils.parser import load_config, parse_args
from config.defaults import assert_and_infer_cfg
from models.temporalclip_video_model import TemporalClipVideo
import torch
import random
import numpy as np


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
    

def main():
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

    model = TemporalClipVideo(cfg)
    optimizer = construct_optimizer(model, cfg)
    
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.TRAIN.MIXED_PRECISION)
    start_epoch = 0


if __name__ == '__main__':
    main()