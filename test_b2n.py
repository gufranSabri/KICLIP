import os
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch
from datasets.build import build_dataset
from datasets import utils
from utils.env import pathmgr
from utils.meters import TestMeter
from utils.parser import load_config, parse_args
import utils.checkpoint as cu
from config.defaults import assert_and_infer_cfg
from models.temporalclip_video_model import TemporalClipVideo
from models.scar import SCAR
from pprint import pprint


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

def perform_test(loader, model, test_meter, cfg):
    model.eval()
    test_meter.iter_tic()

    for cur_iter, (inputs, labels, video_idx, time, meta) in tqdm(enumerate(loader), total=len(loader)):
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    inputs[i] = inputs[i].to(cfg.DEVICE)
            else:
                inputs = inputs.to(cfg.DEVICE)

            labels = labels.to(cfg.DEVICE)
            video_idx = video_idx.to(cfg.DEVICE)
            for key, val in meta.items():
                if isinstance(val, (list,)): 
                    for i in range(len(val)):
                        val[i] = val[i].to(cfg.DEVICE)
                else:
                    meta[key] = val.to(cfg.DEVICE)

            test_meter.data_toc()

            preds = None
            with torch.no_grad():
                if cfg.MODEL.KEEP_RAW_MODEL and cfg.MODEL.ENSEMBLE_PRED:
                    preds, raw_preds = model(inputs)
                    preds = cfg.MODEL.ENSEMBLE_RAWMODEL_RATIO * raw_preds + (1 - cfg.MODEL.ENSEMBLE_RAWMODEL_RATIO) * preds
                else:
                    preds = model(inputs)
            torch.cuda.empty_cache()  

            if cfg.NUM_GPUS:
                preds = preds.cpu()
                labels = labels.cpu()
                video_idx = video_idx.cpu()

            test_meter.iter_toc()
            if not cfg.VIS_MASK.ENABLE:
                test_meter.update_stats(preds.detach(), labels.detach(), video_idx.detach())

            test_meter.log_iter_stats(cur_iter) 
            test_meter.iter_tic()

    if not cfg.DETECTION.ENABLE:
        all_preds = test_meter.video_preds.clone().detach()
        all_labels = test_meter.video_labels
        if cfg.NUM_GPUS:
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()

        if cfg.TEST.SAVE_RESULTS_PATH != "":
            save_path = os.path.join(cfg.OUTPUT_DIR, cfg.TEST.SAVE_RESULTS_PATH)

            with pathmgr.open(save_path, "wb") as f:
                pickle.dump([all_preds, all_labels], f)

    test_meter.finalize_metrics()
    return test_meter
            
def test():
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

    if len(cfg.TEST.NUM_TEMPORAL_CLIPS) == 0:
        cfg.TEST.NUM_TEMPORAL_CLIPS = [cfg.TEST.NUM_ENSEMBLE_VIEWS]

    test_meters = []
    for num_view in cfg.TEST.NUM_TEMPORAL_CLIPS:
        cfg.TEST.NUM_ENSEMBLE_VIEWS = num_view

        model = None
        if cfg.MODEL.MODEL_NAME == "FROSTER":
            model = TemporalClipVideo(cfg).to(cfg.DEVICE)
        if cfg.MODEL.MODEL_NAME == "SCAR_T":
            cfg.MODEL.VIL = False
            cfg.MODEL.LSTM = True
            model = SCAR(cfg).to(cfg.DEVICE)
        if cfg.MODEL.MODEL_NAME == "SCAR_S":
            cfg.MODEL.VIL = True
            cfg.MODEL.LSTM = False
            model = SCAR(cfg).to(cfg.DEVICE)
        if cfg.MODEL.MODEL_NAME == "SCAR_X":
            cfg.MODEL.VIL = True
            cfg.MODEL.LSTM = True
            model = SCAR(cfg).to(cfg.DEVICE)

        if not cfg.TEST.CUSTOM_LOAD:
            cu.load_test_checkpoint(cfg, model)

        if cfg.TEST.CUSTOM_LOAD:
            custom_load_file = cfg.TEST.CUSTOM_LOAD_FILE

            checkpoint = torch.load(custom_load_file, map_location='cpu')
            checkpoint_model = checkpoint['model_state']
            state_dict = model.state_dict()


            if 'module' in list(state_dict.keys())[0]:
                new_checkpoint_model = {} 
                for key, value in checkpoint_model.items():
                    new_checkpoint_model['module.' + key] = value
                checkpoint_model = new_checkpoint_model
            
            model.load_state_dict(checkpoint_model, strict=True)
            model = model.to(cfg.DEVICE)

            test_loader = construct_loader(cfg, "test")

            test_meter = TestMeter(
                test_loader.dataset.num_videos // (cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS),
                cfg.TEST.NUM_ENSEMBLE_VIEWS * cfg.TEST.NUM_SPATIAL_CROPS,
                cfg.MODEL.NUM_CLASSES
                if not cfg.TASK == "ssl" else cfg.CONTRASTIVE.NUM_CLASSES_DOWNSTREAM,
                len(test_loader),
                cfg.DATA.MULTI_LABEL,
                cfg.DATA.ENSEMBLE_METHOD,
            )

            test_meter = perform_test(test_loader, model, test_meter, cfg)
            test_meters.append(test_meter)

    for view, test_meter in zip(cfg.TEST.NUM_TEMPORAL_CLIPS, test_meters):
        print("View: {}".format(view))
        print("Top1 Acc: {}".format(test_meter.stats["top1_acc"]))
        print("Top5 Acc: {}".format(test_meter.stats["top5_acc"]))
        print("=====================================\n")

if __name__ == "__main__":
    test()