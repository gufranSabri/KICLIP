ROOT=./
CKPT=./

# TRAIN_FILE can be set as train_1.csv or train_2.csv or train_3.csv;

TRAIN_FILE=train.csv
VAL_FILE=val.csv
TEST_FILE=test.csv

cd $ROOT

CUDA_VISIBLE_DEVICES=1 python3 train_b2n.py \
    --cfg ./config_files/Kinetics/KICLIP_vitb16_8x16_STAdapter_HMDB51.yaml \
    --opts DATA.PATH_TO_DATA_DIR ./zs_label_db/B2N_hmdb \
    DATA.PATH_PREFIX ./data/hmdb51/videos \
    TRAIN_FILE $TRAIN_FILE \
    VAL_FILE $VAL_FILE \
    TEST_FILE $TEST_FILE \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE ./zs_label_db/B2N_hmdb/train_rephrased.json \
    TRAIN.ENABLE True \
    OUTPUT_DIR $CKPT/basetraining/B2N_hmdb51_KICLIP \
    TRAIN.BATCH_SIZE 4 \
    TEST.BATCH_SIZE 4 \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    NUM_GPUS 1 \
    SOLVER.MAX_EPOCH 12 \
    SOLVER.WARMUP_EPOCHS 2.0 \
    SOLVER.BASE_LR 3.33e-6 \
    SOLVER.WARMUP_START_LR 3.33e-8 \
    SOLVER.COSINE_END_LR 3.33e-8 \
    TRAIN.MIXED_PRECISION True \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 26 \
    MIXUP.ENABLE False \
    AUG.ENABLE False \
    AUG.NUM_SAMPLE 1 \
    TRAIN.EVAL_PERIOD 1 \
    TRAIN.CHECKPOINT_PERIOD 1 \
    MODEL.LOSS_FUNC soft_cross_entropy \
    TRAIN.LINEAR_CONNECT_CLIMB False \
    TRAIN.CLIP_ORI_PATH /root/.cache/clip/ViT-B-16.pt \
    TRAIN.LINEAR_CONNECT_LOSS_RATIO 0.0 \
    MODEL.RAW_MODEL_DISTILLATION True \
    MODEL.KEEP_RAW_MODEL True \
    MODEL.DISTILLATION_RATIO 2.0