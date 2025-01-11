ROOT=./
CKPT=./basetraining/B2N_ucf101_scar_2025-01-11_17-38-10
OUT_DIR=./basetraining/B2N_ucf101_scar_2025-01-11_17-38-10/testing
LOAD_CKPT_FILE=./basetraining/B2N_ucf101_scar_2025-01-11_17-38-10/wa_checkpoints/swa_2_22.pth

# TEST_FILE can be set as val.csv (base set) or test.csv (novel set).
# rephrased_file can be set as train_rephrased.json (base set) or test_rephrased.json (novel set)
# NUM_CLASSES can be set as 51 (base set) or 50 (novel set)
TRAIN_FILE=train_s1.csv
VAL_FILE=val_1.csv
TEST_FILE=test_raw.csv
rephrased_file=test_rephrased.json
NUM_CLASSES=50

cd $ROOT

CUDA_VISIBLE_DEVICES=0 python3 test_b2n.py \
    --cfg ./config_files/Kinetics/SCAR_vitb16_8x16_STAdapter_UCF101.yaml \
    --opts DATA.PATH_TO_DATA_DIR ./zs_label_db/B2N_ucf101 \
    TRAIN_FILE $TRAIN_FILE \
    VAL_FILE $VAL_FILE \
    TEST_FILE $TEST_FILE \
    DATA.PATH_PREFIX ./data/ucf101 \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE ./zs_label_db/B2N_ucf101/$rephrased_file \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 4 \
    NUM_GPUS 1 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES $NUM_CLASSES \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL False \
    TEST.CLIP_ORI_PATH ~/.cache/clip/ViT-B-16.pt \
    DATA_LOADER.NUM_WORKERS 4 \

TRAIN_FILE=train_s1.csv
VAL_FILE=val_1.csv
TEST_FILE=val_1.csv
rephrased_file=train_rephrased.json
NUM_CLASSES=51

cd $ROOT

CUDA_VISIBLE_DEVICES=0 python3 test_b2n.py \
    --cfg ./config_files/Kinetics/SCAR_vitb16_8x16_STAdapter_UCF101.yaml \
    --opts DATA.PATH_TO_DATA_DIR ./zs_label_db/B2N_ucf101 \
    TRAIN_FILE $TRAIN_FILE \
    VAL_FILE $VAL_FILE \
    TEST_FILE $TEST_FILE \
    DATA.PATH_PREFIX ./data/ucf101 \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE ./zs_label_db/B2N_ucf101/$rephrased_file \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 4 \
    NUM_GPUS 1 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES $NUM_CLASSES \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL False \
    TEST.CLIP_ORI_PATH ~/.cache/clip/ViT-B-16.pt \
    DATA_LOADER.NUM_WORKERS 4 \