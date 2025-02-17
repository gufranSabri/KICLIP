ROOT=./
CKPT=./basetraining/B2N_k400_KICLIP
OUT_DIR=./basetraining/B2N_k400_KICLIP/testing
LOAD_CKPT_FILE=./basetraining/B2N_k400_KICLIP/wa_checkpoints/swa_2_22.pth

# TRAIN_FILE can be set as train_1.csv or train_2.csv or train_3.csv;
# TEST_FILE can be set as val_new.csv (base set) or test.csv (novel set).
# rephrased_file can be set as train_rephrased.json (base set) or test_rephrased.json (novel set)


B2N_k400_file=B2N_k400
TRAIN_FILE=train_s1.csv
VAL_FILE=val_new.csv
TEST_FILE=val_new.csv
rephrased_file=train_rephrased.json

cd $ROOT

CUDA_VISIBLE_DEVICES=0 python test_b2n.py \
    --cfg config_files/Kinetics/KICLIP_vitb16_8x16_STAdapter_K400.yaml \
    --opts DATA.PATH_TO_DATA_DIR ./zs_label_db/B2N_k400 \
    TRAIN_FILE $TRAIN_FILE \
    VAL_FILE $VAL_FILE \
    TEST_FILE $TEST_FILE \
    DATA.PATH_PREFIX /data/Hamzah/kinetics-400/raw/kinetics-dataset/k400 \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE ./zs_label_db/B2N_k400/$rephrased_file \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 4 \
    NUM_GPUS 1 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 200 \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL False \
    TEST.CLIP_ORI_PATH ~/.cache/clip/ViT-B-16.pt \
    DATA_LOADER.NUM_WORKERS 4 \


B2N_k400_file=B2N_k400
TRAIN_FILE=train_s1.csv
VAL_FILE=val_new.csv
TEST_FILE=test_1.csv
rephrased_file=test_rephrased.json

cd $ROOT

CUDA_VISIBLE_DEVICES=0 python test_b2n.py \
    --cfg config_files/Kinetics/KICLIP_vitb16_8x16_STAdapter_K400.yaml \
    --opts DATA.PATH_TO_DATA_DIR ./zs_label_db/B2N_k400 \
    TRAIN_FILE $TRAIN_FILE \
    VAL_FILE $VAL_FILE \
    TEST_FILE $TEST_FILE \
    DATA.PATH_PREFIX /data/Hamzah/kinetics-400/raw/kinetics-dataset/k400 \
    DATA.PATH_LABEL_SEPARATOR , \
    DATA.INDEX_LABEL_MAPPING_FILE ./zs_label_db/B2N_k400/$rephrased_file \
    TRAIN.ENABLE False \
    OUTPUT_DIR $OUT_DIR \
    TEST.BATCH_SIZE 4 \
    NUM_GPUS 1 \
    DATA.DECODING_BACKEND "pyav" \
    MODEL.NUM_CLASSES 200 \
    TEST.CUSTOM_LOAD True \
    TEST.CUSTOM_LOAD_FILE $LOAD_CKPT_FILE \
    TEST.SAVE_RESULTS_PATH temp.pyth \
    TEST.NUM_ENSEMBLE_VIEWS 3 \
    TEST.NUM_SPATIAL_CROPS 1 \
    TEST.PATCHING_MODEL False \
    TEST.CLIP_ORI_PATH ~/.cache/clip/ViT-B-16.pt \
    DATA_LOADER.NUM_WORKERS 4 \