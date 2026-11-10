#!/bin/bash
# Base-to-novel: for each method x dataset x seed on disk, train on the 16-shot base split, test on base and novel.
#
#   bash scripts/experiments/b2n.sh                                                    # RERUN=1 to retrain/retest
#   METHODS="kiclip kiclip_no_aux kiclip_no_injection froster" bash scripts/experiments/b2n.sh
#   LAUNCH="torchrun --standalone --nproc_per_node=4" bash scripts/experiments/b2n.sh
#   python scripts/tables/b2n.py

cd "$(dirname "$0")/../.." || exit 1

RERUN=${RERUN:-0}
LAUNCH=${LAUNCH:-python}
read -ra METHODS <<< "${METHODS:-kiclip froster}"
DATASETS=(
    k400
    hmdb51
    ucf101
    ssv2
)

for METHOD in "${METHODS[@]}"; do
    for DS in "${DATASETS[@]}"; do
        for TRAIN in splits/b2n/"$DS"/train_s*.csv; do
            SEED=$(basename "$TRAIN" .csv)
            SEED=${SEED#train_}
            RUN=b2n_${DS}_${SEED}_${METHOD}
            CONFIGS=(--config "configs/b2n/$DS.yaml" --config "configs/methods/$METHOD.yaml")

            echo "========================================"
            echo "Method: $METHOD | Dataset: $DS | Seed: $SEED"
            echo "========================================"

            if [[ "$RERUN" != "1" && -f "runs/$RUN/averaged.pt" ]]; then
                echo "  [skip] train runs/$RUN (averaged.pt exists; RERUN=1 to force)"
            elif ! $LAUNCH kiclip.py "${CONFIGS[@]}" train --run-name "$RUN" --set "data.train=b2n/$DS/train_$SEED"; then
                echo "  [fail] train runs/$RUN; skipping its test"
                continue
            fi

            if [[ "$RERUN" != "1" && -f "runs/$RUN/test_b2n_${DS}_base.json" && -f "runs/$RUN/test_b2n_${DS}_novel.json" ]]; then
                echo "  [skip] test runs/$RUN (base and novel results exist; RERUN=1 to force)"
            else
                $LAUNCH kiclip.py "${CONFIGS[@]}" test --checkpoint "runs/$RUN/averaged.pt"
            fi
        done
    done
done
