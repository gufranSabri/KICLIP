#!/bin/bash
# Cross-dataset: for each method, train on all 400 K400 classes, test zero-shot on UCF101 / HMDB51 (full + 3 splits) and K600.
#
#   bash scripts/experiments/cross_dataset.sh                                                    # RERUN=1 to retrain/retest
#   METHODS="kiclip froster" bash scripts/experiments/cross_dataset.sh
#   LAUNCH="torchrun --standalone --nproc_per_node=4" bash scripts/experiments/cross_dataset.sh
#   python scripts/tables/cross_dataset.py

cd "$(dirname "$0")/../.." || exit 1

RERUN=${RERUN:-0}
LAUNCH=${LAUNCH:-python}
read -ra METHODS <<< "${METHODS:-kiclip froster}"
SPLITS=(
    cross/ucf101/full cross/ucf101/split1 cross/ucf101/split2 cross/ucf101/split3
    cross/hmdb51/full cross/hmdb51/split1 cross/hmdb51/split2 cross/hmdb51/split3
    cross/k600/split1 cross/k600/split2 cross/k600/split3
)

for METHOD in "${METHODS[@]}"; do
    RUN=cross_k400_${METHOD}
    CONFIGS=(--config configs/cross/k400.yaml --config "configs/methods/$METHOD.yaml")

    echo "========================================"
    echo "Method: $METHOD | Train: K400 (400 classes) | Run: $RUN"
    echo "========================================"

    if [[ "$RERUN" != "1" && -f "runs/$RUN/averaged.pt" ]]; then
        echo "  [skip] train runs/$RUN (averaged.pt exists; RERUN=1 to force)"
    elif ! $LAUNCH kiclip.py "${CONFIGS[@]}" train --run-name "$RUN"; then
        echo "  [fail] train runs/$RUN; skipping its tests"
        continue
    fi

    TODO=()
    for SPLIT in "${SPLITS[@]}"; do
        if [[ "$RERUN" != "1" && -f "runs/$RUN/test_${SPLIT//\//_}.json" ]]; then
            echo "  [skip] test $SPLIT (result exists; RERUN=1 to force)"
        else
            TODO+=("$SPLIT")
        fi
    done

    if (( ${#TODO[@]} )); then
        $LAUNCH kiclip.py "${CONFIGS[@]}" test --checkpoint "runs/$RUN/averaged.pt" --splits "${TODO[@]}"
    fi
done
