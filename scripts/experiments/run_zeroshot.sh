#!/bin/bash
# Zero-shot transfer: train once on K400, test zero-shot on HMDB51/UCF101 (full + 3 splits each) and
# K600 (3 splits) -- MSTA Table 4's protocol (train on K400, no further tuning). No seeds: this is a
# single training run, scored on 3 splits per target dataset to get the mean reported there. Uses
# configs/cross/best/k400.yaml (the b2n/k400 best model hyperparameters, on cross's own full-K400 data).
#
# PERCENT sets what fraction of K400's training videos to actually train on (data.train_fraction),
# seeded by the config's own seed so the subset is fixed across reruns. Defaults to 5 (i.e. 5%), since
# training on all 400 K400 classes takes a long time; raise it for a more faithful run once you have it.
#
#   bash scripts/experiments/run_zeroshot.sh            # 5% of K400
#   PERCENT=20 bash scripts/experiments/run_zeroshot.sh # 20% of K400
#   PERCENT=100 bash scripts/experiments/run_zeroshot.sh # the full K400 training set

run() {
    local name=$1; shift
    local dir="runs/$name"
    [[ -f "$dir/averaged.pt" ]] || python kiclip.py "$@" train --run-name "$name" || return
    grep -qs '"ucf101/full"' "$dir/results.json" || python kiclip.py "$@" test --checkpoint "$dir/averaged.pt"
}

PERCENT=${PERCENT:-5}

best="configs/cross/best/k400.yaml"
if [[ ! -f "$best" ]]; then
    echo "missing $best; copy configs/b2n/best/k400.yaml's model: block onto cross/default/k400.yaml's data first" >&2
    exit 1
fi
run zeroshot/k400_cgas_p${PERCENT} --config "$best" --set data.train_fraction=$(python3 -c "print($PERCENT/100)")

# ── COLLECT ─────────────────────────────────────────────────────────────────
python scripts/tables/runs_csv.py --runs-root runs --out docs/tables/runs.csv
