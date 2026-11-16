#!/bin/bash
# cgas_ce_weight sweep, all four b2n datasets: UCF101 w=1.0 scored 85.66 HM (novel 77.4), well below
# plain froster's 88.33; w=0.1 fixed it to 88.77 HM (novel 82.18). This checks whether one weight
# works across datasets or each needs its own. K400/SSv2 fix seed=0 on their default train_s1 split
# (their own 3-run axis is the split, not the seed -- see scripts/experiments/run_b2n.sh). Run command-by-command
# in an interactive SLURM terminal.

run() {
    local name=$1; shift
    local dir="runs/$name"
    [[ -f "$dir/averaged.pt" ]] || python kiclip.py "$@" train --run-name "$name" || return
    grep -qs '"HM"' "$dir/results.json" || python kiclip.py "$@" test --checkpoint "$dir/averaged.pt"
}

# for ds in hmdb51 ucf101 k400; do
for ds in k400; do
    BASE="--config configs/b2n/default/${ds}.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml"
    # for w in 0.02 0.05 0.1 0.25 0.5 1.0; do
    for w in 0.1 0.25 0.5 1.0; do
        run ceweight_sweep/${ds}_w${w} $BASE --set model.cgas_ce_weight=$w --set seed=0
    done
done

# ── COLLECT ─────────────────────────────────────────────────────────────────
python scripts/tables/runs_csv.py --runs-root runs --out docs/tables/runs.csv
