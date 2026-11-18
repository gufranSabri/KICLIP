#!/bin/bash
# CGAS target ablation for Figure 5: HMDB-51 b2n, from the best config (configs/b2n/best/hmdb51.yaml,
# written by scripts/pick_best_config.py -- run that first), varying only model.cgas_target.
# cgas_target=both (CE on base labels + KL to the frozen teacher) is the main method's own setting,
# already trained as runs/main/hmdb51_cgas -- not repeated here.
# Run command-by-command in an interactive SLURM terminal (after scripts/troubleshooting.sh's STEP 0/1/1b).

BASE="--config configs/b2n/best/hmdb51.yaml"
if [[ ! -f configs/b2n/best/hmdb51.yaml ]]; then
    echo "missing configs/b2n/best/hmdb51.yaml; run 'python scripts/pick_best_config.py' first" >&2
    exit 1
fi

run() {
    local name=$1; shift
    local dir="runs/cgas_target_ablation/$name"
    [[ -f "$dir/averaged.pt" ]] || python kiclip.py "$@" train --run-name "cgas_target_ablation/$name" || return
    grep -qs '"HM"' "$dir/results.json" || python kiclip.py "$@" test --checkpoint "$dir/averaged.pt"
}

run label $BASE --set model.cgas_target=label
run teacher $BASE --set model.cgas_target=teacher

# ── COLLECT ─────────────────────────────────────────────────────────────────
python scripts/tables/runs_csv.py --runs-root runs/cgas_target_ablation
