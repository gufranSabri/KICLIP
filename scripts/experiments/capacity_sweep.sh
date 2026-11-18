#!/bin/bash
# CGAS capacity (pooling width) sweep for the capacity figure: HMDB-51 b2n, from the best config
# (configs/b2n/best/hmdb51.yaml, written by scripts/pick_best_config.py -- run that first), mean
# pooling throughout, cgas_layers=last_1, cgas_grad_log=true so history.json carries the per-step
# backbone/head gradient-norm ratio. Width 768 on a 768-wide CLS is the no-pooling identity.
# Feeds scripts/analysis/dcgd_figures.py (Figure 4b: capacity bottleneck).
# Run command-by-command in an interactive SLURM terminal (after scripts/troubleshooting.sh's STEP 0/1/1b).

BASE="--config configs/b2n/best/hmdb51.yaml --set model.cgas_layers=last_1 --set model.cgas_pool_modes=[mean] --set model.cgas_grad_log=true"
if [[ ! -f configs/b2n/best/hmdb51.yaml ]]; then
    echo "missing configs/b2n/best/hmdb51.yaml; run 'python scripts/pick_best_config.py' first" >&2
    exit 1
fi

run() {
    local name=$1; shift
    local dir="runs/capacity_sweep/$name"
    [[ -f "$dir/averaged.pt" ]] || python kiclip.py "$@" train --run-name "capacity_sweep/$name" || return
    grep -qs '"HM"' "$dir/results.json" || python kiclip.py "$@" test --checkpoint "$dir/averaged.pt"
}

for width in 16 32 64 128 256 768; do
    run seg${width} $BASE --set model.cgas_segments=$width
done

# ── COLLECT ─────────────────────────────────────────────────────────────────
python scripts/tables/runs_csv.py --runs-root runs/capacity_sweep
