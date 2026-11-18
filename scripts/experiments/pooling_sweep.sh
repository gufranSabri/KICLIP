#!/bin/bash
# CGAS pooling-mode sweep for the alignment figure: HMDB-51 b2n, from the best config
# (configs/b2n/best/hmdb51.yaml, written by scripts/pick_best_config.py -- run that first), with
# cgas_layers=last_1 (clean single-block readout) and cgas_segments=64 fixed, varying only pool mode.
# Feeds scripts/analysis/gradient_alignment.py (Figure 4a: alignment by pooling mode).
# Run command-by-command in an interactive SLURM terminal (after scripts/troubleshooting.sh's STEP 0/1/1b).

BASE="--config configs/b2n/best/hmdb51.yaml --set model.cgas_layers=last_1 --set model.cgas_segments=64"
if [[ ! -f configs/b2n/best/hmdb51.yaml ]]; then
    echo "missing configs/b2n/best/hmdb51.yaml; run 'python scripts/pick_best_config.py' first" >&2
    exit 1
fi

run() {
    local name=$1; shift
    local dir="runs/pooling_sweep/$name"
    [[ -f "$dir/averaged.pt" ]] || python kiclip.py "$@" train --run-name "pooling_sweep/$name" || return
    grep -qs '"HM"' "$dir/results.json" || python kiclip.py "$@" test --checkpoint "$dir/averaged.pt"
}

for mode in mean dropout max min; do
    run $mode $BASE --set model.cgas_pool_modes=[$mode]
done

# ── COLLECT ─────────────────────────────────────────────────────────────────
python scripts/tables/runs_csv.py --runs-root runs/pooling_sweep
