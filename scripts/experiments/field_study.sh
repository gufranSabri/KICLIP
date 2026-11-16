#!/bin/bash
# CGAS pooling study: HMDB-51 b2n, from the best config (configs/b2n/best/hmdb51.yaml, written by
# scripts/pick_best_config.py -- run that first), with cgas_layers=last_1 (clean single-block readout).
# Run command-by-command in an interactive SLURM terminal (after scripts/troubleshooting.sh's STEP 0/1/1b).
#
# TABLE1: pool mode (mean, dropout-mean, no pool) x segments (16, 32, 128, 256). "No pool" only makes
# sense at segments=768 on a 768-wide CLS with mode=mean -- an exact identity (pool_segments reshapes
# to (...,768,1).mean(-1)), so it is one extra run outside the segments grid, not a 4th grid row.
#
# TABLE2: dropout_p (0.1, 0.5, 0.7) x segments (16, 32, 128, 256), mode=dropout throughout.

BASE="--config configs/b2n/best/hmdb51.yaml --set model.cgas_layers=last_1"
if [[ ! -f configs/b2n/best/hmdb51.yaml ]]; then
    echo "missing configs/b2n/best/hmdb51.yaml; run 'python scripts/pick_best_config.py' first" >&2
    exit 1
fi

run() {
    local name=$1; shift
    local dir="runs/field_study/$name"
    [[ -f "$dir/averaged.pt" ]] || python kiclip.py "$@" train --run-name "field_study/$name" || return
    grep -qs '"HM"' "$dir/results.json" || python kiclip.py "$@" test --checkpoint "$dir/averaged.pt"
}

# ── TABLE1: pool mode x segments ────────────────────────────────────────────
for segments in 16 32 128 256; do
    run table1_mean_seg${segments} $BASE --set model.cgas_pool_modes=[mean] --set model.cgas_segments=$segments
    run table1_dropoutmean_seg${segments} $BASE --set model.cgas_pool_modes=[dropout] --set model.cgas_segments=$segments
done
run table1_nopool $BASE --set model.cgas_pool_modes=[mean] --set model.cgas_segments=768

# ── TABLE2: dropout_p x segments ────────────────────────────────────────────
for p in 0.1 0.5 0.7; do
    for segments in 16 32 128 256; do
        run table2_p${p}_seg${segments} $BASE --set model.cgas_pool_modes=[dropout] --set model.cgas_dropout_p=$p --set model.cgas_segments=$segments
    done
done

# ── COLLECT ─────────────────────────────────────────────────────────────────
python scripts/tables/runs_csv.py --runs-root runs/field_study
