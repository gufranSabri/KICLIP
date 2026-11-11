#!/bin/bash
# KICLIP: interactive pipeline. Run command-by-command in an interactive
# SLURM terminal -- not meant to be executed as a whole.

# ── STEP 0: ALLOCATE (run ONE of these) ────────────────────────────────────

salloc --gpus-per-node=l40s:1 --cpus-per-task=16 --mem=64G --time=3:00:00 --account=aip-lsigal
# salloc --gpus-per-node=l40s:4 --cpus-per-task=32 --mem=128G --time=12:00:00 --account=aip-lsigal
# salloc --gpus-per-node=h100:1 --cpus-per-task=16 --mem=64G --time=1:00:00 --account=aip-lsigal

# ── STEP 1: ENVIRONMENT ────────────────────────────────────────────────────

module load StdEnv/2023 gcc/12.3 cuda/13.2 python/3.11.5
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/ahmedubc/scratch/hf_cache

bash scripts/install.sh

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name())"

# ── STEP 1b: DATA (once; on a login node, reruns skip what is done) ────────

bash scripts/data/download.sh
python scripts/data/organize.py --check --workers 8 --delete-raw

# ── STEP 2: SMOKE TEST (no KI vs KI vs KI + aux, HMDB51, a few minutes each) ─

# run <run-name> [config args]: train into runs/<run-name>, then test that checkpoint; skips finished steps
run() {
    local name=$1; shift
    local dir="runs/$name"
    [[ -f "$dir/averaged.pt" ]] || python kiclip.py "$@" train --run-name "$name" || return
    [[ -f "$dir/test_b2n_hmdb51_base.json" && -f "$dir/test_b2n_hmdb51_novel.json" ]] \
        || python kiclip.py "$@" test --checkpoint "$dir/averaged.pt"
}

run no_ki_b2 --config configs/b2n/hmdb51.yaml --config configs/methods/froster.yaml --set model.distill_weight=2.0
run ki_b2 --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --set model.aux_classifier=false  --set model.distill_weight=2.0
run ki_aux_b2 --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --set model.aux_classifier=true  --set model.distill_weight=2.0

# ── STEP 3: L_KEEP SWEEP (notes/l_keep.md, HMDB51, on top of KI + aux) ──────
# step-0 scale on real HMDB clips: absolute ~0.008 (with KI) / ~0.057 (no KI); geometric ~3e-4 (with KI) / ~7e-4 (no KI)

run keep_abs_10 --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --set model.l_keep=absolute --set model.keep_weight=10
run keep_geo_10 --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --set model.l_keep=geometric --set model.keep_weight=10

python scripts/tables/runs_csv.py --runs-root runs --out docs/tables/runs.csv

