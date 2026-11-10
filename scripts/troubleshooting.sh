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
export TORCH_HOME=/home/ahmedubc/scratch/torch_cache

bash scripts/install.sh

python -c "import torch; print('CUDA:', torch.cuda.is_available(), '|', torch.cuda.get_device_name())"

# ── STEP 1b: DATA (once; on a login node, reruns skip what is done) ────────

bash scripts/data/download.sh
python scripts/data/organize.py --check --workers 8 --delete-raw

# ── STEP 2: SMOKE TEST (KICLIP and FROSTER on HMDB51, a few minutes each) ───

python kiclip.py --config configs/b2n/hmdb51.yaml train --run-name smoke_kiclip --set train.epochs=3 --set train.warmup_epochs=1
python kiclip.py --config configs/b2n/hmdb51.yaml test --checkpoint runs/smoke_kiclip/averaged.pt

python kiclip.py --config configs/b2n/hmdb51.yaml --config configs/methods/froster.yaml train --run-name smoke_froster --set train.epochs=3 --set train.warmup_epochs=1
python kiclip.py --config configs/b2n/hmdb51.yaml --config configs/methods/froster.yaml test --checkpoint runs/smoke_froster/averaged.pt

python scripts/figures/eigencam.py --split b2n/hmdb51/novel --index 0 \
    --checkpoints clip runs/smoke_froster/averaged.pt runs/smoke_kiclip/averaged.pt --labels CLIP FROSTER KICLIP

# ── STEP 3: EXPERIMENTS ────────────────────────────────────────────────────
# LAUNCH runs every train/test on all allocated GPUs; batch sizes stay the paper's.

export LAUNCH="torchrun --standalone --nproc_per_node=$SLURM_GPUS_ON_NODE"
export METHODS="kiclip froster"
bash scripts/experiments/b2n.sh
bash scripts/experiments/cross_dataset.sh

python scripts/tables/b2n.py
python scripts/tables/cross_dataset.py

