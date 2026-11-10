#!/bin/bash
# Batch job for KICLIP: every base-to-novel and cross-dataset run for $METHODS (default KICLIP + FROSTER), then the tables. Resubmit to resume.
#SBATCH --job-name=kiclip
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --open-mode=append
#SBATCH --account=aip-lsigal
#SBATCH --mail-user=ahmed.ghuf@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

set -uo pipefail

module load StdEnv/2023 gcc/12.3 cuda/13.2 python/3.11.5
virtualenv --no-download "$SLURM_TMPDIR/env"
source "$SLURM_TMPDIR/env/bin/activate"

export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/ahmedubc/scratch/hf_cache

bash scripts/install.sh || exit 1

export METHODS=${METHODS:-kiclip froster}
export LAUNCH="torchrun --standalone --nproc_per_node=${SLURM_GPUS_ON_NODE:-1}"
bash scripts/experiments/b2n.sh
bash scripts/experiments/cross_dataset.sh

python scripts/tables/b2n.py
python scripts/tables/cross_dataset.py



SHORT="--set train.epochs=3 --set train.warmup_epochs=1"
python kiclip.py --config configs/b2n/hmdb51.yaml train --run-name smoke_kiclip $SHORT
python kiclip.py --config configs/b2n/hmdb51.yaml test --checkpoint runs/smoke_kiclip/averaged.pt
python kiclip.py --config configs/b2n/hmdb51.yaml --config configs/methods/froster.yaml train --run-name smoke_froster $SHORT
python kiclip.py --config configs/b2n/hmdb51.yaml --config configs/methods/froster.yaml test --checkpoint runs/smoke_froster/averaged.pt
python scripts/figures/eigencam.py --config configs/b2n/hmdb51.yaml --split b2n/hmdb51/novel --index 0 \
    --checkpoints clip runs/smoke_froster/averaged.pt runs/smoke_kiclip/averaged.pt --labels CLIP FROSTER KICLIP