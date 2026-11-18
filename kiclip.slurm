#!/bin/bash
# Batch job for KICLIP: base-to-novel, few-shot and zero-shot runs, then the tables. Resubmit to resume.
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

bash scripts/experiments/ceweight_sweep.sh
python scripts/pick_best_config.py --runs-root runs/ceweight_sweep

bash scripts/experiments/run_b2n.sh
bash scripts/experiments/run_fewshot.sh
bash scripts/experiments/run_zeroshot.sh

python scripts/tables/b2n.py
python scripts/tables/cross_dataset.py
python scripts/tables/main_results.py