#!/bin/bash
# KICLIP: interactive pipeline. Run command-by-command in an interactive
# SLURM terminal -- not meant to be executed as a whole.

# ── STEP 0: ALLOCATE (run ONE of these) ────────────────────────────────────

salloc --gpus-per-node=l40s:1 --cpus-per-task=16 --mem=24G --time=3:00:00 --account=aip-lsigal
# salloc --gpus-per-node=l40s:4 --cpus-per-task=32 --mem=128G --time=12:00:00 --account=aip-lsigal
# salloc --gpus-per-node=h100:1 --cpus-per-task=16 --mem=24G --time=3:00:00 --account=aip-lsigal

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

# ── STEP 2: EXPERIMENTS ─────────────────────────────────────────────────────

bash scripts/experiments/ceweight_sweep.sh
python scripts/pick_best_config.py --runs-root runs/ceweight_sweep

bash scripts/experiments/seed_ablation.sh
bash scripts/experiments/run_b2n.sh
bash scripts/experiments/run_fewshot.sh
bash scripts/experiments/run_zeroshot.sh

# ── STEP 3: CGAS ABLATIONS ──────────────────────────────────────────────────

bash scripts/experiments/field_study.sh
bash scripts/experiments/distill_cgas_weight_sweep.sh

# ── STEP 4: DCGD NARRATIVE FIGURES (HMDB-51 only) ───────────────────────────
# needs runs/main/hmdb51_{froster,dd,cgas_noki,cgas} from run_b2n.sh

bash scripts/experiments/pooling_sweep.sh
bash scripts/experiments/capacity_sweep.sh
bash scripts/experiments/cgas_target_ablation.sh

python scripts/analysis/dcgd_figures.py impulse --checkpoint runs/main/hmdb51_dd
python scripts/analysis/dcgd_figures.py scatter \
    --checkpoints runs/main/hmdb51_froster runs/main/hmdb51_dd runs/main/hmdb51_cgas_noki runs/main/hmdb51_cgas \
    --labels FROSTER "Drift Damping" CGAS DCGD
python scripts/analysis/dcgd_figures.py concentration --checkpoint runs/pooling_sweep/mean
python scripts/analysis/dcgd_figures.py distance \
    --checkpoints runs/main/hmdb51_froster runs/main/hmdb51_dd runs/main/hmdb51_cgas_noki runs/main/hmdb51_cgas \
    --labels FROSTER "Drift Damping" CGAS DCGD
