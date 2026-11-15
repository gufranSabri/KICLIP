#!/bin/bash
# UCF101 CGAS hyperparameter sweep: diagnoses why CGAS (base 95.9-96.3, novel 77.2-77.4, HM ~85.7)
# underperforms plain Froster (base 96.7, novel 81.3, HM 88.3) on UCF101, despite helping on HMDB.
# Hypothesis: cgas_ce/ce ratio is ~23-27x on UCF101 vs ~6-8x on HMDB at the same cgas_ce_weight=1.0
# (see train.log) -- the aux loss may be dominating the shared gradient and hurting generalization.
# Run command-by-command in an interactive SLURM terminal.

run() {
    local name=$1; shift
    local dir="runs/$name"
    [[ -f "$dir/averaged.pt" ]] || python kiclip.py "$@" train --run-name "$name" || return
    grep -qs '"HM"' "$dir/results.json" || python kiclip.py "$@" test --checkpoint "$dir/averaged.pt"
}

BASE="--config configs/b2n/ucf101.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml"

# ── reference points (already have numbers for these; rerun only if needed) ──
# run main/ucf101_froster_s0 --config configs/b2n/ucf101.yaml --config configs/methods/froster.yaml --set seed=0
# run main/ucf101_ki_s0      --config configs/b2n/ucf101.yaml --config configs/methods/kiclip.yaml  --set seed=0
# run main/ucf101_cgas_s0    $BASE --set seed=0   # 85.66 HM, novel 77.4

# ── AXIS 1: cgas_ce_weight -- lower the aux loss's weight in the total loss ──
for w in 0.1 0.25 0.5; do
    run ucf101_sweep/ceweight_${w} $BASE --set model.cgas_ce_weight=$w --set seed=0
done

# ── AXIS 2: cgas_lr -- lower the head's own learning rate (slower to fit, less loss to push back) ──
for lr in 3.33e-5 1e-4; do
    run ucf101_sweep/lr_${lr} $BASE --set model.cgas_lr=$lr --set seed=0
done

# ── AXIS 3: cgas_target -- drop the KL/distillation half, label-only CE ──
run ucf101_sweep/target_label $BASE --set model.cgas_target=label --set seed=0

# ── AXIS 4: cgas_tau -- lower temperature shrinks the tau^2-scaled KL term specifically ──
for tau in 1.0; do
    run ucf101_sweep/tau_${tau} $BASE --set model.cgas_tau=$tau --set seed=0
done

# ── combined: best-looking single-axis changes together, once the above land ──
# run ucf101_sweep/combined $BASE --set model.cgas_ce_weight=<BEST> --set model.cgas_target=<BEST> --set seed=0

# ── COLLECT ─────────────────────────────────────────────────────────────────
python scripts/tables/runs_csv.py --runs-root runs --out docs/tables/runs.csv
