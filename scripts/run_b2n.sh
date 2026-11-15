#!/bin/bash
# Main CGAS experiments: base-to-novel on all four datasets, FROSTER / KI / KI+CGAS, 3 runs each.
# Run command-by-command in an interactive SLURM terminal (after scripts/troubleshooting.sh's STEP 0/1/1b).
# HMDB51 and UCF101 only have one few-shot split (train_s1), so their 3 runs vary the RNG seed; K400 and
# SSv2 ship 3 few-shot splits (train_s1/s2/s3), so their 3 runs vary the split instead, at seed 0 throughout.

run() {
    local name=$1; shift
    local dir="runs/$name"
    [[ -f "$dir/averaged.pt" ]] || python kiclip.py "$@" train --run-name "$name" || return
    grep -qs '"HM"' "$dir/results.json" || python kiclip.py "$@" test --checkpoint "$dir/averaged.pt"
}

METHODS="cgas:configs/methods/kiclip.yaml,configs/methods/cgas.yaml"

# ── HMDB51 / UCF101: one split, three seeds ─────────────────────────────────
for ds in hmdb51 ucf101; do
    for entry in $METHODS; do
        tag=${entry%%:*}
        IFS=',' read -ra cfgs <<< "${entry#*:}"
        args=(--config configs/b2n/$ds.yaml)
        for c in "${cfgs[@]}"; do args+=(--config "$c"); done
        for seed in 0 1 2; do
            run main/"${ds}_${tag}_s${seed}" "${args[@]}" --set seed=$seed
        done
    done
done

# ── K400 / SSv2: three few-shot splits, seed 0 ──────────────────────────────
for ds in k400 ssv2; do
    for entry in $METHODS; do
        tag=${entry%%:*}
        IFS=',' read -ra cfgs <<< "${entry#*:}"
        args=(--config configs/b2n/$ds.yaml)
        for c in "${cfgs[@]}"; do args+=(--config "$c"); done
        for split in 1 2 3; do
            run main/"${ds}_${tag}_split${split}" "${args[@]}" --set data.train=b2n/${ds}/train_s${split}
        done
    done
done

# ── COLLECT ─────────────────────────────────────────────────────────────────
python scripts/tables/runs_csv.py --runs-root runs --out docs/tables/runs.csv
