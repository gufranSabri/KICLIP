#!/bin/bash
# CGAS seed/split ablation: base-to-novel, from each dataset's best config (configs/b2n/best/<ds>.yaml,
# written by scripts/pick_best_config.py -- run that first). HMDB51 and UCF101 only have one few-shot
# split (train_s1), so their 3 runs vary the RNG seed; K400 ships 3 few-shot splits (train_s1/s2/s3), so
# its 3 runs vary the split instead, at seed 0 throughout.
# Run command-by-command in an interactive SLURM terminal (after scripts/troubleshooting.sh's STEP 0/1/1b).

run() {
    local name=$1; shift
    local dir="runs/$name"
    [[ -f "$dir/averaged.pt" ]] || python kiclip.py "$@" train --run-name "$name" || return
    grep -qs '"HM"' "$dir/results.json" || python kiclip.py "$@" test --checkpoint "$dir/averaged.pt"
}

# ── HMDB51 / UCF101: one split, three seeds ─────────────────────────────────
for ds in hmdb51 ucf101; do
    best="configs/b2n/best/$ds.yaml"
    if [[ ! -f "$best" ]]; then
        echo "missing $best; run 'python scripts/pick_best_config.py' first" >&2
        continue
    fi
    for seed in 0 1 2; do
        run seed_ablation/"${ds}_cgas_s${seed}" --config "$best" --set seed=$seed
    done
done

# ── K400: three few-shot splits, seed 0 ─────────────────────────────────────
for ds in k400; do
    best="configs/b2n/best/$ds.yaml"
    if [[ ! -f "$best" ]]; then
        echo "missing $best; run 'python scripts/pick_best_config.py' first" >&2
        continue
    fi
    for split in 1 2 3; do
        run seed_ablation/"${ds}_cgas_split${split}" --config "$best" --set data.train=b2n/${ds}/train_s${split}
    done
done

# ── COLLECT ─────────────────────────────────────────────────────────────────
python scripts/tables/runs_csv.py --runs-root runs/seed_ablation
