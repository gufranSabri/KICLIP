#!/bin/bash
# Few-shot learning: HMDB51 and UCF101, K={2,4,8,16} training clips per class, all classes (no
# base/novel split) -- MSTA Table 3's protocol, using ViFi-CLIP's standard few-shot splits (the ones
# OST/MoTE/MSTA all reuse; see scripts/data/build_fewshot_splits.py). No seeds: one fixed draw per K,
# same as the source split files. SSv2 is excluded (no full 174-class vocabulary in this repo). Uses
# configs/fewshot/best/<ds>.yaml (each dataset's b2n best model hyperparameters, on fewshot's own data).
# Run command-by-command in an interactive SLURM terminal (after scripts/troubleshooting.sh's STEP 0/1/1b).

run() {
    local name=$1; shift
    local dir="runs/$name"
    [[ -f "$dir/averaged.pt" ]] || python kiclip.py "$@" train --run-name "$name" || return
    grep -qs '"top1"' "$dir/results.json" || python kiclip.py "$@" test --checkpoint "$dir/averaged.pt"
}

for ds in hmdb51 ucf101; do
    best="configs/fewshot/best/$ds.yaml"
    if [[ ! -f "$best" ]]; then
        echo "missing $best; copy configs/b2n/best/$ds.yaml's model: block onto fewshot/default/$ds.yaml's data first" >&2
        continue
    fi
    for k in 2 4 8 16; do
        run fewshot/"${ds}_cgas_k${k}" --config "$best" --set data.train=fewshot/${ds}/train_k${k}
    done
done

# ── COLLECT ─────────────────────────────────────────────────────────────────
python scripts/tables/runs_csv.py --runs-root runs --out docs/tables/runs.csv
