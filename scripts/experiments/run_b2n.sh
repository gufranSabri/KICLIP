#!/bin/bash
# Main CGAS experiments: base-to-novel on hmdb51/ucf101/k400, one run each of FROSTER, drift damping
# (DD, no CGAS), CGAS without drift damping underneath (cgas_noki) and CGAS (the full method), all from
# each dataset's best config (configs/b2n/best/<ds>.yaml, written by scripts/pick_best_config.py -- run
# that first). Then EigenCAM: one image per dataset, one row per method, over 10 sampled novel-split
# instances per dataset (indices 0..9).
# Run command-by-command in an interactive SLURM terminal (after scripts/troubleshooting.sh's STEP 0/1/1b).
# Seed/split ablation moved to scripts/experiments/seed_ablation.sh.

run() {
    local name=$1; shift
    local dir="runs/$name"
    [[ -f "$dir/averaged.pt" ]] || python kiclip.py "$@" train --run-name "$name" || return
    grep -qs '"HM"' "$dir/results.json" || python kiclip.py "$@" test --checkpoint "$dir/averaged.pt"
}

# ── TRAIN + TEST: froster, drift damping (DD), cgas_noki, cgas -- hmdb51 / ucf101 / k400 ──
for ds in hmdb51 ucf101 k400; do
    best="configs/b2n/best/$ds.yaml"
    if [[ ! -f "$best" ]]; then
        echo "missing $best; run 'python scripts/pick_best_config.py' first" >&2
        continue
    fi
    run main/${ds}_froster --config "$best" --set model.cgas_enabled=false --set model.drift_damping=false
    run main/${ds}_dd --config "$best" --set model.cgas_enabled=false
    run main/${ds}_cgas_noki --config "$best" --set model.drift_damping=false
    run main/${ds}_cgas --config "$best"
done

# ── EIGENCAM: one image per dataset, rows = froster / dd / cgas_noki / cgas, 10 novel instances ──
for ds in hmdb51 ucf101 k400; do
    for ckpt in main/${ds}_froster main/${ds}_dd main/${ds}_cgas_noki main/${ds}_cgas; do
        [[ -f runs/$ckpt/averaged.pt ]] || { echo "missing runs/$ckpt/averaged.pt, skipping $ds eigencam" >&2; continue 2; }
    done
    for i in 0 1 2 3 4 5 6 7 8 9; do
        python scripts/figures/eigencam.py --split b2n/${ds}/novel --index $i \
            --checkpoints runs/main/${ds}_froster/averaged.pt runs/main/${ds}_dd/averaged.pt \
                runs/main/${ds}_cgas_noki/averaged.pt runs/main/${ds}_cgas/averaged.pt \
            --labels FROSTER "Drift Damping" "CGAS (no DD)" CGAS \
            --out docs/figures/eigencam/${ds}_${i}.png
    done
done

# ── COLLECT ─────────────────────────────────────────────────────────────────
python scripts/tables/runs_csv.py --runs-root runs/main
