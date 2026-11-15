#!/bin/bash
# CGAS mechanism study: run command-by-command in an interactive SLURM terminal
# (after scripts/troubleshooting.sh's STEP 0/1/1b) -- not meant to be executed as a whole.

# run <run-name> [config args]: train into runs/<run-name>, then test that checkpoint; skips finished steps
run() {
    local name=$1; shift
    local dir="runs/$name"
    [[ -f "$dir/averaged.pt" ]] || python kiclip.py "$@" train --run-name "$name" || return
    grep -qs '"HM"' "$dir/results.json" || python kiclip.py "$@" test --checkpoint "$dir/averaged.pt"
}

# ============================================================================
# CGAS ABLATIONS -- all HMDB-51 B2N, train_s1, seed 0 unless noted. Each run below is
# evidence for one claim in docs/cgas.md. The main method's own numbers (configs/methods/cgas.yaml,
# field_layers=all, dropout pooling) come from scripts/run_b2n.sh, not from here -- this file only
# isolates WHY the mechanism works, at field_layers=last_1 where the ablation is clean.
# ============================================================================

# ── BASELINES ────────────────────────────────────────────────────────────
run froster --config configs/b2n/hmdb51.yaml --config configs/methods/froster.yaml
run ki --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml
run ki_s1 --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --set seed=1
run ki_s2 --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --set seed=2

# ── GRADIENT-PATH ISOLATION (does cgas_ce shape the backbone via its OWN gradient?) ──
# field_dummy's old slot (head exists, loss weight 0) is gone: cgas.py no longer resamples pool
# modes via a shared RNG call, so the "does merely instantiating the head change anything" control
# is moot. field_isolated = detach, the cleanest isolation; field_detach = same but with the
# (now-removed) test-time add-on, kept for comparison against the historical record in notes/.

run field_isolated --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml --set model.cgas_detach=true
run field_ce_only --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml

# ── DEPTH: last_1 vs. all 12 blocks ─────────────────────────────────────
run field_layers_last1 --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml --set model.cgas_layers=last_1
run field_layers_last3 --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml --set model.cgas_layers=last_3
run field_layers_last6 --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml --set model.cgas_layers=last_6
# cgas_layers=all is scripts/run_b2n.sh's main cgas run -- not repeated here

# ── POOLING LADDER: does the gate (mean vs. max/min) matter, at a clean single-block readout ──
# cgas_segments=768 on a 768-wide CLS with mode=mean is an EXACT IDENTITY (pool_segments reshapes
# to (...,768,1).mean(-1)) -- i.e. no pooling/compression at all, the sharpest version of the test.

run field_nopool --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml --set model.cgas_pool_modes=[mean] --set model.cgas_layers=last_1 --set model.cgas_segments=768
run field_pool_mean_last1 --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml --set model.cgas_pool_modes=[mean] --set model.cgas_layers=last_1
run field_pool_max_last1 --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml --set model.cgas_pool_modes=[max] --set model.cgas_layers=last_1
run field_pool_min_last1 --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml --set model.cgas_pool_modes=[min] --set model.cgas_layers=last_1
# cgas_pool_modes=[dropout], cgas_layers=last_1 is field_layers_last1 above -- not rerun here

for seed in 1 2; do
    for mode in mean max min; do
        run field_pool_${mode}_last1_s${seed} --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml --set model.cgas_pool_modes=[$mode] --set model.cgas_layers=last_1 --set seed=$seed
    done
    run field_nopool_s${seed} --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml --set model.cgas_pool_modes=[mean] --set model.cgas_layers=last_1 --set model.cgas_segments=768 --set seed=$seed
    run field_layers_last1_s${seed} --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml --set model.cgas_layers=last_1 --set seed=$seed
done

# ── CAPACITY SWEEP: the bottleneck width, mean pooling throughout ──────────
# width 64 = field_pool_mean_last1, width 768 = field_nopool, both already run above

for segments in 16 32 128 256; do
    run field_mean_seg${segments}_last1 --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml --set model.cgas_pool_modes=[mean] --set model.cgas_layers=last_1 --set model.cgas_segments=$segments
done
for seed in 1 2; do
    for segments in 16 32 128 256; do
        run field_mean_seg${segments}_last1_s${seed} --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml --set model.cgas_pool_modes=[mean] --set model.cgas_layers=last_1 --set model.cgas_segments=$segments --set seed=$seed
    done
done

# also at a fixed width, dropout pooling: does narrowing the segments count AND dropout-masking stack?
for segments in 16 256; do
    run field_segments_${segments}_last1 --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml --set model.cgas_layers=last_1 --set model.cgas_segments=$segments
done

# ── DROPOUT RATE: does the mechanism's prediction (more masking -> more gradient to backbone,
# up to the point where the head stops being a useful teacher) hold across p? ──────────────────
for p in 0.1 0.5 0.7; do
    run field_dropout_p${p}_last1 --config configs/b2n/hmdb51.yaml --config configs/methods/kiclip.yaml --config configs/methods/cgas.yaml --set model.cgas_layers=last_1 --set model.cgas_dropout_p=$p
done

# per-run feature dumps for the pooling-mode / capacity figures (GPU, after the runs above)
python scripts/analysis/dump.py --config configs/b2n/hmdb51.yaml --no-clip --runs runs/field_pool_mean_last1 runs/field_pool_max_last1 runs/field_pool_min_last1 runs/field_nopool
python scripts/analysis/cgas_figures.py --recompute

# ── COLLECT ─────────────────────────────────────────────────────────────────
python scripts/tables/runs_csv.py --runs-root runs --out docs/tables/runs.csv
