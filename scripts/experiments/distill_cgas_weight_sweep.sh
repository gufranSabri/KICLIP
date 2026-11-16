#!/bin/bash
# distill_weight x cgas_ce_weight sweep: HMDB-51 b2n, from the best config (configs/b2n/best/hmdb51.yaml,
# written by scripts/pick_best_config.py -- run that first), varying only distill_weight and cgas_ce_weight.
# cgas_ce_weight scales the SUM of CGAS's label-CE and teacher-KL terms (model.cgas_target=both, the main
# method's own setting). distill_weight=0 also drops the distillation projectors (model.py); see CLAUDE.md
# for that detail. Run command-by-command in an interactive SLURM terminal (after scripts/troubleshooting.sh's
# STEP 0/1/1b).

BASE="--config configs/b2n/best/hmdb51.yaml"
if [[ ! -f configs/b2n/best/hmdb51.yaml ]]; then
    echo "missing configs/b2n/best/hmdb51.yaml; run 'python scripts/pick_best_config.py' first" >&2
    exit 1
fi

run() {
    local name=$1; shift
    local dir="runs/distill_cgas_weight_sweep/$name"
    [[ -f "$dir/averaged.pt" ]] || python kiclip.py "$@" train --run-name "distill_cgas_weight_sweep/$name" || return
    grep -qs '"HM"' "$dir/results.json" || python kiclip.py "$@" test --checkpoint "$dir/averaged.pt"
}

for dw in 0 0.5 1 2; do
    for cw in 0 0.5 1 2; do
        run distill${dw}_cgas${cw} $BASE --set model.distill_weight=$dw --set model.cgas_ce_weight=$cw
    done
done

# ── COLLECT ─────────────────────────────────────────────────────────────────
python scripts/tables/runs_csv.py --runs-root runs/distill_cgas_weight_sweep
