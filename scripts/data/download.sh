#!/bin/bash
# Download the raw videos into {data_root}/raw/<dataset>; scripts/data/organize.py then moves them where the split CSVs expect.
#
#   bash scripts/data/download.sh                    # ucf101 hmdb51 k400 k600
#   bash scripts/data/download.sh ucf101 hmdb51      # only these
#   JOBS=16 bash scripts/data/download.sh k400       # parallel tarball streams
#   bash scripts/data/download.sh ssv2               # extracts the official parts you placed in raw/ssv2
#   python scripts/data/organize.py --check --workers 8 --delete-raw

cd "$(dirname "$0")/../.." || exit 1
set -uo pipefail

DATA_ROOT=${DATA_ROOT:-$(python -c "import os, yaml; print(os.path.expanduser(yaml.safe_load(open('configs/default.yaml'))['data_root']))")}
RAW=$DATA_ROOT/raw
JOBS=${JOBS:-6}
CVDF=https://s3.amazonaws.com/kinetics
HF=https://huggingface.co/datasets
DATASETS=("$@")
[ ${#DATASETS[@]} -eq 0 ] && DATASETS=(ucf101 hmdb51 k400 k600)

# one CVDF tarball, streamed straight into tar; with a keep file only the listed clips are kept; a marker skips it on reruns
fetch_tar() {
    local url="$1" dest="$2" keep="${3:-}" name out
    set -o pipefail
    name=$(basename "$url" .tar.gz)
    [ -f "$dest/.done/$name" ] && return 0
    mkdir -p "$dest/.done"
    out=$dest
    [ -n "$keep" ] && out=$dest/.tmp_$name
    for try in 1 2 3; do
        [ -n "$keep" ] && rm -rf "$out" && mkdir -p "$out"
        # single-threaded gzip: login nodes cap threads per user, and pigz workers hit that cap
        if curl -sS --fail --retry 5 --retry-delay 10 "$url" | gzip -dc | tar -xf - -C "$out"; then
            if [ -n "$keep" ]; then
                find "$out" -type f -name '*.mp4' | grep -Ff "$keep" | xargs -r mv -t "$dest"
                rm -rf "$out"
            fi
            touch "$dest/.done/$name"
            echo "  [ok] $name"
            return 0
        fi
        echo "  [retry $try] $name"
        sleep 30
    done
    echo "  [FAIL] $url"
    return 1
}
export -f fetch_tar

fetch_list() {
    local list="$1" dest="$2"
    mkdir -p "$dest"
    curl -sS --fail "$CVDF/$list" | xargs -d '\n' -P "$JOBS" -I{} bash -c 'fetch_tar "$1" "$2"' _ {} "$dest"
}

for DS in "${DATASETS[@]}"; do
    echo "========================================"
    echo "Dataset: $DS -> $RAW/$DS"
    echo "========================================"
    if [ -f "$DATA_ROOT/$DS/.organized" ]; then
        echo "  [skip] already organized under $DATA_ROOT/$DS"
        continue
    fi
    case $DS in
        k400)
            fetch_list 400/val/k400_val_path.txt "$RAW/k400/val"
            fetch_list 400/train/k400_train_path.txt "$RAW/k400/train"
            ;;
        k600)
            # the splits' K600 val clips sit in CVDF's per-class 600/train tarballs (its 600/val ones hold other clips);
            # only the classes the splits test on are streamed, and only the listed clips are kept
            mkdir -p "$RAW/k600/val"
            NEEDED=$(cut -d/ -f2 splits/cross/k600/split*.csv | sort -u)
            curl -sS --fail "$CVDF/600/train/k600_train_path.txt" | while read -r url; do
                cls=$(basename "$url" .tar.gz)
                cls=${cls// /_}
                if grep -qxF "$cls" <<< "$NEEDED"; then
                    mkdir -p "$RAW/k600/val/$cls"
                    grep -hF "val/$cls/" splits/cross/k600/split*.csv | cut -d, -f1 | xargs -n1 basename | sort -u \
                        > "$RAW/k600/val/$cls/.keep"
                    printf '%s\n%s\n%s\n' "${url// /%20}" "$RAW/k600/val/$cls" "$RAW/k600/val/$cls/.keep"
                fi
            done | xargs -d '\n' -n 3 -P "$JOBS" bash -c 'fetch_tar "$1" "$2" "$3"' _
            ;;
        ucf101)
            mkdir -p "$RAW/ucf101"
            if [ ! -f "$RAW/ucf101/.done" ]; then
                curl -sSL --fail -C - -o "$RAW/ucf101/UCF-101.zip" "$HF/quchenyuan/UCF101-ZIP/resolve/main/UCF-101.zip" \
                    && unzip -q -o "$RAW/ucf101/UCF-101.zip" -d "$RAW/ucf101" \
                    && rm -f "$RAW/ucf101/UCF-101.zip" && touch "$RAW/ucf101/.done" && echo "  [ok] ucf101"
            fi
            ;;
        hmdb51)
            mkdir -p "$RAW/hmdb51"
            if [ ! -f "$RAW/hmdb51/.done" ]; then
                HF_HUB_DISABLE_XET=1 python - "$RAW/hmdb51" "$JOBS" <<'PY' && touch "$RAW/hmdb51/.done" && echo "  [ok] hmdb51"
import sys
from huggingface_hub import snapshot_download
snapshot_download("CVML-TueAI/HMDB51", repo_type="dataset", allow_patterns=["hmdb51_org/*/*.avi"],
                  local_dir=sys.argv[1], max_workers=int(sys.argv[2]))
PY
            fi
            ;;
        ssv2)
            # Something-Something v2 needs its research license accepted at
            # https://www.qualcomm.com/developer/software/something-something-v-2-dataset/downloads
            # put the downloaded 20bn-something-something-v2-?? parts in raw/ssv2, then rerun this
            mkdir -p "$RAW/ssv2"
            if [ ! -f "$RAW/ssv2/.done" ]; then
                if ! ls "$RAW"/ssv2/20bn-something-something-v2-?? >/dev/null 2>&1; then
                    echo "  [skip] no 20bn-something-something-v2-?? parts in $RAW/ssv2 (license-gated, download them manually)"
                    continue
                fi
                cat "$RAW"/ssv2/20bn-something-something-v2-?? | tar -xzf - -C "$RAW/ssv2" && touch "$RAW/ssv2/.done" && echo "  [ok] ssv2"
            fi
            ;;
        *)
            echo "unknown dataset $DS (k400 | k600 | ucf101 | hmdb51 | ssv2)"
            exit 1
            ;;
    esac
done
