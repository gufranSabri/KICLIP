#!/bin/bash
# KICLIP dependencies, the single source of truth (no requirements.txt); run inside an activated venv.
#
#   bash scripts/install.sh

set -euo pipefail

echo "[install] core dependencies"

PIP_FLAGS=""
if [ -n "${CC_CLUSTER:-}${SLURM_JOB_ID:-}" ] && pip install --no-index --dry-run pip >/dev/null 2>&1; then
  PIP_FLAGS="--no-index"   # Compute Canada: install from the local wheelhouse
fi

pip install $PIP_FLAGS --upgrade pip

pip install $PIP_FLAGS \
  torch \
  torchvision \
  av \
  open_clip_torch \
  ftfy \
  regex \
  huggingface_hub \
  safetensors \
  numpy \
  pyyaml \
  tqdm \
  matplotlib

echo "[install] core dependencies OK"

python - <<'PY'
import av, open_clip, torch
missing = [c for c in ("h264", "mpeg4", "vp8", "vp9") if c not in av.codecs_available]
assert not missing, f"PyAV was built without decoders {missing}; K400 needs h264, UCF/HMDB mpeg4, SSv2 vp8/vp9"
tokens = open_clip.tokenize(["a video of a person juggling"])
assert tokens.shape == (1, 77), tokens.shape
print(f"[install] OK  (torch {torch.__version__}, cuda {torch.cuda.is_available()}, av {av.__version__}, open_clip {open_clip.__version__})")
PY

echo "[install] done"
