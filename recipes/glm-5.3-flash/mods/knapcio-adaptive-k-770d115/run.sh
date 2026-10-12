#!/bin/bash
# run.sh — adaptive-K draft scheduler mod (GLM-5.3-Flash DFlash2, TP2)
#
# Stages the two-file adaptive draft-length scheduler onto the v11 image and
# exports PYTHONPATH so `--scheduler-cls adaptive_draft_scheduler.AdaptiveDraftScheduler`
# resolves at engine boot. No vLLM files are modified — this mod only stages a
# Python package; the serve process picks it up via PYTHONPATH.
#
#   1. adaptive_k_scheduler.py   — AdaptiveKScheduler: per-request VERIFY length
#      (subclass of vllm.v1.core.sched.async_scheduler.AsyncScheduler; hook
#      `_update_after_schedule` verified present in the v11 image).
#   2. adaptive_draft_scheduler.py — AdaptiveDraftScheduler: additionally
#      chooses how many tokens the DRAFTER produces each step (k_lo for prose,
#      k_hi for code/structured) from the same acceptance EMA.
#
# Both files copied verbatim from
# github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4 @ pinned commit
# 770d1153062aa916b06591411c61d5c593ea0f03 (Apache-2.0; separable MIT
# modifications by Jacopo Nardiello, LICENSES/MIT-Jacopo-Nardiello.txt).
#
# NOTE: environment variables exported here DO NOT reach `vllm serve` —
# sparkrun runs run.sh and the serve command in separate docker execs. The
# VLLM_ADAPTIVE_K_* knobs therefore live in the recipe's `env:` map. This
# script only verifies them (fail-closed) and stages the files.
set -euo pipefail

PREFIX="[knapcio-adaptive-k]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Writable staging path persisted in the sparkrun runtime cache (survives
# container recreate; PYTHONPATH in the recipe env points here).
STAGE_ROOT="${ADAPTIVE_K_STAGE_DIR:-/cache/runtime/adaptive-k-mod}"
STAGE="$STAGE_ROOT/overlay"

echo "=== adaptive-K scheduler mod (GLM-5.3-Flash DFlash2) ==="

for f in adaptive_k_scheduler.py adaptive_draft_scheduler.py; do
    if [ ! -f "$MOD_DIR/$f" ]; then
        echo "$PREFIX missing $f" >&2
        exit 1
    fi
done

# Fail-closed knob check: the scheduler silently no-ops into the fallback
# dense table if these are missing, which would look like a speed regression
# rather than a bug. The recipe env must carry them.
for v in VLLM_ADAPTIVE_K_LO VLLM_ADAPTIVE_K_HI VLLM_ADAPTIVE_K_MODE; do
    if [ -z "${!v:-}" ]; then
        echo "$PREFIX $v not set in environment (recipe env: map must define it)" >&2
        exit 1
    fi
done

# Sanity: the vLLM hook the scheduler subclasses must exist in this image.
if python3 -c "from vllm.v1.core.sched.async_scheduler import AsyncScheduler; assert hasattr(AsyncScheduler, '_update_after_schedule')" 2>/dev/null; then
    echo "$PREFIX AsyncScheduler._update_after_schedule present"
else
    echo "$PREFIX FATAL: installed vLLM lacks AsyncScheduler._update_after_schedule — scheduler incompatible with this image" >&2
    exit 1
fi

# Stage into the persistent cache (idempotent; always refresh from the mod dir
# so a recipe pin bump restages cleanly).
mkdir -p "$STAGE"
python3 -m py_compile "$MOD_DIR/adaptive_k_scheduler.py" "$MOD_DIR/adaptive_draft_scheduler.py"
cp "$MOD_DIR/adaptive_k_scheduler.py" "$MOD_DIR/adaptive_draft_scheduler.py" "$STAGE/"
# Clear stale bytecode in the staging dir.
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "$PREFIX staged 2 files -> $STAGE (PYTHONPATH must include $STAGE_ROOT/overlay)"

echo "=== OK: adaptive-K scheduler staged ==="
