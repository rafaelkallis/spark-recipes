#!/bin/bash
# run.sh — tonyd2wild v11 image patches (GLM-5.3-Flash DFlash2, TP2)
#
# Applies the two single-file patches that tonyd2wild's
# launch-glm53-vllm-tp2-dflash2.sh bind-mounts onto the v16 image (vllm
# 0.1.dev20051+g487ecf187) at launch time. sparkrun cannot replicate file-level
# bind mounts, so a mod stages both instead:
#
#   1. sparse_attn_indexer_kpool.py — the SM121 top-k persistent-kernel fix.
#      Without it the engine hard-crashes on any decode past ~24K context
#      (docs/SM121-CRASH-FORENSICS-2026-08-27.md). Full-file replacement.
#   2. patch_prefix_cache_draft_group.py — the #18 prefix-cache repair for the
#      DFlash2 draft group (in-place edit of kv_cache_coordinator.py, anchored
#      and fail-closed; the script also self-checks its draft-group predicate
#      against the installed vLLM). Measured 2026-09-18: 13K-token repeat TTFT
#      21.1 s -> 6.3 s; repeat-prefix hit rate 0.986 on the 262K probe.
#
# Sources (pinned upstream HEAD d061f26ad3ec5c3c04f64aad7516dbe62a8aa6af):
#   docker/sparse_attn_indexer_kpool_sm121.py
#   docker/dflash2-overlay/patch_prefix_cache_draft_group.py
#   github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark
set -euo pipefail

PREFIX="[tonyd2wild-20261006]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE="/usr/local/lib/python3.12/dist-packages/vllm"
KPOOL_TARGET="$SITE/model_executor/layers/sparse_attn_indexer_kpool.py"
COORDINATOR_TARGET="$SITE/v1/core/kv_cache_coordinator.py"

echo "=== tonyd2wild v11 patch mod (GLM-5.3-Flash DFlash2) ==="

if [ ! -d "$SITE" ]; then
    echo "$PREFIX vLLM tree not found at $SITE" >&2
    exit 1
fi
for f in sparse_attn_indexer_kpool_sm121.py patch_prefix_cache_draft_group.py; do
    if [ ! -f "$MOD_DIR/$f" ]; then
        echo "$PREFIX missing $f" >&2
        exit 1
    fi
done
# The patch script hardcodes the python3.12 site path; fail loudly if this
# image moves it instead of silently skipping the repair.
if [ ! -f "$COORDINATOR_TARGET" ]; then
    echo "$PREFIX kv_cache_coordinator.py not found at $COORDINATOR_TARGET" >&2
    exit 1
fi

# 1. SM121 indexer kpool fix — full-file replacement (launcher bind-mounts
#    $HOME/patches/sparse_attn_indexer_kpool.py over the same target).
python3 -m py_compile "$MOD_DIR/sparse_attn_indexer_kpool_sm121.py"
cp "$MOD_DIR/sparse_attn_indexer_kpool_sm121.py" "$KPOOL_TARGET"
echo "$PREFIX kpool fix installed -> $KPOOL_TARGET"

# 2. #18 prefix-cache repair for the DFlash2 draft group. The upstream script
#    is anchored (one expected edit site each) and fail-closed on drift, and
#    self-checks the preview against the installed vLLM at the end — an anchor
#    or self-check failure aborts the pod boot before vllm serve starts.
python3 "$MOD_DIR/patch_prefix_cache_draft_group.py"

# Clear stale bytecode so the engine reloads the patched files.
find "$SITE" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "=== OK: tonyd2wild patches installed ==="
