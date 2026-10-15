#!/bin/bash
# run.sh — technigmaai native TP2 overlay staging (GLM-5.3-Flash, 2x DGX Spark)
#
# Replicates, in sparkrun-mod form, what upstream's compose.yaml does at launch:
# ~33 read-only bind mounts of vendored overlay files over the pinned Kindling
# nightly vLLM build, plus the MiaAI-derived chat template, plus the ephemeral
# FlashInfer autotune wipe. Upstream pins:
#   - repo      github.com/technigmaai/glm-5.3-flash-gb10-tp2-native @ 34ecd2f309f448cdbee7f0d4b1f6de4f11fa0d78 (2026-10-13)
#               (overlays vendored from 3a119df75e771374dacc3fb0f73373a0558aa84a, 2026-10-10 —
#                byte-identical at 34ecd2f: its only delta, PR #7, touched no files/)
#   - image     technigmaai/glm-5.3-flash-gb10-tp2-native:c748079-displaykv1-arm64-cu130
#               digest sha256:46afd8bba26106aa7ea6e00f6fd4bd3ba3d0c7c49e86af85f269eb2c88c71fa5
#               (vLLM 0.30.1rc1.dev193+gddd6fbca1, torch 2.13.0+cu130, FlashInfer 0.7.0)
# The overlays MUST land on that exact pinned build: upstream's compose mounts
# them over the image's site-packages paths (python3.12 dist-packages) at every
# container start, which is what this mod replays per sparkrun launch.
#
# The staged file set covers every overlay this deployment's entrypoint
# recognizes (arx, snapshot, adaptive-k, fp8, megamoe, fixes, sp, recoverssm)
# and the display-KV worker utils; see README.md for the full mapping and env
# notes.
#
# Like upstream's bind mounts, some staged files are net-new modules (arx.py,
# adaptive_k.py, weight_snapshot.py, ...) rather than replacements — so only
# parent directories are anchor-checked, not every target file.
set -euo pipefail

PREFIX="[technigmaai-native-34ecd2f]"
MOD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE="/usr/local/lib/python3.12/dist-packages"
VLLM="$SITE/vllm"
FLASHINFER="$SITE/flashinfer"

echo "=== technigmaai native overlay staging (GLM-5.3-Flash TP2) ==="

if [ ! -d "$VLLM" ]; then
    echo "$PREFIX vLLM not found at $VLLM (wrong image for this mod)" >&2
    exit 1
fi

# Anchor checks: every target directory we replace files in must already exist
# in the pinned image. A missing anchor means the image is not the pinned
# Kindling build (or vLLM reorganized modules) — fail before a ~25 min load.
# weight_cache is anchored even though nothing is staged into it: the staged
# weight_snapshot.py imports vllm.model_executor.model_loader.weight_cache
# (protocol submodule, WeightCacheKey/hash_checkpoint), so an image without
# that package breaks on the snapshot path, not at staging.
for d in \
    "$VLLM/distributed/device_communicators" \
    "$VLLM/model_executor/model_loader" \
    "$VLLM/model_executor/model_loader/weight_cache" \
    "$VLLM/model_executor/layers/quantization" \
    "$VLLM/model_executor/layers/fused_moe/experts" \
    "$VLLM/model_executor/layers/fused_moe/router" \
    "$VLLM/model_executor/kernels/mhc" \
    "$VLLM/models/glm5next/common" \
    "$VLLM/v1/core/sched" \
    "$VLLM/v1/worker" \
    "$VLLM/v1/attention/backends/mla" \
    "$VLLM/v1/attention/backends" \
    "$VLLM/v1/worker/gpu/spec_decode/dflash2" \
    "$VLLM/parser" \
    "$FLASHINFER/mla/_batch_mla/_backends"; do
    if [ ! -d "$d" ]; then
        echo "$PREFIX anchor missing: $d — image does not match the pinned " >&2
        echo "$PREFIX vLLM nightly (ddd6fbca1); refusing to stage overlays." >&2
        exit 1
    fi
done

# copy <staged-file> <target-relative-to-SITE-or-absolute> [dir mode]
stage() {
    local src="$1" dst="$2"
    local target="$dst"
    case "$dst" in
        /*) ;;
        *) target="$SITE/$dst" ;;
    esac
    if [ ! -f "$MOD_DIR/$src" ]; then
        echo "$PREFIX missing staged file: $src" >&2
        exit 1
    fi
    mkdir -p "$(dirname "$target")"
    if [ -e "$target" ]; then
        cp "$MOD_DIR/$src" "$target"
        echo "$PREFIX replaced $target"
    else
        # Upstream compose creates net-new bind-mount targets the same way.
        cp "$MOD_DIR/$src" "$target"
        echo "$PREFIX added (net-new upstream compose target) $target"
    fi
}

# --- arx (custom RoCE collectives, replaces device_communicators join point) ---
stage overlays/arx/arx.py               vllm/distributed/device_communicators/arx.py
stage overlays/arx/arx_vllm.cu          vllm/distributed/device_communicators/arx_vllm.cu
stage overlays/arx/arxbig.cu            vllm/distributed/device_communicators/arxbig.cu
stage overlays/arx/cuda_communicator.py vllm/distributed/device_communicators/cuda_communicator.py

# --- snapshot (processed-weight load/cache: dense FP8/W4 transforms) ---
stage overlays/snapshot/weight_snapshot.py       vllm/model_executor/model_loader/weight_snapshot.py
stage overlays/snapshot/base_loader.py           vllm/model_executor/model_loader/base_loader.py
stage overlays/snapshot/dense_fp8.py             vllm/model_executor/model_loader/dense_fp8.py
stage overlays/snapshot/model_source.py          vllm/model_executor/model_loader/model_source.py
stage overlays/snapshot/modelopt.py              vllm/model_executor/layers/quantization/modelopt.py
stage overlays/snapshot/flashinfer_cutlass_moe.py vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py

# --- adaptive-k scheduler (scheduler-cls target) ---
stage overlays/adaptive-k/adaptive_k.py vllm/v1/core/sched/adaptive_k.py

# --- draft-trunc (adaptive DFlash2 draft truncation stack) ---
stage overlays/draft-trunc/draft_trunc.py         vllm/v1/worker/gpu/spec_decode/draft_trunc.py
stage overlays/draft-trunc/rejection_sampler.py   vllm/v1/worker/gpu/spec_decode/rejection_sampler.py
stage overlays/draft-trunc/speculator.py          vllm/v1/worker/gpu/spec_decode/speculator.py
stage overlays/draft-trunc/dflash2_speculator.py  vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py
stage overlays/draft-trunc/grouped_topk_router.py vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py
stage overlays/draft-trunc/model_runner.py        vllm/v1/worker/gpu/model_runner.py

# --- megamoe (MegaMoE decode/prefill kernels: .cu sources JIT'd at boot) ---
stage overlays/megamoe/megamoe_vllm.py vllm/model_executor/layers/fused_moe/megamoe_vllm.py
stage overlays/megamoe/megadense4.cu   /opt/megamoe/megadense4.cu
stage overlays/megamoe/megamoe_rm.cu   /opt/megamoe/megamoe_rm.cu
stage overlays/megamoe/moe_prefill.cu  /opt/megamoe/moe_prefill.cu

# --- fixes (model/attention/KV/smoke-fix overlays, incl. glm47_moe parser) ---
stage overlays/fixes/_fa_common.py             flashinfer/mla/_batch_mla/_backends/_fa_common.py
stage overlays/fixes/attention.py              vllm/models/glm5next/common/attention.py
stage overlays/fixes/kda.py                    vllm/models/glm5next/common/kda.py
stage overlays/fixes/model.py                  vllm/models/glm5next/common/model.py
stage overlays/fixes/recoverssm.py             vllm/models/glm5next/common/recoverssm.py
stage overlays/fixes/tilelang_kernels.py       vllm/model_executor/kernels/mhc/tilelang_kernels.py
stage overlays/fixes/mhc_tilelang.py           vllm/model_executor/kernels/mhc/tilelang.py
stage overlays/fixes/flashinfer_mla_sparse_sm90.py vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py
stage overlays/fixes/gb10_sparse_mla.py        vllm/v1/attention/backends/mla/gb10_sparse_mla.py
stage overlays/fixes/gdn_attn.py               vllm/v1/attention/backends/gdn_attn.py
stage overlays/fixes/causal_conv1d.py          vllm/model_executor/layers/mamba/ops/causal_conv1d.py
stage overlays/fixes/kv_cache_utils.py         vllm/v1/core/kv_cache_utils.py
stage overlays/fixes/kv_cache_coordinator.py   vllm/v1/core/kv_cache_coordinator.py
stage overlays/fixes/kv_cache_interface.py     vllm/v1/kv_cache_interface.py
stage overlays/fixes/warmup.py                 vllm/v1/worker/gpu/warmup.py
stage overlays/fixes/glm47_moe.py              vllm/parser/glm47_moe.py

# --- display-KV worker utils (activation gated by GLM53_DISPLAY_KV_ENABLE) ---
stage display-kv/worker_utils.py vllm/v1/worker/utils.py

# --- MiaAI-derived chat template (upstream: --chat-template CHAT_TEMPLATE) ---
mkdir -p /opt/glm53-native
stage chat_template.jinja /opt/glm53-native/chat_template.jinja

# Byte-compile every staged python file so truncation/mismatch fails here,
# before a ~25 minute weight load.
for f in \
    "$SITE/vllm/distributed/device_communicators/arx.py" \
    "$SITE/vllm/distributed/device_communicators/cuda_communicator.py" \
    "$SITE/vllm/model_executor/model_loader/weight_snapshot.py" \
    "$SITE/vllm/model_executor/model_loader/base_loader.py" \
    "$SITE/vllm/model_executor/model_loader/dense_fp8.py" \
    "$SITE/vllm/model_executor/model_loader/model_source.py" \
    "$SITE/vllm/model_executor/layers/quantization/modelopt.py" \
    "$SITE/vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py" \
    "$SITE/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py" \
    "$SITE/vllm/model_executor/layers/fused_moe/megamoe_vllm.py" \
    "$SITE/vllm/v1/core/sched/adaptive_k.py" \
    "$SITE/vllm/v1/worker/gpu/spec_decode/draft_trunc.py" \
    "$SITE/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py" \
    "$SITE/vllm/v1/worker/gpu/spec_decode/speculator.py" \
    "$SITE/vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py" \
    "$SITE/vllm/v1/worker/gpu/model_runner.py" \
    "$SITE/vllm/v1/worker/gpu/warmup.py" \
    "$SITE/vllm/v1/worker/utils.py" \
    "$SITE/vllm/models/glm5next/common/attention.py" \
    "$SITE/vllm/models/glm5next/common/kda.py" \
    "$SITE/vllm/models/glm5next/common/model.py" \
    "$SITE/vllm/models/glm5next/common/recoverssm.py" \
    "$SITE/vllm/model_executor/kernels/mhc/tilelang_kernels.py" \
    "$SITE/vllm/model_executor/kernels/mhc/tilelang.py" \
    "$SITE/vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py" \
    "$SITE/vllm/v1/attention/backends/mla/gb10_sparse_mla.py" \
    "$SITE/vllm/v1/attention/backends/gdn_attn.py" \
    "$SITE/vllm/model_executor/layers/mamba/ops/causal_conv1d.py" \
    "$SITE/vllm/v1/core/kv_cache_utils.py" \
    "$SITE/vllm/v1/core/kv_cache_coordinator.py" \
    "$SITE/vllm/v1/kv_cache_interface.py" \
    "$SITE/vllm/parser/glm47_moe.py" \
    "$SITE/flashinfer/mla/_batch_mla/_backends/_fa_common.py"; do
    python3 -m py_compile "$f"
done
echo "$PREFIX py_compile OK on all staged python overlays"

# FlashInfer autotune cache is EPHEMERAL at TP>1: a persisted cache deadlocks
# the next boot (rank 0 saves per-rank keys, ranks 1+ miss and wait on a CPU
# group all_reduce forever). Upstream wipes it on every start; do the same.
# The env pointing vLLM at /tmp lives in the recipe env (VLLM_FLASHINFER_
# AUTOTUNE_CACHE_DIR): mod env does not reach the serve process.
rm -rf /tmp/flashinfer_autotune
echo "$PREFIX autotune cache cleared (ephemeral, TP>1 rule)"

# The recipe points the image's spark memory-trace stats at the runtime cache
# (upstream's /logs bind has no sparkrun counterpart; TORCH_MEM_STATS_DIR env
# lives in the recipe). Pre-create it so the tracer never no-ops on a missing
# directory.
mkdir -p /cache/runtime/torch-mem-stats

# Clear stale bytecode so the engines reload the staged files.
find "$VLLM" "$FLASHINFER" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

# The display-KV backing needs the image's prebuilt helper + host /dev/dri.
if [ "${GLM53_DISPLAY_KV_ENABLE:-0}" = "1" ] && [ ! -f /opt/glm53-display-kv/libglm53_display_kv.so ]; then
    echo "$PREFIX WARN: display-KV enabled but image helper is missing" >&2
fi

# Recap in the same vocabulary as upstream's entrypoint overlays() gate:
su=""
[ -e "$VLLM/distributed/device_communicators/arx.py" ] && su="$su arx"
[ -e "$VLLM/model_executor/model_loader/weight_snapshot.py" ] && su="$su snapshot"
[ -e "$VLLM/v1/core/sched/adaptive_k.py" ] && su="$su adaptive-k"
[ -e "$VLLM/model_executor/model_loader/dense_fp8.py" ] && su="$su fp8"
[ -e "$VLLM/model_executor/layers/fused_moe/megamoe_vllm.py" ] && su="$su megamoe"
[ -e "$VLLM/v1/attention/backends/mla/gb10_sparse_mla.py" ] && su="$su fixes"
[ -e "$VLLM/models/glm5next/common/recoverssm.py" ] && su="$su recoverssm"
echo "=== OK: technigmaai native overlays staged ($su) ==="
