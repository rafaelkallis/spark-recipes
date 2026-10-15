# technigmaai-native-3a119df

Stages the full vendored overlay set of
[technigmaai/glm-5.3-flash-gb10-tp2-native](https://github.com/technigmaai/glm-5.3-flash-gb10-tp2-native)
into the pinned runtime image — the sparkrun equivalent of its ~33 read-only
compose bind mounts (plus the MiaAI-derived chat template, the ephemeral
FlashInfer autotune wipe, and a pre-created spark memory-trace stats directory
under the runtime cache).

## Provenance (pinned)

| Artifact | Pin |
|---|---|
| Upstream repo | `3a119df75e771374dacc3fb0f73373a0558aa84a` (main, 2026-10-10) |
| Image | `technigmaai/glm-5.3-flash-gb10-tp2-native:c748079-displaykv1-arm64-cu130` |
| Image digest | `sha256:46afd8bba26106aa7ea6e00f6fd4bd3ba3d0c7c49e86af85f269eb2c88c71fa5` |
| vLLM in image | `0.30.1rc1.dev193+gddd6fbca1` (torch 2.13.0+cu130, FlashInfer 0.7.0) |
| Target model | `nvidia/GLM-5.3-Flash-NVFP4` @ `da920bb0b9f4a06727223a349e55468e38352348` |
| Drafter | `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4eacc1810f76656d1811693ff6c6737d2a` (CC BY-NC-ND) |

The overlays are file-level edits against that exact image; the source tree is
vendored verbatim (1.3 MB) at the pinned commit. `run.sh` anchor-checks every
target directory against the pinned module layout — plus the image's
`weight_cache` package, an import target of the staged snapshot loader
(`weight_cache.protocol`) — and py-compiles everything it stages — a wrong
image fails in seconds instead of after a ~25 min load. If upstream bumps its
overlay set, re-vendor from the new commit and bump the mod name.

## What gets staged where (SITE = /usr/local/lib/python3.12/dist-packages)

| Vendored file | Target in image | Overlay |
|---|---|---|
| `overlays/arx/arx.py` | `vllm/distributed/device_communicators/arx.py` | arx |
| `overlays/arx/arx_vllm.cu` | `vllm/distributed/device_communicators/arx_vllm.cu` | — |
| `overlays/arx/arxbig.cu` | `vllm/distributed/device_communicators/arxbig.cu` | — |
| `overlays/arx/cuda_communicator.py` | `vllm/distributed/device_communicators/cuda_communicator.py` | — |
| `overlays/snapshot/weight_snapshot.py` | `vllm/model_executor/model_loader/weight_snapshot.py` | snapshot |
| `overlays/snapshot/base_loader.py` | `vllm/model_executor/model_loader/base_loader.py` | — |
| `overlays/snapshot/dense_fp8.py` | `vllm/model_executor/model_loader/dense_fp8.py` | — |
| `overlays/snapshot/model_source.py` | `vllm/model_executor/model_loader/model_source.py` | — (from upstream scripts/) |
| `overlays/snapshot/modelopt.py` | `vllm/model_executor/layers/quantization/modelopt.py` | — |
| `overlays/snapshot/flashinfer_cutlass_moe.py` | `vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py` | — |
| `overlays/adaptive-k/adaptive_k.py` | `vllm/v1/core/sched/adaptive_k.py` | adaptive-k |
| `overlays/draft-trunc/draft_trunc.py` | `vllm/v1/worker/gpu/spec_decode/draft_trunc.py` | draft-trunc |
| `overlays/draft-trunc/rejection_sampler.py` | `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` | — |
| `overlays/draft-trunc/speculator.py` | `vllm/v1/worker/gpu/spec_decode/speculator.py` | — |
| `overlays/draft-trunc/dflash2_speculator.py` | `vllm/v1/worker/gpu/spec_decode/dflash2/speculator.py` | — |
| `overlays/draft-trunc/grouped_topk_router.py` | `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py` | — |
| `overlays/draft-trunc/model_runner.py` | `vllm/v1/worker/gpu/model_runner.py` | — |
| `overlays/megamoe/megamoe_vllm.py` | `vllm/model_executor/layers/fused_moe/megamoe_vllm.py` | megamoe |
| `overlays/megamoe/megadense4.cu` | `/opt/megamoe/megadense4.cu` | — (JIT at boot) |
| `overlays/megamoe/megamoe_rm.cu` | `/opt/megamoe/megamoe_rm.cu` | — |
| `overlays/megamoe/moe_prefill.cu` | `/opt/megamoe/moe_prefill.cu` | — |
| `overlays/fixes/_fa_common.py` | `flashinfer/mla/_batch_mla/_backends/_fa_common.py` | fixes |
| `overlays/fixes/attention.py` | `vllm/models/glm5next/common/attention.py` | — |
| `overlays/fixes/kda.py` | `vllm/models/glm5next/common/kda.py` | — |
| `overlays/fixes/model.py` | `vllm/models/glm5next/common/model.py` | — |
| `overlays/fixes/recoverssm.py` | `vllm/models/glm5next/common/recoverssm.py` | — |
| `overlays/fixes/tilelang_kernels.py` | `vllm/model_executor/kernels/mhc/tilelang_kernels.py` | — |
| `overlays/fixes/mhc_tilelang.py` | `vllm/model_executor/kernels/mhc/tilelang.py` | — |
| `overlays/fixes/flashinfer_mla_sparse_sm90.py` | `vllm/v1/attention/backends/mla/flashinfer_mla_sparse_sm90.py` | — |
| `overlays/fixes/gb10_sparse_mla.py` | `vllm/v1/attention/backends/mla/gb10_sparse_mla.py` | — |
| `overlays/fixes/gdn_attn.py` | `vllm/v1/attention/backends/gdn_attn.py` | — |
| `overlays/fixes/causal_conv1d.py` | `vllm/model_executor/layers/mamba/ops/causal_conv1d.py` | — |
| `overlays/fixes/kv_cache_utils.py` | `vllm/v1/core/kv_cache_utils.py` | — |
| `overlays/fixes/kv_cache_coordinator.py` | `vllm/v1/core/kv_cache_coordinator.py` | — |
| `overlays/fixes/kv_cache_interface.py` | `vllm/v1/kv_cache_interface.py` | — |
| `overlays/fixes/warmup.py` | `vllm/v1/worker/gpu/warmup.py` | — |
| `overlays/fixes/glm47_moe.py` | `vllm/parser/glm47_moe.py` | ` (thinking-flag selection matched to the template) |
| `display-kv/worker_utils.py` | `vllm/v1/worker/utils.py` | display-KV (gated by env) |
| `chat_template.jinja` | `/opt/glm53-native/chat_template.jinja` | template (recipe `--chat-template`) |

## Env coupling (set in the recipe, not here)

Overlays activate via env validated against upstream's compose. The recipe
carries the full set; key notes:

- `VLLM_ADAPTIVE_K_MODEL=32.1,1.32,0.427` — the entrypoint's TP2 + RecoverSSM
  fit (compose's empty value resolves through the TP2 case default). `COST_MS`
  and `ONLINE=1`/control dir complete the adaptive-k stack.
- `VLLM_ARX_*` / `VLLM_GLM_SP_*` / `VLLM_MEGAMOE*` / `VLLM_MHC_FUSED_MAX_TOKENS`
  — as upstream compose. `arx.py` additionally reads sparkrun-managed
  `NCCL_IB_HCA` / `NCCL_IB_GID_INDEX` (never set them in the recipe).
- Persistent control/cache dirs are re-pointed to sparkrun's runtime cache:
  `VLLM_WEIGHT_SNAPSHOT_DIR=/cache/runtime/weight-snapshots` (~96 GiB/rank on
  first build, big boot-time win after), `VLLM_ADAPTIVE_K_CONTROL` and
  `VLLM_DRAFT_TRUNC_CONTROL` under `/cache/runtime/`, `TILELANG_CACHE_DIR`
  under `/cache/runtime/` (sparkrun does not manage TileLang; its
  TRITON/VLLM/FLASHINFER caches are already managed — do not duplicate them).
- `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=/tmp/flashinfer_autotune` + the mod's
  wipe: at TP>1 a persisted autotune cache deadlocks every subsequent boot
  (upstream measured this); ~2 min of autotune per boot is the accepted cost.
- `GLM53_DISPLAY_KV_ENABLE=0` default: the display-backed KV (1.75 GiB inside
  the 8 GiB pool, via the image's `libglm53_display_kv.so`) needs host
  `/dev/dri/card0` + DRM group and a stopped display manager. Enable only in
  headless-verified deployments.

## Not ported (upstream-specific)

- `files/entrypoint.sh` orchestration: status server, self-test gate,
  fabric/RoCE pre-flight and GID discovery, `NATIVE_DRY_RUN` command emit, and
  the `GLOO_SOCKET_IFNAME`/`NCCL_*` derivation — sparkrun supplies per-node
  ranks, rendezvous and the RoCE comm env (`NCCL_IB_*`, sockets), and `arx.py`
  reads those sparkrun-provided values directly.
- `SNAPSHOT_SEED_DIR` bind + `VLLM_WEIGHT_SNAPSHOT_SEED_DIR`: the snapshot
  selector treats an unset seed dir as "none" and
  `VLLM_WEIGHT_SNAPSHOT_NORMAL=1` skips the seed lookup entirely.
- `ALLOW_STOCK` gate, model aliases beyond the served name, tool-parser
  fail-closed plugin (upstream ships it, only for degraded upstream stacks).

## Licenses

Overlay files carry their upstream notices (vLLM SPDX Apache-2.0 headers
retained; `overlays/megamoe/LICENSE` for the MegaMoE sources; display-KV
component under `files/display-kv/` is AGPL-3.0 — this mod stages only its
`worker_utils.py`, which is already inside the published image).

The governing license texts and attribution notes are vendored beside their
artifacts, mirroring upstream's layout:

| File | Covers |
|---|---|
| `licenses/Apache-2.0.txt` | the staged vLLM/FlashInfer-channel overlays |
| `display-kv/LICENSE.AGPL-3.0` | `display-kv/worker_utils.py` |
| `display-kv/ORIGIN.md` | display-KV derivation (coolbho3k) + adaptation notes |
| `chat_template.ORIGIN.md` | MiaAI chat-template attribution + checksum |

Model/drafter weights are not included anywhere in this repo.
