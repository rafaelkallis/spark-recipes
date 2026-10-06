# tonyd2wild-20261006

Two file-level patches for the `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2`
image (vLLM `0.1.dev20051+g487ecf187`), staged by a sparkrun mod because sparkrun
cannot replicate the launcher's per-file bind mounts:

| File | Target | Purpose |
|---|---|---|
| `sparse_attn_indexer_kpool_sm121.py` | `vllm/model_executor/layers/sparse_attn_indexer_kpool.py` (full replacement) | SM121 top-k persistent-kernel fix. Without it the engine **hard-crashes on any decode past ~24K context**. |
| `patch_prefix_cache_draft_group.py` | `vllm/v1/core/kv_cache_coordinator.py` (in-place, anchored) | #18 prefix-cache repair for the DFlash2 draft group: stop the draft sliding-window group from zeroing the shared cache hit. Repeat-prefix hit rate 0 → 0.986 on the 262K probe; 13K-token repeat TTFT 21.1 s → 6.3 s. |

Both files are copied **verbatim** from
[tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark](https://github.com/tonyd2wild/GLM-5.3-Flash-NVFP4-DFlash2-2x-DGX-Spark)
at pinned HEAD `d061f26ad3ec5c3c04f64aad7516dbe62a8aa6af` (the launcher's own
`$HOME/patches/` staging step downloads these same files from the repo):

- `docker/sparse_attn_indexer_kpool_sm121.py` (Apache-2.0 header, vLLM-derived)
- `docker/dflash2-overlay/patch_prefix_cache_draft_group.py`

The upstream scripts are fail-closed: anchored string replacements that abort on
drift, and the patch script self-checks its draft-group predicate against the
installed vLLM before declaring success — a mismatch fails the pod boot before
`vllm serve` starts.

## Upstream references

- `launch-glm53-vllm-tp2-dflash2.sh` (the verified TP2 DFlash2 launcher, the
  46.9 tok/s configuration) — its `PREFIX_FIX` and `SM121 patch` bind mounts.
- `docs/SM121-CRASH-FORENSICS-2026-08-27.md` (crash the kpool file fixes)
- tp2 `README.md` §Current default (2026-09-29).

## Not included (host-level, cannot port into a sparkrun mod)

- The b12x RoCE all-reduce bundle (`speed-night-2026-09-18/roce/`,
  `build-roce-bundle.sh`) — host-build per node; without it the launcher
  degrades to stock NCCL, which is what sparkrun's network management provides
  anyway.
- knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4's overlay stack (adaptive draft
  scheduler, RoCEnante image, lossless8 checkpoints, FP8 drafter) — those
  require a locally-built image and locally-converted weights; see
  `runs/2026-09-29-knapcio-tp2/RUNBOOK.md` upstream.
