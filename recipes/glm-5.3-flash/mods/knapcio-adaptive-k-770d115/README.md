# knapcio-adaptive-k-770d115

Adaptive draft-length scheduling for the DFlash2 lane, staged as a sparkrun
mod onto the `ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2` image
(vLLM `0.1.dev20051+g487ecf187`). Fixes the fixed-k=7 tax: free-form prose
accepts few draft positions, so drafting+verifying 7 tokens every step loses
to plain decoding; code and structured output accept long prefixes and want
the full k. The scheduler makes the per-step draft/verify length adaptive from
a per-request acceptance EMA.

| File | Role |
|---|---|
| `adaptive_k_scheduler.py` | `AdaptiveKScheduler` — per-request VERIFY length; subclasses `vllm.v1.core.sched.async_scheduler.AsyncScheduler` and overrides `_update_after_schedule` (hook verified present in the v11 image). |
| `adaptive_draft_scheduler.py` | `AdaptiveDraftScheduler` — additionally picks how many tokens the DRAFTER produces per step (max over running decode requests of the policy decision); replaces the batch-size→K lookup. |

Both files are copied **verbatim** from
[knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4](https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4)
at pinned commit `770d1153062aa916b06591411c61d5c593ea0f03`
(Apache-2.0; separable MIT modifications by Jacopo Nardiello — see upstream
`LICENSES/MIT-Jacopo-Nardiello.txt`). Provenance: tonyd2wild's
`runs/2026-09-29-knapcio-tp2` documents this exact scheduler pair as the
decode-side stack on their TP2 port.

## What ships / what doesn't

The scheduler only imports `vllm.logger` and
`vllm.v1.core.sched.async_scheduler` — no compiled extensions, no glm5next
model patches. This is the portable 2-file subset of knapcio's 80+-file
overlay (their full profile additionally uses `glm_levers_sched.LeversScheduler`,
draft truncation tables, device-side verify-shape selection, prefill
scheduling — all overlay-local).

## Wiring (in the recipe, not the mod)

sparkrun executes a mod's `run.sh` and the `vllm serve` command in **separate
docker execs**, so nothing exported in `run.sh` reaches the engine. All
knobs live in the recipe:

```yaml
mods:
  - tonyd2wild-20261006
  - knapcio-adaptive-k-770d115

defaults:
  # launch table must cover EVERY verify depth 1..k_hi so FULL CUDA graphs
  # exist for all rejection lengths (upstream gotcha: missing depths cause
  # DEVSELECT_BUILD_FAILED-style fallbacks and -48% single-stream)
  speculative_config: >-
    {"method":"dflash","model":"incoai/GLM-5.3-Flash-DFlash2",
     "num_speculative_tokens":7,
     "num_speculative_tokens_per_batch_size":[[1,1,7],[2,2,5],[3,27,3],[28,28,6],[29,29,4],[30,30,2],[31,32,1]]}

env:
  PYTHONPATH: /cache/runtime/adaptive-k-mod/overlay
  VLLM_ADAPTIVE_K_LO: '3'
  VLLM_ADAPTIVE_K_HI: '7'
  VLLM_ADAPTIVE_K_MODE: per-request
  VLLM_ADAPTIVE_K_SIGNAL: pos
  VLLM_ADAPTIVE_K_DOWN: '0.42'
  VLLM_ADAPTIVE_K_UP: '0.58'
  VLLM_ADAPTIVE_K_ALPHA: '0.15'
  VLLM_ADAPTIVE_K_SEED: '1.0'

command: |
  ... --scheduler-cls adaptive_draft_scheduler.AdaptiveDraftScheduler ...
```

`run.sh` fail-closes on missing knobs and on a vLLM without the scheduler
hook — a misconfigured boot aborts before `vllm serve` starts.

## Semantics (from the source docstrings)

- **Verify trimming (base class):** each request verifies `k_req ∈ {k_lo, k_hi}`
  positions per step, chosen by an acceptance EMA
  (`VLLM_ADAPTIVE_K_UP`/`_DOWN` hysteresis, `_ALPHA` smoothing). Speculation
  never changes outputs.
- **Draft production (subclass):** step K = max over running decode requests of
  `policy.decide(req)` — a lone prose stream drafts+verifies k_lo (3), a lone
  code stream k_hi (7), a mixed batch drafts k_hi while prose requests verify
  k_lo.
- `VLLM_ADAPTIVE_K_SEED=1.0` starts every new request in the high state
  (optimistic), decaying if acceptance disappoints.
- Optional batch cap: `VLLM_ADAPTIVE_DRAFT_CAP="3:5"` = at batch ≥ 3, draft at
  most 5 (not wired in our recipe).
