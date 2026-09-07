# mia-957890ac

sparkrun mod that applies the **MiaAI-Lab DSpark hotfix stack** for
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` on the **Anemll** image.

- Upstream: <https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark> (MIT)
- Pin: `957890ac5e26ce149646719298169f80603a3e1f`
- Target image: `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
  (vLLM `0.25.2.dev0+g752a3a504.d20260714`)

## Why this exists

The upstream project ships its fixes as a docker-compose entrypoint that runs
~20 patch scripts against the in-image vLLM tree before `exec vllm serve`.
sparkrun has no compose entrypoint, so that pre-exec sequence lives here as a
mod instead. Without it the Anemll image cannot serve this model: the DSML
encoder is not in the image, the vision tower is not registered, and
`nvfp4_ds_mla` decode falls back to a slow path.

## Not interchangeable with the eugr mods

`mods/mia-aa25c89-deepseek-v4-flash-vision-exp/` is an **eugr-nightly** port
with locally adapted `vision_exp/` sources. This mod targets the Anemll image
and uses upstream's patch sources unmodified. Do not mix them in one recipe.

## Provenance contract

Every model-affecting step in `run.sh` comes from the MiaAI repo at the pin:
the patch tree is **fetched and digest-verified**, and the reasoning-effort
snippet is the entrypoint's inline python kept byte-identical. Audited
2026-09-07 against the extracted compose entrypoint: all 18 invoked hotfix
files appear in upstream's own sequence (same order), and every recipe env
variable is upstream-defined.

The only locally-authored pieces are launch plumbing, not model content:

- the cache → pinned-fetch resolver (how the mod obtains upstream content),
- the **fail-closed encoder guard** — upstream WARNs and boots without the
  encoder; here a missing `encoding_dsv4.py` aborts the launch instead.

Any future model-behavior customization (own hotfixes, adapted sources) must
live in a **separate mod**, never in this one — see
`mia-*-deepseek-v4-flash-vision-exp/` for the eugr-nightly ports that follow
that rule.

## Layout

| Path | Provenance |
| --- | --- |
| `run.sh` | Ours. Resolves the patch tree (cache → pinned fetch), then ports the always-on portion of the compose entrypoint — including the entrypoint's inline `python3 -c` reasoning-effort snippet, kept byte-identical in a heredoc. |

## How `run.sh` works

`run.sh` is deliberately thin — upstream's compose is a full bash entrypoint,
so the mod **reuses it instead of hand-porting it**:

1. **Resolve the upstream snapshot** (cache → pinned codeload fetch). The
   commit SHA fully pins the fetch (codeload archives are content-addressed
   by commit); one whole-snapshot digest exists only to guard the persisted
   cache from silent corruption between launches.
2. **Stage the `/opt` bindings by parsing upstream's own compose volume list**
   (regex over `${VAR:-./path}:/opt/...:ro`) — when upstream adds a hotfix
   binding, it flows in through the pin, no edits here.
3. **Extract the entrypoint script** from the pinned compose's `command:`
   (docker-compose `$$` un-escaped), dropping only the trailing
   `exec vllm serve …` line — sparkrun builds the serve command.
4. **Execute it** — hotfix selection, ordering, `DSPARK_SKIP_*` escape
   hatches and env-var handling are upstream's code, not ours.
5. **Fail closed on a missing encoder.** Upstream WARNs and boots without the
   model's encoder; here the boot aborts instead (a silent encoder-less boot
   changes tool-call/reasoning token encoding).

Upstream snapshot cache: `$DSPARK_PATCH_CACHE_DIR` (default
`/cache/runtime/dspark-mod`)/`mia-<sha8>/`, survives container recreation; a
cold cache needs GitHub egress once per node per pin (~2.4 MB).

## What used to be hand-ported (now upstream's)

Before 2026-09-07 this mod contained a hand-ported always-on hotfix sequence
(~18 invocations) and a local copy of the entrypoint's inline python. It now
executes upstream's entrypoint verbatim instead; the description of the
always-on set below reflects what that entrypoint applies (issue55 →
nvfp4-ds-mla-issue22 → gb10-spin-wait → issue117 (+`--status`) → six perf
backports → vision-exp → empty-encoder-output → issue27 → issue43 → issue26 →
issue133 → suppress-stops-in-reasoning), but the authoritative source is the
pinned compose, not this file.

### Deliberate deviations from upstream

- **Missing encoder is fatal.** Upstream logs `WARN` and continues; here it exits
  non-zero. Serving without the model's encoder silently changes tool-call and
  reasoning token encoding, which is worse than a failed boot.
- **The serve line is dropped from the entrypoint** (sparkrun builds it) — as
  a consequence the entrypoint's TP=3 and GB10-plugin paths cannot run (they
  only matter when the serve line changes). API-key redaction and the
  Responses-API store are upstream opt-ins that stay off because the recipe
  doesn't set those variables.

### Environment

Read from the recipe's `env` block:

| Variable | Effect |
| --- | --- |
| `DSPARK_PATCH_CACHE_DIR` | Root for the persisted upstream-snapshot cache. Default `/cache/runtime/dspark-mod`. |
| `DSPARK_MODEL` | HF repo id used to find the encoder snapshot. Default `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`. |
| `DSPARK_REVISION` | Preferred snapshot revision. Should match the recipe's `revision`. |
| `DSPARK_ENCODING_FILE` | Explicit override for the encoder path. |
| `VLLM_ROOT` | vLLM tree. Default `/usr/local/lib/python3.12/dist-packages/vllm`. |
| `DSPARK_HF_HUB_DIR` | HF hub dir. Default `/cache/huggingface/hub`, which is where sparkrun mounts it. |
| `DSPARK_SKIP_*`, `DSPARK_ENABLE_*`, `DSPARK_API_KEYS`, … | Upstream's own entrypoint env; all honored as upstream defines them. |

## Bumping the pin

This mod is pinned to `957890ac`; do not re-point it at a different commit.
A new upstream commit means a **new mod directory**:

1. Copy this directory to `mods/mia-<new-sha8>/`.
2. In the copy's `run.sh`: set `UPSTREAM_SHA` and `SNAPSHOT_MANIFEST_SHA256`
   (LC_ALL=C manifest digest over the extracted tree *excluding*
   `entrypoint.sh`, which the script generates itself) and update the header
   comment's pin line.
3. Review the compose diff between the two pins (entrypoint + volume list
   changes flow in automatically — that is the point of this design).
4. Point the new recipe's `mods:` list at the new directory.

Older pins (e.g. `mia-aa25c89-…`) stay as-is while a recipe still references them.

Also re-check the image: these patches are source-anchored to
`0.25.2.dev0+g752a3a504.d20260714` and fail closed on a mismatch, so an Anemll
image bump needs an upstream pin that supports it.
