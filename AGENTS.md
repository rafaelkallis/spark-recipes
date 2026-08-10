# AGENTS.md

This repository is a **sparkrun recipe registry**. It hosts YAML recipes used to
serve LLMs on NVIDIA DGX Spark hardware via [sparkrun](https://sparkrun.dev/).
Anyone can add this repo as a registry with `sparkrun registry add <url>` and run
its recipes by name.

## Repository layout

Per the [sparkrun registry spec](https://sparkrun.dev/recipes/registries/#creating-a-registry):

```
.sparkrun/
  registry.yaml        # REQUIRED manifest; without it registry auto-discovery fails
recipes/              # recipe YAML files (flat by default)
tuning/              # optional Triton kernel tuning configs
benchmarking/        # optional benchmark profile YAML files
mods/               # optional shared mods (run.sh + supporting files)
README.md
```

## Registry manifest (`.sparkrun/registry.yaml`)

```yaml
registries:
  - name: rafaelkallis
    description: Rafael's recipes
```

- Registry name is `rafaelkallis` → recipes are invoked as `@rafaelkallis/<recipe>`.
- **Do not use reserved prefixes** (`sparkrun`, `official`, `arena`,
  `spark-arena`, etc.) — `sparkrun registry add` rejects reserved prefixes unless the
  repo is hosted under an approved GitHub organization.
- The `recipes` field defaults to `recipes`; keep manifest minimal unless tuning/
  benchmarking/mods subdirectories are added. Both canonical keys (`subpath`) and short
  aliases (`recipes`) are supported; canonical wins.

## Recipe format

Each recipe in `recipes/` follows the
[sparkrun recipe format](https://sparkrun.dev/recipes/format/). Current recipes use
the `vllm` runtime and typically declare:

- `model` — HF repo id of the served model
- `runtime` — serving runtime (`vllm`, `sglang`, `tensorrt-llm`, `llama-cpp`, `atlas`)
- `container` — image used to serve the model
- `min_nodes` / `tensor_parallel` / `pipeline_parallel` layout
- `defaults` — ports, host, `max_model_len`, `max_num_batched_tokens`,
  `gpu_memory_utilization`, `speculative_config`, `load_format`, etc.
- `metadata.description` and `metadata.maintainer`
- `env` — runtime environment variables (e.g. `VLLM_*` flags)
- `command` — optional explicit `vllm serve ...` command; otherwise the runtime
  generates default CLI flags from `defaults`

### Conventions when editing recipes

- Keep a `metadata` block with a concise `description` and `maintainer`.
- Follow the existing style: `model`, `runtime`, `container`, `defaults`, `env`,
  `command`.
- Prefer `.yml` consistently (existing files use `.yml`).
- Do NOT invent models, containers, or paths that do not exist. Verify HF model ids,
  container tags, and cache keys against the actual repo contents.
- When adding commented-out alternatives (e.g. multiple container tags or speculative
  configs), keep them above the active line with `# ` annotations, as the existing
  recipes do.
- Never repeat a `defaults` key — YAML silently uses the last occurrence, so a
  duplicate key is a dead line that hides drift (e.g. a double
  `max_cudagraph_capture_size`).
- For embedding/pooling recipes, cap `max_cudagraph_capture_size` to the expected
  batch-1 context (e.g. 256 for 4k context) — vLLM generates capture sizes up to that
  bound, so values past the max context only add startup time. sparkrun does **not**
  set `VLLM_CACHE_ROOT`; vLLM's compile/FP4-GEMM autotune cache is not persisted
  across launches out of the box.

## Recipe discovery rules

Recipes are resolved in this order:
`@spark-arena/UUID` shortcut → URL → `@registry/name` → file path → CWD scan →
registry search. Within a registry:

- A flat `<recipes>/<name>.yaml` beats a recursive scan of the same directory.
- `.yaml` beats a same-stem `.yml` in the same directory (use one, not both).

Avoid introducing name collisions that make a recipe ambiguous.

## Validation workflow

When you change a manifest or recipe, verify against the CLI (if available):

```bash
sparkrun recipe show @rafaelkallis/DeepSeek-V4-Flash
sparkrun list
```

There is no build step or test suite in this repo — validation is done by adding the
local registry, listing it, and (optionally) running a recipe on a live cluster.
