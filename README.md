# Spark Recipes

A [sparkrun](https://sparkrun.dev/) recipe registry for serving LLMs on **NVIDIA DGX Spark** hardware.

This repo hosts battle-tested recipes that bring up the best open-weight models on
DGX Spark with `vLLM`, including fp8 KV caches, Tensor Parallel, and model-native
speculative decoding (DeepSeek-style draft, MTP) for maximal throughput.

> Register it with sparkrun and run any recipe by name — no manual cluster setup.

## Quick Start

Add this repository as a registry to sparkrun:

```bash
sparkrun registry add https://github.com/rafaelkallis/spark-recipes
```

List all available recipes:

```bash
sparkrun list
```

Inspect a recipe before running:

```bash
sparkrun recipe show @rafaelkallis/DeepSeek-V4-Flash
```

Run a recipe on your DGX Spark (2+ nodes):

```bash
sparkrun run @rafaelkallis/DeepSeek-V4-Flash
```

## Recipes

| Recipe | Model | Description |
| --- | --- | --- |
| `DeepSeek-V4-Flash` | [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | Sparse MoE Flash model with DeepSeek endogenous speculative decoding and instanttensor load |
| `Qwen3.5-122B-A10B-Prisma` | [`rdtand/Qwen3.5-122B-A10B-PrismaQuant-4.75bit-vllm`](https://huggingface.co/rdtand/Qwen3.5-122B-A10B-PrismaQuant-4.75bit-vllm) | Qwen3.5 MoE model, Prisma-quantized to 4.75 bits, with MTP draft model |

### `DeepSeek-V4-Flash`

The smallest member of the DeepSeek-V4 family, designed as a single-rack Flash model.

- **Runtime:** vLLM
- **Nodes:** 2 (tensor parallel 2, pipeline parallel 1)
- **Context:** up to 1M tokens
- **Speculative decoding:** endogenous DeepSeek draft model (`dspark`, 7 speculative tokens, greedy)
- **KV cache:** fp8 at block size 256
- **Model loading:** `instanttensor` for fast startup

### `Qwen3.5-122B-A10B-Prisma`

A 122B-parameter MoE serving with only 10B active parameters per token.

- **Runtime:** vLLM
- **Nodes:** 2 (tensor parallel 2, pipeline parallel 1)
- **Context:** up to 256K tokens
- **Speculative decoding:** MTP draft model (3 speculative tokens)
- **KV cache:** fp8
- **Model loading:** `instanttensor`

## Repository Layout

```
.sparkrun/
  registry.yaml   # registry manifest (name: rafaelkallis)
recipes/
  DeepSeek-V4-Flash.yml
  Qwen3.5-122B-A10B-Prisma.yml
README.md
```

## Adding Your Own Recipe

Recipes are plain YAML. See the
[sparkrun recipe format](https://sparkrun.dev/recipes/format/) docs, or copy one:

```bash
cp recipes/DeepSeek-V4-Flash.yml recipes/My-Model.yml
# edit model, container, defaults, and metadata
```

Then verify it resolves:

```bash
sparkrun recipe show @rafaelkallis/My-Model
```

## Contributing

PRs welcome! Keep the metadata `description` concise, use `.yml`, verify model ids
and container tags against upstream, and confirm with `sparkrun list` after changes.

## License

[Apache 2.0](LICENSE)
