"""vLLM multimodal processor for DeepSeek-V4-Flash-Vision-Exp (current nightly).

Images only. The checkpoint has no video encoder; GIF is a still frame via PIL.

Compatibility: vLLM >= 0.1.dev20489 (2026-09-04 nightly) removed the
``_call_hf_processor`` hook from ``BaseMultiModalProcessor`` and rewrote
``_apply_hf_processor_main`` around ``info.get_hf_processor()`` +
``ctx.call_hf_processor(...)``. This model has no HF processor, so we bypass
that path entirely:

* ``get_mm_max_tokens_per_item`` returns the capped per-image token count so
  ``MultiModalBudget`` never synthesizes dummy inputs to probe the processor.
* ``_apply_hf_processor`` is overridden to run the whole-prompt expansion
  directly (``_build_image_features``). ``compress_pad`` depends on the
  expanded token position of each image, so a content-only processor cache
  would reuse the wrong layout; the whole-prompt path keeps it exact and
  uncached, mirroring the pre-#53275 design the mod was originally written for.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

from vllm.inputs import MultiModalDataDict
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
    TimingContext,
)
from vllm.multimodal.processing.processor import MultiModalProcessingInfo

from .image_processor import (
    IMAGE_PAD,
    IMAGE_PLACEHOLDER,
    as_pil,
    build_image_block,
    pil_to_patches,
    vision_args_from_config,
)


def _image_token_id(tokenizer) -> int:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is not None:
        token_id = convert(IMAGE_PLACEHOLDER)
        if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
            return int(token_id)
    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
    if IMAGE_PLACEHOLDER in vocab:
        return int(vocab[IMAGE_PLACEHOLDER])
    # Vision-Exp added-token id (vocab_size 129280; placeholder is in the tail).
    return 129264


def _collect_images(mm_data: Mapping[str, object]) -> list[Any]:
    for key in ("images", "image"):
        value = mm_data.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]
    return []


class DeepseekV4VisionExpProcessingInfo(BaseProcessingInfo):
    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": 16}

    def get_hf_processor(self, **kwargs: object):
        raise RuntimeError(
            "DeepSeek-V4-Flash-Vision-Exp has no Hugging Face processor; "
            "the vLLM Vision-Exp processor handles images directly."
        )

    def get_max_image_tokens(self) -> int:
        return int(getattr(self.get_hf_config(), "vision_max_n_token", 384))

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        # Let MultiModalBudget size the encoder cache without synthesizing
        # dummy inputs (which would route into get_hf_processor). Every image
        # is safely resized so its block length is at most vision_max_n_token,
        # so the cap is the correct worst-case budget.
        return {"image": self.get_max_image_tokens()}


class DeepseekV4VisionExpDummyInputsBuilder(
    BaseDummyInputsBuilder[DeepseekV4VisionExpProcessingInfo]
):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        return IMAGE_PLACEHOLDER * mm_counts.get("image", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, Any],
    ) -> MultiModalDataDict:
        num_images = mm_counts.get("image", 0)
        image_overrides = mm_options.get("image")
        # Oversized square; official safe_resize caps at vision_max_n_token.
        return {
            "image": self._get_dummy_images(
                width=2048,
                height=2048,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class DeepseekV4VisionExpMultiModalProcessor(
    BaseMultiModalProcessor[DeepseekV4VisionExpProcessingInfo]
):
    def _hf_processor_applies_updates(
        self,
        prompt_text: str,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        tokenization_kwargs: Mapping[str, object],
    ) -> bool:
        # Kept for lineages/versions that still honor the flag. The model has
        # no HF processor, so prompt updates are always produced by vLLM.
        return False

    def _cached_apply_hf_processor(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalProcessingInfo:
        # compress_pad depends on the expanded token position of each image,
        # so a content-only processor cache would reuse the wrong layout.
        if inputs.mm_data_items.get_count("image", strict=False) > 0:
            return self._apply_hf_processor(inputs, timing_ctx)
        return super()._cached_apply_hf_processor(inputs, timing_ctx)

    def _apply_hf_processor(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalProcessingInfo:
        """Run the vision-exp whole-prompt expansion directly.

        Overrides the base implementation so image processing never goes
        through ``info.get_hf_processor()`` / ``ctx.call_hf_processor`` (this
        model has no HF processor). The prompt token ids from ``inputs.prompt``
        already contain the ``<｜deepseek_image｜>`` placeholder tokens, so we
        can compute each image's position-dependent block layout exactly.
        """
        valid_mm_items = inputs.mm_data_items.select(
            {k for k, c in inputs.mm_data_items.get_all_counts().items() if c > 0}
        )
        processor_data, passthrough_data = self._get_hf_mm_data(valid_mm_items)
        images = _collect_images(processor_data)

        with timing_ctx.record("apply_hf_processor"):
            if images:
                mm_processed_data = self._build_image_features(inputs.prompt, images)
                mm_processed_data.update(passthrough_data)
            else:
                from transformers.feature_extraction_utils import BatchFeature

                mm_processed_data = BatchFeature(dict(passthrough_data))

            mm_kwargs = MultiModalKwargsItems.from_hf_inputs(
                mm_processed_data,
                self._get_mm_fields_config(
                    mm_processed_data, inputs.hf_processor_mm_kwargs
                ),
            )

        with timing_ctx.record("get_mm_hashes"):
            mm_hashes = inputs.get_mm_hashes(
                self.info.model_id,
                self.info.ctx.get_mm_config().mm_hasher_algorithm,
            )

        mm_prompt_updates = self._get_mm_prompt_updates(
            inputs.mm_data_items,
            inputs.hf_processor_mm_kwargs,
            mm_kwargs,
        )

        return MultiModalProcessingInfo(
            kwargs=mm_kwargs,
            hashes=mm_hashes,
            prompt_updates=mm_prompt_updates,
        )

    def _build_image_features(
        self,
        prompt_token_ids: list[int],
        images: Sequence[Any],
    ) -> BatchFeature:
        # Prompt token ids already contain image_token_id at each placeholder
        # position (the renderer tokenized IMAGE_PLACEHOLDER). Walking them
        # gives the exact expanded position of every image, which
        # build_image_block needs for the C4 compress_pad alignment.
        tokenizer = self.info.get_tokenizer()
        image_token_id = _image_token_id(tokenizer)
        prompt_ids = list(prompt_token_ids)

        args = vision_args_from_config(self.info.get_hf_config())
        max_tokens = args.vision_max_n_token
        pixel_rows: list[torch.Tensor] = []
        n_vit_h_rows: list[int] = []
        n_vit_w_rows: list[int] = []
        types_rows: list[torch.Tensor] = []
        perm_rows: list[torch.Tensor] = []
        num_token_rows: list[int] = []
        max_patches = 1

        expanded_len = 0
        image_iter = iter(images)
        for tok in prompt_ids:
            if tok != image_token_id:
                expanded_len += 1
                continue
            pil = as_pil(next(image_iter))
            patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = pil_to_patches(pil, args)
            types, perm = build_image_block(n_llm_h, n_llm_w, expanded_len)
            if types.numel() > max_tokens:
                raise ValueError(
                    f"Image token block length {types.numel()} exceeds "
                    f"vision_max_n_token={max_tokens}"
                )
            pixel_rows.append(patches)
            n_vit_h_rows.append(int(n_vit_h))
            n_vit_w_rows.append(int(n_vit_w))
            types_rows.append(types)
            perm_rows.append(perm)
            num_token_rows.append(int(types.numel()))
            max_patches = max(max_patches, int(patches.shape[0]))
            expanded_len += int(types.numel())

        leftover = list(image_iter)
        if leftover:
            raise ValueError(
                f"Found {len(images)} images but only "
                f"{len(images) - len(leftover)} {IMAGE_PLACEHOLDER!r} placeholders"
            )
        if len(pixel_rows) != prompt_ids.count(image_token_id):
            raise ValueError(
                f"Found {prompt_ids.count(image_token_id)} image placeholders "
                f"but processed {len(pixel_rows)} images"
            )

        padded_pixels = []
        for patches in pixel_rows:
            if patches.shape[0] < max_patches:
                pad = patches.new_zeros(
                    max_patches - patches.shape[0], *patches.shape[1:]
                )
                patches = torch.cat([patches, pad], dim=0)
            padded_pixels.append(patches.to(torch.bfloat16))

        def _pad_1d(rows: list[torch.Tensor], fill: int, width: int) -> torch.Tensor:
            out = []
            for row in rows:
                if row.numel() < width:
                    row = torch.cat(
                        [
                            row,
                            torch.full((width - row.numel(),), fill, dtype=row.dtype),
                        ]
                    )
                out.append(row[:width])
            return torch.stack(out, dim=0)

        return BatchFeature(
            data={
                "pixel_values": torch.stack(padded_pixels, dim=0),
                "n_vit_h": torch.tensor(n_vit_h_rows, dtype=torch.int32),
                "n_vit_w": torch.tensor(n_vit_w_rows, dtype=torch.int32),
                "types": _pad_1d(types_rows, IMAGE_PAD, max_tokens),
                "perm": _pad_1d(perm_rows, -1, max_tokens),
                "num_tokens": torch.tensor(num_token_rows, dtype=torch.int32),
            }
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return dict(
            pixel_values=MultiModalFieldConfig.batched("image"),
            n_vit_h=MultiModalFieldConfig.batched("image"),
            n_vit_w=MultiModalFieldConfig.batched("image"),
            types=MultiModalFieldConfig.batched("image"),
            perm=MultiModalFieldConfig.batched("image"),
            num_tokens=MultiModalFieldConfig.batched("image"),
        )

    def _get_prompt_updates(
        self,
        mm_items,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs,
    ) -> Sequence[PromptUpdate]:
        image_token_id = _image_token_id(self.info.get_tokenizer())

        def get_replacement(item_idx: int):
            item = out_mm_kwargs["image"][item_idx]
            raw = item["num_tokens"]
            num_tokens = raw.data if hasattr(raw, "data") else raw
            if hasattr(num_tokens, "reshape"):
                num_tokens = int(num_tokens.reshape(-1)[0].item())
            else:
                num_tokens = int(num_tokens)
            if num_tokens <= 0:
                raise ValueError(f"Image {item_idx} produced 0 vision tokens")
            return [image_token_id] * num_tokens

        return [
            PromptReplacement(
                modality="image",
                target=[image_token_id],
                replacement=get_replacement,
            )
        ]


def register_vision_exp_processor(model_cls):
    return MULTIMODAL_REGISTRY.register_processor(
        DeepseekV4VisionExpMultiModalProcessor,
        info=DeepseekV4VisionExpProcessingInfo,
        dummy_inputs=DeepseekV4VisionExpDummyInputsBuilder,
    )(model_cls)
