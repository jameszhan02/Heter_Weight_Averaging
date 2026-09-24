from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from dimension_adaptation import (
    experimental_remap_kv_proj_weight,
    experimental_remap_o_proj_weight,
    experimental_remap_q_proj_weight,
    expand_down_proj_weight,
    expand_embedding_or_lm_head_weight,
    expand_kv_proj_weight,
    expand_norm_weight,
    expand_o_proj_weight,
    expand_q_proj_weight,
    expand_up_or_gate_proj_weight,
    truncate_tensor,
)


@dataclass(frozen=True)
class ModelShape:
    model_type: str
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str
    vocab_size: int
    tie_word_embeddings: bool
    attention_bias: bool
    rope_theta: object
    rope_scaling: object


def shape_from_config(config: object) -> ModelShape:
    hidden_size = int(getattr(config, "hidden_size"))
    num_attention_heads = int(getattr(config, "num_attention_heads"))
    explicit_head_dim = getattr(config, "head_dim", None)

    if explicit_head_dim is None:
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        head_dim = hidden_size // num_attention_heads
    else:
        head_dim = int(explicit_head_dim)

    num_key_value_heads = getattr(config, "num_key_value_heads", num_attention_heads)

    return ModelShape(
        model_type=str(getattr(config, "model_type")),
        hidden_size=hidden_size,
        intermediate_size=int(getattr(config, "intermediate_size")),
        num_hidden_layers=int(getattr(config, "num_hidden_layers")),
        num_attention_heads=num_attention_heads,
        num_key_value_heads=int(num_key_value_heads),
        head_dim=head_dim,
        hidden_act=str(getattr(config, "hidden_act")),
        vocab_size=int(getattr(config, "vocab_size")),
        tie_word_embeddings=bool(getattr(config, "tie_word_embeddings", False)),
        attention_bias=bool(getattr(config, "attention_bias", False)),
        rope_theta=getattr(config, "rope_theta", None),
        rope_scaling=getattr(config, "rope_scaling", None),
    )


def attention_kind(shape: ModelShape) -> str:
    if shape.num_attention_heads == shape.num_key_value_heads:
        return "mha"
    if shape.num_key_value_heads == 1:
        return "mqa"
    return "gqa"


def validate_union_compatibility(
    source_shape: ModelShape,
    target_shape: ModelShape,
) -> None:
    reasons: list[str] = []

    equal_fields = [
        "model_type",
        "hidden_act",
        "vocab_size",
        "tie_word_embeddings",
        "attention_bias",
        "rope_theta",
        "rope_scaling",
        "head_dim",
    ]
    for field_name in equal_fields:
        source_value = getattr(source_shape, field_name)
        target_value = getattr(target_shape, field_name)
        if source_value != target_value:
            reasons.append(
                f"{field_name} mismatch: source={source_value!r}, target={target_value!r}"
            )

    if attention_kind(source_shape) != attention_kind(target_shape):
        reasons.append(
            "attention kind mismatch: "
            f"source={attention_kind(source_shape)}, target={attention_kind(target_shape)}"
        )

    if source_shape.hidden_size > target_shape.hidden_size:
        reasons.append("source hidden_size is larger than target hidden_size")
    if source_shape.intermediate_size > target_shape.intermediate_size:
        reasons.append("source intermediate_size is larger than target intermediate_size")
    if source_shape.num_hidden_layers > target_shape.num_hidden_layers:
        reasons.append("source has more layers than target")
    if source_shape.num_attention_heads > target_shape.num_attention_heads:
        reasons.append("source has more query heads than target")
    if source_shape.num_key_value_heads > target_shape.num_key_value_heads:
        reasons.append("source has more KV heads than target")

    if reasons:
        joined = "\n- ".join(reasons)
        raise ValueError(f"Union merge is not supported:\n- {joined}")


def validate_intersection_compatibility(
    source_shape: ModelShape,
    target_shape: ModelShape,
) -> None:
    reasons: list[str] = []

    equal_fields = [
        "model_type",
        "hidden_act",
        "vocab_size",
        "tie_word_embeddings",
        "attention_bias",
        "rope_theta",
        "rope_scaling",
    ]
    for field_name in equal_fields:
        source_value = getattr(source_shape, field_name)
        target_value = getattr(target_shape, field_name)
        if source_value != target_value:
            reasons.append(
                f"{field_name} mismatch: source={source_value!r}, target={target_value!r}"
            )

    if source_shape.hidden_size < target_shape.hidden_size:
        reasons.append("source hidden_size is smaller than target hidden_size")
    if source_shape.intermediate_size < target_shape.intermediate_size:
        reasons.append("source intermediate_size is smaller than target intermediate_size")
    if source_shape.num_hidden_layers < target_shape.num_hidden_layers:
        reasons.append("source has fewer layers than target")

    if reasons:
        joined = "\n- ".join(reasons)
        raise ValueError(f"Intersection merge is not supported:\n- {joined}")


def weighted_average_tensors(
    left: torch.Tensor,
    right: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: left={tuple(left.shape)}, right={tuple(right.shape)}")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")
    if not left.is_floating_point() or not right.is_floating_point():
        if alpha == 0:
            return left.clone()
        if alpha == 1:
            return right.clone()
        raise TypeError("non-floating tensors only support alpha 0 or 1")

    result_dtype = right.dtype
    merged = (1 - alpha) * left.to(torch.float32) + alpha * right.to(torch.float32)
    return merged.to(dtype=result_dtype)


def merge_state_dict_union(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    source_config: object,
    target_config: object,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Expand source tensors into target space, then average with target tensors."""
    source_shape = shape_from_config(source_config)
    target_shape = shape_from_config(target_config)
    validate_union_compatibility(source_shape, target_shape)

    merged: dict[str, torch.Tensor] = {}

    for key, target_tensor in target_state.items():
        expanded_source = expand_tensor_for_key(
            key=key,
            source_tensor=source_state.get(key),
            target_tensor=target_tensor,
            source_shape=source_shape,
            target_shape=target_shape,
            allow_missing_as_new_layer=True,
        )
        merged[key] = weighted_average_tensors(
            expanded_source,
            target_tensor,
            alpha=alpha,
        )

    return merged


def merge_state_dict_intersection(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    source_config: object,
    target_config: object,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Truncate source tensors into target space, then average with target tensors."""
    source_shape = shape_from_config(source_config)
    target_shape = shape_from_config(target_config)
    validate_intersection_compatibility(source_shape, target_shape)

    merged: dict[str, torch.Tensor] = {}

    for key, target_tensor in target_state.items():
        if key not in source_state:
            raise KeyError(f"source state_dict is missing target key: {key}")

        truncated_source = truncate_tensor(source_state[key], tuple(target_tensor.shape))
        merged[key] = weighted_average_tensors(
            target_tensor,
            truncated_source,
            alpha=alpha,
        )

    return merged


def merge_state_dict_experimental_target_anchor(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    source_config: object,
    target_config: object,
    alpha: float,
    merge_attention: bool = True,
    merge_mlp: bool = True,
    merge_norm: bool = True,
) -> dict[str, torch.Tensor]:
    """Experimental cross-architecture merge anchored on the target checkpoint.

    This is not the paper's strict Appendix B method. It is intended for cases
    such as OLMo -> Llama where vocabularies, attention layouts, and model
    families differ. The output keeps the target key set and keeps target
    embedding/lm_head weights.
    """
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be in [0, 1]")

    source_shape = shape_from_config(source_config)
    target_shape = shape_from_config(target_config)
    merged: dict[str, torch.Tensor] = {}

    for key, target_tensor in target_state.items():
        if _is_vocab_weight(key):
            merged[key] = target_tensor.clone()
            continue

        source_tensor = source_state.get(key)
        if source_tensor is None:
            merged[key] = target_tensor.clone()
            continue

        if _is_attention_weight(key):
            if not merge_attention:
                merged[key] = target_tensor.clone()
                continue
            remapped_source = experimental_remap_attention_weight(
                key=key,
                source_tensor=source_tensor,
                target_tensor=target_tensor,
                source_shape=source_shape,
                target_shape=target_shape,
            )
            merged[key] = weighted_average_tensors(
                target_tensor,
                remapped_source,
                alpha=alpha,
            )
            continue

        if _is_mlp_weight(key):
            if merge_mlp and tuple(source_tensor.shape) == tuple(target_tensor.shape):
                merged[key] = weighted_average_tensors(
                    target_tensor,
                    source_tensor,
                    alpha=alpha,
                )
            else:
                merged[key] = target_tensor.clone()
            continue

        if _is_norm_weight(key):
            if merge_norm and tuple(source_tensor.shape) == tuple(target_tensor.shape):
                merged[key] = weighted_average_tensors(
                    target_tensor,
                    source_tensor,
                    alpha=alpha,
                )
            else:
                merged[key] = target_tensor.clone()
            continue

        merged[key] = target_tensor.clone()

    return merged


def experimental_remap_attention_weight(
    key: str,
    source_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    source_shape: ModelShape,
    target_shape: ModelShape,
) -> torch.Tensor:
    target_tensor_shape = _as_matrix_shape(tuple(target_tensor.shape))

    if key.endswith("q_proj.weight"):
        return experimental_remap_q_proj_weight(
            source=source_tensor,
            target_shape=target_tensor_shape,
            source_num_q_heads=source_shape.num_attention_heads,
            target_num_q_heads=target_shape.num_attention_heads,
            source_hidden_size=source_shape.hidden_size,
            target_hidden_size=target_shape.hidden_size,
            source_head_dim=source_shape.head_dim,
            target_head_dim=target_shape.head_dim,
        )

    if key.endswith("k_proj.weight") or key.endswith("v_proj.weight"):
        return experimental_remap_kv_proj_weight(
            source=source_tensor,
            target_shape=target_tensor_shape,
            source_num_kv_heads=source_shape.num_key_value_heads,
            target_num_kv_heads=target_shape.num_key_value_heads,
            source_hidden_size=source_shape.hidden_size,
            target_hidden_size=target_shape.hidden_size,
            source_head_dim=source_shape.head_dim,
            target_head_dim=target_shape.head_dim,
        )

    if key.endswith("o_proj.weight"):
        return experimental_remap_o_proj_weight(
            source=source_tensor,
            target_shape=target_tensor_shape,
            source_num_q_heads=source_shape.num_attention_heads,
            target_num_q_heads=target_shape.num_attention_heads,
            source_hidden_size=source_shape.hidden_size,
            target_hidden_size=target_shape.hidden_size,
            source_head_dim=source_shape.head_dim,
            target_head_dim=target_shape.head_dim,
        )

    raise NotImplementedError(f"no experimental attention remap rule for key: {key}")


def expand_tensor_for_key(
    key: str,
    source_tensor: torch.Tensor | None,
    target_tensor: torch.Tensor,
    source_shape: ModelShape,
    target_shape: ModelShape,
    allow_missing_as_new_layer: bool,
) -> torch.Tensor:
    if source_tensor is None:
        if allow_missing_as_new_layer:
            return initialize_new_target_tensor(key, target_tensor)
        raise KeyError(f"source state_dict is missing target key: {key}")

    if tuple(source_tensor.shape) == tuple(target_tensor.shape):
        return source_tensor.clone()

    target_tensor_shape = tuple(target_tensor.shape)

    if key.endswith("q_proj.weight"):
        return expand_q_proj_weight(
            source=source_tensor,
            target_shape=_as_matrix_shape(target_tensor_shape),
            source_num_q_heads=source_shape.num_attention_heads,
            target_num_q_heads=target_shape.num_attention_heads,
            source_hidden_size=source_shape.hidden_size,
            target_hidden_size=target_shape.hidden_size,
            source_head_dim=source_shape.head_dim,
            target_head_dim=target_shape.head_dim,
        )

    if key.endswith("k_proj.weight") or key.endswith("v_proj.weight"):
        return expand_kv_proj_weight(
            source=source_tensor,
            target_shape=_as_matrix_shape(target_tensor_shape),
            source_num_kv_heads=source_shape.num_key_value_heads,
            target_num_kv_heads=target_shape.num_key_value_heads,
            source_hidden_size=source_shape.hidden_size,
            target_hidden_size=target_shape.hidden_size,
            source_head_dim=source_shape.head_dim,
            target_head_dim=target_shape.head_dim,
        )

    if key.endswith("o_proj.weight"):
        return expand_o_proj_weight(
            source=source_tensor,
            target_shape=_as_matrix_shape(target_tensor_shape),
            source_num_q_heads=source_shape.num_attention_heads,
            target_num_q_heads=target_shape.num_attention_heads,
            source_hidden_size=source_shape.hidden_size,
            target_hidden_size=target_shape.hidden_size,
            source_head_dim=source_shape.head_dim,
            target_head_dim=target_shape.head_dim,
        )

    if key.endswith("gate_proj.weight") or key.endswith("up_proj.weight"):
        return expand_up_or_gate_proj_weight(
            source=source_tensor,
            target_shape=_as_matrix_shape(target_tensor_shape),
            source_intermediate_size=source_shape.intermediate_size,
            target_intermediate_size=target_shape.intermediate_size,
            source_hidden_size=source_shape.hidden_size,
            target_hidden_size=target_shape.hidden_size,
            head_dim=source_shape.head_dim,
        )

    if key.endswith("down_proj.weight"):
        return expand_down_proj_weight(
            source=source_tensor,
            target_shape=_as_matrix_shape(target_tensor_shape),
            source_hidden_size=source_shape.hidden_size,
            target_hidden_size=target_shape.hidden_size,
            source_intermediate_size=source_shape.intermediate_size,
            target_intermediate_size=target_shape.intermediate_size,
            head_dim=source_shape.head_dim,
        )

    if key.endswith("embed_tokens.weight") or key.endswith("lm_head.weight"):
        return expand_embedding_or_lm_head_weight(
            source=source_tensor,
            target_shape=_as_matrix_shape(target_tensor_shape),
            source_vocab_size=source_shape.vocab_size,
            target_vocab_size=target_shape.vocab_size,
            source_hidden_size=source_shape.hidden_size,
            target_hidden_size=target_shape.hidden_size,
            head_dim=source_shape.head_dim,
        )

    if key.endswith("norm.weight") or key.endswith("layernorm.weight"):
        return expand_norm_weight(
            source=source_tensor,
            target_size=target_tensor.numel(),
            source_hidden_size=source_shape.hidden_size,
            target_hidden_size=target_shape.hidden_size,
            head_dim=source_shape.head_dim,
        )

    raise NotImplementedError(f"no expand rule for key: {key}")


def initialize_new_target_tensor(
    key: str,
    target_tensor: torch.Tensor,
) -> torch.Tensor:
    if key.endswith("norm.weight") or key.endswith("layernorm.weight"):
        return torch.ones_like(target_tensor)
    if target_tensor.is_floating_point():
        return torch.zeros_like(target_tensor)
    return target_tensor.clone()


def _as_matrix_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"expected 2D tensor shape, got {shape}")
    return shape


def _is_vocab_weight(key: str) -> bool:
    return key.endswith("embed_tokens.weight") or key.endswith("lm_head.weight")


def _is_attention_weight(key: str) -> bool:
    return (
        key.endswith("q_proj.weight")
        or key.endswith("k_proj.weight")
        or key.endswith("v_proj.weight")
        or key.endswith("o_proj.weight")
    )


def _is_mlp_weight(key: str) -> bool:
    return (
        key.endswith("gate_proj.weight")
        or key.endswith("up_proj.weight")
        or key.endswith("down_proj.weight")
    )


def _is_norm_weight(key: str) -> bool:
    return key.endswith("norm.weight") or key.endswith("layernorm.weight")
