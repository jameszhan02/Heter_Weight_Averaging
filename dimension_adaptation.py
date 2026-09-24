from __future__ import annotations

import torch


def map_slot(source_slots: int, target_slots: int) -> list[int]:
    """Map source slot ids into target slot ids for expand-style placement."""
    if source_slots <= 0:
        raise ValueError("source_slots must be positive")
    if target_slots <= 0:
        raise ValueError("target_slots must be positive")
    if source_slots > target_slots:
        raise ValueError("source_slots must be <= target_slots for expansion")

    return [
        (source_index * target_slots) // source_slots
        for source_index in range(source_slots)
    ]


def block_slot_indices(
    source_blocks: int,
    target_blocks: int,
    block_size: int,
) -> tuple[list[int], list[int]]:
    """Expand block-level slot mapping into flat source/target indices."""
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    target_block_indices = map_slot(source_blocks, target_blocks)
    source_indices: list[int] = []
    target_indices: list[int] = []

    for source_block_index, target_block_index in enumerate(target_block_indices):
        source_start = source_block_index * block_size
        target_start = target_block_index * block_size

        for offset in range(block_size):
            source_indices.append(source_start + offset)
            target_indices.append(target_start + offset)

    return source_indices, target_indices


def expand_vector_by_indices(
    source: torch.Tensor,
    target_size: int,
    source_indices: list[int],
    target_indices: list[int],
    fill_value: float = 0,
) -> torch.Tensor:
    """Create a larger 1D tensor and copy source values by explicit mapping."""
    if source.ndim != 1:
        raise ValueError("source must be a 1D tensor")
    if target_size < source.shape[0]:
        raise ValueError("target_size must be >= source length")
    if len(source_indices) != len(target_indices):
        raise ValueError("source_indices and target_indices must have the same length")

    target = torch.full(
        (target_size,),
        fill_value=fill_value,
        dtype=source.dtype,
        device=source.device,
    )

    source_index_tensor = torch.tensor(
        source_indices,
        dtype=torch.long,
        device=source.device,
    )
    target_index_tensor = torch.tensor(
        target_indices,
        dtype=torch.long,
        device=source.device,
    )

    target[target_index_tensor] = source[source_index_tensor]
    return target


def expand_matrix_by_indices(
    source: torch.Tensor,
    target_shape: tuple[int, int],
    row_source_indices: list[int],
    row_target_indices: list[int],
    col_source_indices: list[int],
    col_target_indices: list[int],
    fill_value: float = 0,
) -> torch.Tensor:
    """Create a larger 2D tensor and copy source values by row/column mappings."""
    if source.ndim != 2:
        raise ValueError("source must be a 2D tensor")
    if len(target_shape) != 2:
        raise ValueError("target_shape must have length 2")
    if len(row_source_indices) != len(row_target_indices):
        raise ValueError("row source/target indices must have the same length")
    if len(col_source_indices) != len(col_target_indices):
        raise ValueError("column source/target indices must have the same length")

    target = torch.full(
        target_shape,
        fill_value=fill_value,
        dtype=source.dtype,
        device=source.device,
    )

    row_source = torch.tensor(row_source_indices, dtype=torch.long, device=source.device)
    row_target = torch.tensor(row_target_indices, dtype=torch.long, device=source.device)
    col_source = torch.tensor(col_source_indices, dtype=torch.long, device=source.device)
    col_target = torch.tensor(col_target_indices, dtype=torch.long, device=source.device)

    target[
        row_target[:, None],
        col_target[None, :],
    ] = source[
        row_source[:, None],
        col_source[None, :],
    ]

    return target


def _check_same_head_dim(source_head_dim: int, target_head_dim: int) -> int:
    if source_head_dim != target_head_dim:
        raise ValueError("head_dim mismatch is not supported in expand v0")
    return source_head_dim


def _check_shape(actual: torch.Size, expected: tuple[int, ...], name: str) -> None:
    if tuple(actual) != expected:
        raise ValueError(f"{name} shape mismatch: expected {expected}, got {tuple(actual)}")


def _hidden_block_indices(
    source_hidden_size: int,
    target_hidden_size: int,
    head_dim: int,
) -> tuple[list[int], list[int]]:
    if source_hidden_size % head_dim != 0:
        raise ValueError("source_hidden_size must be divisible by head_dim")
    if target_hidden_size % head_dim != 0:
        raise ValueError("target_hidden_size must be divisible by head_dim")

    return block_slot_indices(
        source_blocks=source_hidden_size // head_dim,
        target_blocks=target_hidden_size // head_dim,
        block_size=head_dim,
    )


def expand_q_proj_weight(
    source: torch.Tensor,
    target_shape: tuple[int, int],
    source_num_q_heads: int,
    target_num_q_heads: int,
    source_hidden_size: int,
    target_hidden_size: int,
    source_head_dim: int,
    target_head_dim: int,
) -> torch.Tensor:
    """Expand q_proj.weight: rows are query-head blocks, columns are hidden blocks."""
    head_dim = _check_same_head_dim(source_head_dim, target_head_dim)
    _check_shape(
        source.shape,
        (source_num_q_heads * head_dim, source_hidden_size),
        "source q_proj.weight",
    )
    if target_shape != (target_num_q_heads * head_dim, target_hidden_size):
        raise ValueError("target_shape does not match target q_proj metadata")

    row_source, row_target = block_slot_indices(
        source_blocks=source_num_q_heads,
        target_blocks=target_num_q_heads,
        block_size=head_dim,
    )
    col_source, col_target = _hidden_block_indices(
        source_hidden_size=source_hidden_size,
        target_hidden_size=target_hidden_size,
        head_dim=head_dim,
    )

    return expand_matrix_by_indices(
        source=source,
        target_shape=target_shape,
        row_source_indices=row_source,
        row_target_indices=row_target,
        col_source_indices=col_source,
        col_target_indices=col_target,
        fill_value=0,
    )


def expand_kv_proj_weight(
    source: torch.Tensor,
    target_shape: tuple[int, int],
    source_num_kv_heads: int,
    target_num_kv_heads: int,
    source_hidden_size: int,
    target_hidden_size: int,
    source_head_dim: int,
    target_head_dim: int,
) -> torch.Tensor:
    """Expand k_proj.weight or v_proj.weight: rows are KV-head blocks."""
    head_dim = _check_same_head_dim(source_head_dim, target_head_dim)
    _check_shape(
        source.shape,
        (source_num_kv_heads * head_dim, source_hidden_size),
        "source kv_proj.weight",
    )
    if target_shape != (target_num_kv_heads * head_dim, target_hidden_size):
        raise ValueError("target_shape does not match target kv_proj metadata")

    row_source, row_target = block_slot_indices(
        source_blocks=source_num_kv_heads,
        target_blocks=target_num_kv_heads,
        block_size=head_dim,
    )
    col_source, col_target = _hidden_block_indices(
        source_hidden_size=source_hidden_size,
        target_hidden_size=target_hidden_size,
        head_dim=head_dim,
    )

    return expand_matrix_by_indices(
        source=source,
        target_shape=target_shape,
        row_source_indices=row_source,
        row_target_indices=row_target,
        col_source_indices=col_source,
        col_target_indices=col_target,
        fill_value=0,
    )


def expand_o_proj_weight(
    source: torch.Tensor,
    target_shape: tuple[int, int],
    source_num_q_heads: int,
    target_num_q_heads: int,
    source_hidden_size: int,
    target_hidden_size: int,
    source_head_dim: int,
    target_head_dim: int,
) -> torch.Tensor:
    """Expand o_proj.weight: rows are hidden blocks, columns are query-head blocks."""
    head_dim = _check_same_head_dim(source_head_dim, target_head_dim)
    _check_shape(
        source.shape,
        (source_hidden_size, source_num_q_heads * head_dim),
        "source o_proj.weight",
    )
    if target_shape != (target_hidden_size, target_num_q_heads * head_dim):
        raise ValueError("target_shape does not match target o_proj metadata")

    row_source, row_target = _hidden_block_indices(
        source_hidden_size=source_hidden_size,
        target_hidden_size=target_hidden_size,
        head_dim=head_dim,
    )
    col_source, col_target = block_slot_indices(
        source_blocks=source_num_q_heads,
        target_blocks=target_num_q_heads,
        block_size=head_dim,
    )

    return expand_matrix_by_indices(
        source=source,
        target_shape=target_shape,
        row_source_indices=row_source,
        row_target_indices=row_target,
        col_source_indices=col_source,
        col_target_indices=col_target,
        fill_value=0,
    )


def expand_up_or_gate_proj_weight(
    source: torch.Tensor,
    target_shape: tuple[int, int],
    source_intermediate_size: int,
    target_intermediate_size: int,
    source_hidden_size: int,
    target_hidden_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Expand gate_proj.weight or up_proj.weight."""
    _check_shape(
        source.shape,
        (source_intermediate_size, source_hidden_size),
        "source up/gate projection weight",
    )
    if target_shape != (target_intermediate_size, target_hidden_size):
        raise ValueError("target_shape does not match target up/gate metadata")

    row_source, row_target = block_slot_indices(
        source_blocks=source_intermediate_size,
        target_blocks=target_intermediate_size,
        block_size=1,
    )
    col_source, col_target = _hidden_block_indices(
        source_hidden_size=source_hidden_size,
        target_hidden_size=target_hidden_size,
        head_dim=head_dim,
    )

    return expand_matrix_by_indices(
        source=source,
        target_shape=target_shape,
        row_source_indices=row_source,
        row_target_indices=row_target,
        col_source_indices=col_source,
        col_target_indices=col_target,
        fill_value=0,
    )


def expand_down_proj_weight(
    source: torch.Tensor,
    target_shape: tuple[int, int],
    source_hidden_size: int,
    target_hidden_size: int,
    source_intermediate_size: int,
    target_intermediate_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Expand down_proj.weight."""
    _check_shape(
        source.shape,
        (source_hidden_size, source_intermediate_size),
        "source down projection weight",
    )
    if target_shape != (target_hidden_size, target_intermediate_size):
        raise ValueError("target_shape does not match target down metadata")

    row_source, row_target = _hidden_block_indices(
        source_hidden_size=source_hidden_size,
        target_hidden_size=target_hidden_size,
        head_dim=head_dim,
    )
    col_source, col_target = block_slot_indices(
        source_blocks=source_intermediate_size,
        target_blocks=target_intermediate_size,
        block_size=1,
    )

    return expand_matrix_by_indices(
        source=source,
        target_shape=target_shape,
        row_source_indices=row_source,
        row_target_indices=row_target,
        col_source_indices=col_source,
        col_target_indices=col_target,
        fill_value=0,
    )


def expand_embedding_or_lm_head_weight(
    source: torch.Tensor,
    target_shape: tuple[int, int],
    source_vocab_size: int,
    target_vocab_size: int,
    source_hidden_size: int,
    target_hidden_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Expand embedding/lm_head when vocabulary rows match exactly."""
    if source_vocab_size != target_vocab_size:
        raise ValueError("vocab_size mismatch is not supported in expand v0")

    _check_shape(
        source.shape,
        (source_vocab_size, source_hidden_size),
        "source embedding/lm_head weight",
    )
    if target_shape != (target_vocab_size, target_hidden_size):
        raise ValueError("target_shape does not match target embedding/lm_head metadata")

    row_indices = list(range(source_vocab_size))
    col_source, col_target = _hidden_block_indices(
        source_hidden_size=source_hidden_size,
        target_hidden_size=target_hidden_size,
        head_dim=head_dim,
    )

    return expand_matrix_by_indices(
        source=source,
        target_shape=target_shape,
        row_source_indices=row_indices,
        row_target_indices=row_indices,
        col_source_indices=col_source,
        col_target_indices=col_target,
        fill_value=0,
    )


def expand_norm_weight(
    source: torch.Tensor,
    target_size: int,
    source_hidden_size: int,
    target_hidden_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Expand RMSNorm/LayerNorm scale with fill value 1 for new coordinates."""
    _check_shape(source.shape, (source_hidden_size,), "source norm weight")
    if target_size != target_hidden_size:
        raise ValueError("target_size must match target_hidden_size")

    source_indices, target_indices = _hidden_block_indices(
        source_hidden_size=source_hidden_size,
        target_hidden_size=target_hidden_size,
        head_dim=head_dim,
    )

    return expand_vector_by_indices(
        source=source,
        target_size=target_size,
        source_indices=source_indices,
        target_indices=target_indices,
        fill_value=1,
    )


def truncate_tensor(
    source: torch.Tensor,
    target_shape: tuple[int, ...],
) -> torch.Tensor:
    """Project a larger tensor into a smaller prefix-shaped tensor."""
    if source.ndim != len(target_shape):
        raise ValueError(
            f"rank mismatch: source rank {source.ndim}, target rank {len(target_shape)}"
        )

    for dim_index, (source_size, target_size) in enumerate(
        zip(source.shape, target_shape)
    ):
        if target_size <= 0:
            raise ValueError("target dimensions must be positive")
        if source_size < target_size:
            raise ValueError(
                "source dimension is smaller than target dimension: "
                f"dim {dim_index}, source {source_size}, target {target_size}"
            )

    slices = tuple(slice(0, target_size) for target_size in target_shape)
    return source[slices].clone()
