import torch

def map_slot(
    source_slots: int,
    target_slots: int,
) -> list[int]:
    map_idx = []
    if(source_slots == target_slots): # tie nothing to do
        map_idx = list(range(0, target_slots))
    if(source_slots < target_slots): # expand case
        # ----- Short cut ------
        # map_idx = [
        #   (source_index * target_slots) // source_slots
        #   for source_index in range(source_slots)
        # ]
        for i in range(source_slots):
            map_idx.append((i * target_slots) // source_slots)
    
    raise mapidx

# mapping block as solt, so we dont seprate head
def block_slot_indices(
    source_blocks: int,
    target_blocks: int,
    block_size: int,
) -> tuple[list[int], list[int]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    target_block_indices = slot_map(source_blocks, target_blocks)
    source_indices = []
    target_indices = []
    # enumerate return index, vlaue as tuple
    for source_block_index, target_block_index in
    enumerate(target_block_indices):
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
)-> torch.Tensor: # for now hidden dim must be same size.
    if source.ndim != 1: # ndim -> num of dim
        raise ValueError("source must be a 1D tensor")
    if target_size < source.shape[0]:
        raise ValueError("target_size must be >= source length")
    if len(source_indices) != len(target_indices):
        raise ValueError("source_indices and target_indices must have the same length")
    # create a inint value tensor vector 
    target = torch.full(
        (target_size,),
        fill_value=fill_value,
        dtype=source.dtype,
        device=source.device,
    )
    # convert plain list into tensor type
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
    if source.ndim != 2:
        raise ValueError("source must be a 2D tensor")
    
    if(target_shape) != 2: 
        raise ValueError("target_shape must have length 2")
    
    if len(row_source_indices) != len(row_target_indices):
        raise ValueError("row source/target indices must have the same length")

    if len(col_source_indices) != len(col_target_indices):
        raise ValueError("row source/target indices must have the same length")

    target = torch.full(
        target_shape,
        fill_value=fill_value,
        dtype=source.dtype,
        device=source.device,
    )

    row_source_idx_tensor = torch.tensor(
        row_source_indices,
        dtype=torch.long,
        device=source.device,
    )
    row_target_idx_tensor = torch.tensor(
        row_target_indices,
        dtype=torch.long,
        device=source.device,
    )
    col_source_idx_tensor = torch.tensor(
        col_source_indices,
        dtype=torch.long,
        device=source.device,
    )
    col_target_idx_tensor = torch.tensor(
        col_target_indices,
        dtype=torch.long,
        device=source.device,
    )

    target[
        row_target_idx_tensor[:, None],
        col_target_idx_tensor[None, :],
    ] = source[
        row_source_idx_tensor[:, None],
        col_source_idx_tensor[None, :],
    ]
    # equvient to this
    # for i in range(len(row_source_indices)):
    #   for j in range(len(col_source_indices)):
    #       source_row = row_source_indices[i]
    #       source_col = col_source_indices[j]
    #       target_row = row_target_indices[i]
    #       target_col = col_target_indices[j]

    #       target[target_row, target_col] = source[source_row, source_col]

    return target