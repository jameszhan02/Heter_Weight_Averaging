def map_slot(
    source_size: int,
    target_size: int,
    source_index: int,
) -> int:
    if(source_size == 0 or target_size == 0):
        raise ValueError
    if(source_index >= source_size or source_index < 0):
        raise IndexError
    if source_size == target_size:
        return source_index
    if source_size < target_size: # Expand (paper Appdenix B.1)
        return (source_index * target_size) // source_size
    if source_size > target_size:
        raise ValueError("target_size must not be smaller than source_size")
    
    raise NotImplementedError




def map_axis_block(
    source_block_count: int,
    target_block_count: int,
    source_block_size: int,
    target_block_size: int,
    source_block_index: int,
) -> tuple[slice, slice]:
    if (source_block_size <= 0 or target_block_size <= 0 or target_block_size != source_block_size):
         raise ValueError

    target_head_index = map_slot(
          source_size=source_block_count,
          target_size=target_block_count,
          source_index=source_block_index,
      )

    source_start = source_block_index * source_block_size
    source_end = source_start + source_block_size

    target_start = target_head_index * target_block_size
    target_end = target_start + target_block_size

    source_slice = slice(source_start, source_end)
    target_slice = slice(target_start, target_end)


    return source_slice, target_slice