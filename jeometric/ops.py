from typing import Optional

import jax
import jax.numpy as jnp


def segment_sum(
    data: jnp.ndarray,
    segment_ids: jnp.ndarray,
    num_segments: Optional[int] = None,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
):
    return jax.ops.segment_sum(
        data=data,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
