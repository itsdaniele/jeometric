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
    """
    Alias to `jax.ops.segment_sum`.

    From https://github.com/google-deepmind/jraph/blob/master/jraph/_src/utils.py#L424
    """
    return jax.ops.segment_sum(
        data=data,
        segment_ids=segment_ids,
        num_segments=num_segments,
        indices_are_sorted=indices_are_sorted,
        unique_indices=unique_indices,
    )
