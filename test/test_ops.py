import numpy as np
import jax.numpy as jnp
from jeometric import ops


def test_segment_sum():
    """
    Adapted from https://github.com/google-deepmind/jraph/blob/master/jraph/_src/utils_test.py
    """
    result = ops.segment_sum(jnp.arange(9), jnp.array([0, 1, 2, 0, 4, 0, 1, 1, 0]), 6)
    np.testing.assert_allclose(result, jnp.array([16, 14, 2, 0, 4, 0]))


def test_segment_sum_optional_num_segments():
    """
    Adapted from https://github.com/google-deepmind/jraph/blob/master/jraph/_src/utils_test.py
    """
    result = ops.segment_sum(jnp.arange(9), jnp.array([0, 1, 2, 0, 4, 0, 1, 1, 0]))
    np.testing.assert_allclose(result, jnp.array([16, 14, 2, 0, 4]))
