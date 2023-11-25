import jax
import jax.numpy as jnp
import jax.tree_util as tree

from flax import linen as nn

from typing import Callable, Dict

from jeometric.ops import segment_sum


NodeFeatures = EdgeFeatures = Globals = jnp.ndarray
AggregateNodesToGlobalsFn = Callable[[NodeFeatures, jnp.ndarray, int], NodeFeatures]

aggregate_fn_dict: Dict[str, AggregateNodesToGlobalsFn] = {"sum": segment_sum}


def gcn_conv(
    update_node_fn: Callable[[NodeFeatures], NodeFeatures],
    aggregation_type: str = "sum",
    add_self_edges: bool = False,
    symmetric_normalization: bool = True,
):
    """
    Adapted from https://github.com/google-deepmind/jraph/tree/master
    """

    assert aggregation_type in aggregate_fn_dict
    aggregate_nodes_fn = aggregate_fn_dict[aggregation_type]

    def _apply(x: jnp.array, senders: jnp.array, receivers: jnp.array):
        x = update_node_fn(x)
        total_num_nodes = tree.tree_leaves(x)[0].shape[0]

        if add_self_edges:
            conv_receivers = jnp.concatenate(
                (receivers, jnp.arange(total_num_nodes)), axis=0
            )
            conv_senders = jnp.concatenate(
                (senders, jnp.arange(total_num_nodes)), axis=0
            )
        else:
            conv_senders = senders
            conv_receivers = receivers

        if symmetric_normalization:
            # Calculate the normalization values.
            def count_edges(x):
                return segment_sum(jnp.ones_like(conv_senders), x, total_num_nodes)

            sender_degree = count_edges(conv_senders)
            receiver_degree = count_edges(conv_receivers)

            # Pre normalize by sqrt sender degree.
            # Avoid dividing by 0 by taking maximum of (degree, 1).
            nodes = tree.tree_map(
                lambda x: x * jax.lax.rsqrt(jnp.maximum(sender_degree, 1.0))[:, None],
                x,
            )
            # Aggregate the pre normalized nodes.
            nodes = tree.tree_map(
                lambda x: aggregate_nodes_fn(
                    x[conv_senders], conv_receivers, total_num_nodes
                ),
                nodes,
            )
            # Post normalize by sqrt receiver degree.
            # Avoid dividing by 0 by taking maximum of (degree, 1).
            nodes = tree.tree_map(
                lambda x: (
                    x * jax.lax.rsqrt(jnp.maximum(receiver_degree, 1.0))[:, None]
                ),
                nodes,
            )
        else:
            nodes = tree.tree_map(
                lambda x: aggregate_nodes_fn(
                    x[conv_senders], conv_receivers, total_num_nodes
                ),
                x,
            )
        return nodes

    return _apply


class GCNLayer(nn.Module):
    """
    A single GCN layer.
    """

    input_dim: int
    output_dim: int
    add_self_edges: bool = False
    symmetric_normalization: bool = True

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, senders: jnp.ndarray, receivers: jnp.ndarray
    ) -> jnp.ndarray:
        weight = self.param(
            "weight",
            nn.initializers.xavier_uniform(),
            (self.input_dim, self.output_dim),
        )
        bias = self.param("bias", nn.initializers.zeros, (self.output_dim,))

        def update_node_fn(node_features):
            return jnp.dot(node_features, weight) + bias

        _gcn_conv = gcn_conv(
            update_node_fn=update_node_fn,
            aggregation_type="sum",
            add_self_edges=self.add_self_edges,
            symmetric_normalization=self.symmetric_normalization,
        )

        return _gcn_conv(x, senders, receivers)

    def __str__(self):
        return f"GCNLayer({self.input_dim}, {self.output_dim})"
