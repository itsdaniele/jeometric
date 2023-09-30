import jax.numpy as jnp
import jax.tree_util as tree
import jax

from typing import Any, Callable, Iterable, Mapping, Optional, Union, Dict

from ops import segment_sum
from data import Data

from flax import linen as nn


NodeFeatures = EdgeFeatures = Globals = jnp.ndarray
AggregateEdgesToNodesFn = Callable[[EdgeFeatures, jnp.ndarray, int], NodeFeatures]

aggregate_fn_dict: Dict[str, AggregateEdgesToNodesFn] = {"sum": segment_sum}


def GCNConv(
    update_node_fn: Callable[[NodeFeatures], NodeFeatures],
    aggregate_nodes_fn: str = "sum",
    add_self_edges: bool = False,
    symmetric_normalization: bool = True,
):

    assert aggregate_nodes_fn in aggregate_fn_dict
    aggregate_nodes_fn = aggregate_fn_dict[aggregate_nodes_fn]

    def _ApplyGCN(graph: Data):

        x = update_node_fn(graph.x)
        total_num_nodes = tree.tree_leaves(x)[0].shape[0]

        if add_self_edges:
            # We add self edges to the senders and receivers so that each node
            # includes itself in aggregation.
            # In principle, a `GraphsTuple` should partition by n_edge, but in
            # this case it is not required since a GCN is agnostic to whether
            # the `GraphsTuple` is a batch of graphs or a single large graph.
            conv_receivers = jnp.concatenate(
                (graph.receivers, jnp.arange(total_num_nodes)), axis=0
            )
            conv_senders = jnp.concatenate(
                (graph.senders, jnp.arange(total_num_nodes)), axis=0
            )
        else:
            conv_senders = graph.senders
            conv_receivers = graph.receivers

        if symmetric_normalization:
            # Calculate the normalization values.
            count_edges = lambda x: segment_sum(
                jnp.ones_like(conv_senders), x, total_num_nodes
            )
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
            # pylint: enable=g-long-lambda
       #graph._x = nodes
        return nodes

    return _ApplyGCN


class GCNLayer(nn.Module):
    input_dim: int
    output_dim: int
    aggregate_nodes_fn: str = "sum"
    add_self_edges: bool = False
    symmetric_normalization: bool = True

    @nn.compact
    def __call__(self, graph: Data) -> Data:
        # Define the weight matrix as a parameter of the layer.
        weight = self.param('weight', nn.initializers.xavier_uniform(), (self.input_dim, self.output_dim))
        bias = self.param('bias', nn.initializers.zeros, (self.output_dim,))

        # Define an update function that applies the weight matrix to node features.
        def update_node_fn(node_features):
            return jnp.dot(node_features, weight) + bias

        # Create a GCN convolution with the update function.
        gcn_conv = GCNConv(
            update_node_fn=update_node_fn,
            aggregate_nodes_fn=self.aggregate_nodes_fn,
            add_self_edges=self.add_self_edges,
            symmetric_normalization=self.symmetric_normalization
        )
        
        return gcn_conv(graph)