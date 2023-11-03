import jax
import jax.numpy as jnp

from data import Data
from gnn import GCNLayer
import numpy as np


def _get_random_graph(max_n_graph=10):
    n_graph = np.random.randint(1, max_n_graph + 1)
    n_node = np.random.randint(0, 10, n_graph)
    n_edge = np.random.randint(0, 20, n_graph)
    # We cannot have any edges if there are no nodes.
    n_edge[n_node == 0] = 0

    senders = []
    receivers = []
    offset = 0
    for n_node_in_graph, n_edge_in_graph in zip(n_node, n_edge):
        if n_edge_in_graph != 0:
            senders += list(
                np.random.randint(0, n_node_in_graph, n_edge_in_graph) + offset
            )
            receivers += list(
                np.random.randint(0, n_node_in_graph, n_edge_in_graph) + offset
            )
        offset += n_node_in_graph

    return Data(
        x=jnp.asarray(np.random.random(size=(np.sum(n_node), 4))),
        edge_attr=jnp.asarray(np.random.random(size=(np.sum(n_edge), 3))),
        senders=jnp.asarray(senders),
        receivers=jnp.asarray(receivers),
    )


# def get_gcnconv():
#     return GCNConv(
#         update_node_fn=lambda x: x,
#         aggregate_nodes_fn="sum",
#         add_self_edges=False,
#         symmetric_normalization=True,
#     )


def get_gcnlayer():
    return GCNLayer(input_dim=4, output_dim=32, aggregate_nodes_fn="sum")


if __name__ == "__main__":
    graph = _get_random_graph()
    # apply_fn = get_gcnconv()
    # print(apply_fn)
    # print(apply_fn(graph))

    layer = get_gcnlayer()
    params = layer.init(jax.random.PRNGKey(0), graph)

    out_graph = layer.apply(params, graph)
    print(out_graph.shape)
