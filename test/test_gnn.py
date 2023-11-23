import jax
import jax.numpy as jnp

from jeometric.gnn import GCNLayer

from util import _get_random_graph


def get_gcnlayer():
    return GCNLayer(input_dim=4, output_dim=32, aggregate_nodes_fn="sum")


def test_gcnlayer():
    graph = _get_random_graph()

    layer = get_gcnlayer()
    params = layer.init(
        jax.random.PRNGKey(0), graph.x, graph.senders, graph.receivers, graph.num_nodes
    )

    out_graph = layer.apply(
        params, graph.x, graph.senders, graph.receivers, graph.num_nodes
    )
    assert out_graph.shape == (graph.num_nodes, 32)
    assert out_graph.dtype == jnp.float32
    assert out_graph is not None
