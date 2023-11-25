import jax
import jax.numpy as jnp

from jeometric.gnn import GCNLayer

from util import _get_random_graph


def get_gcnlayer(input_dim, output_dim):
    return GCNLayer(input_dim=input_dim, output_dim=output_dim)


def test_gcnlayer():
    graph = _get_random_graph()

    layer = get_gcnlayer(4, 32)
    params = layer.init(jax.random.PRNGKey(0), graph.x, graph.senders, graph.receivers)

    out_graph = layer.apply(params, graph.x, graph.senders, graph.receivers)
    assert out_graph.shape == (graph.num_nodes, 32)
    assert out_graph.dtype == jnp.float32
    assert out_graph is not None
    assert str(layer) == "GCNLayer(4, 32)"
