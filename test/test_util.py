from util import _get_random_graph
from jeometric.util import pad_with_graph
from jeometric.data import Batch


"""
def pad_with_graph(graphs: Batch, n_node: int, n_edge: int) -> Batch:

    assert type(graphs) == Batch, "graphs must be a Batch object"
    n_graph = graphs.num_graphs + 1
    graph = jax.device_get(graphs)
    graph.glob = graphs.glob
    # TODO device_get is not copying graphs.glob, which is a dict. Fix this.

    pad_n_node = int(n_node - graph.num_nodes)
    pad_n_edge = int(n_edge - graph.num_edges)
    pad_n_graph = int(n_graph - graph.num_graphs)
    if pad_n_node <= 0 or pad_n_edge < 0 or pad_n_graph <= 0:
        raise RuntimeError(
            "Given graph is too large for the given padding. difference: "
            f"n_node {pad_n_node}, n_edge {pad_n_edge}, n_graph {pad_n_graph}"
        )

    def tree_nodes_pad(leaf):
        return np.zeros((pad_n_node,) + leaf.shape[1:], dtype=leaf.dtype)

    def tree_edges_pad(leaf):
        return np.zeros((pad_n_edge,) + leaf.shape[1:], dtype=leaf.dtype)

    # for globs, leaf is a dict. we want to create zero-value dicts with the same keys
    def tree_globs_pad(leaf):
        return np.zeros(1)

    graph = Data(
        x=tree.tree_map(tree_nodes_pad, graph.x),
        edge_attr=tree.tree_map(tree_edges_pad, graph.edge_attr),
        senders=jnp.zeros(pad_n_edge, dtype=jnp.int32),
        receivers=jnp.zeros(pad_n_edge, dtype=jnp.int32),
        glob=tree.tree_map(tree_globs_pad, graph.glob),
    )

    batch = graphs._add_to_batch(graph)
    return batch


"""


def test_pad_with_graph():
    graph1 = _get_random_graph()
    graph2 = _get_random_graph()
    batch = Batch.from_data_list([graph1, graph2])

    padded_batch = pad_with_graph(batch, 300, 400)
    assert padded_batch.num_nodes == 300
    assert padded_batch.num_edges == 400
    assert padded_batch.num_graphs == 3
    assert padded_batch.senders.shape == (400,)
    assert padded_batch.receivers.shape == (400,)
    assert padded_batch.batch.shape == (300,)
    assert padded_batch.batch[0] == 0
