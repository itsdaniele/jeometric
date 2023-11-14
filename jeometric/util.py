import jax.numpy as jnp
import jax.tree_util as tree


from jeometric.data import Data, Batch

from typing import Sequence

import jax
import numpy as np


def batch(graphs: Sequence[Data]) -> Data:
    """Returns a batched graph given a list of graphs.

    This method will concatenate the ``nodes``, ``edges`` and ``globals``,
    ``n_node`` and ``n_edge`` of a sequence of ``GraphsTuple`` along axis 0. For
    ``senders`` and ``receivers``, offsets are computed so that connectivity
    remains valid for the new node indices.

    For example::

      key = jax.random.PRNGKey(0)
      graph_1 = GraphsTuple(nodes=jax.random.normal(key, (3, 64)),
                        edges=jax.random.normal(key, (5, 64)),
                        senders=jnp.array([0,0,1,1,2]),
                        receivers=[1,2,0,2,1],
                        n_node=jnp.array([3]),
                        n_edge=jnp.array([5]),
                        globals=jax.random.normal(key, (1, 64)))
      graph_2 = GraphsTuple(nodes=jax.random.normal(key, (5, 64)),
                        edges=jax.random.normal(key, (10, 64)),
                        senders=jnp.array([0,0,1,1,2,2,3,3,4,4]),
                        receivers=jnp.array([1,2,0,2,1,0,2,1,3,2]),
                        n_node=jnp.array([5]),
                        n_edge=jnp.array([10]),
                        globals=jax.random.normal(key, (1, 64)))
      batch = graph.batch([graph_1, graph_2])

      batch.nodes.shape
      >> (8, 64)
      batch.edges.shape
      >> (15, 64)
      # Offsets computed on senders and receivers
      batch.senders
      >> DeviceArray([0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7], dtype=int32)
      batch.receivers
      >> DeviceArray([1, 2, 0, 2, 1, 4, 5, 3, 5, 4, 3, 5, 4, 6, 5], dtype=int32)
      batch.n_node
      >> DeviceArray([3, 5], dtype=int32)
      batch.n_edge
      >> DeviceArray([5, 10], dtype=int32)

    If a ``GraphsTuple`` does not contain any graphs, it will be dropped from the
    batch.

    This method is not compilable because it is data dependent.

    This functionality was implementation as  ``utils_tf.concat`` in the
    Tensorflow version of graph_nets.

    Args:
      graphs: sequence of ``GraphsTuple``s which will be batched into a single
        graph.
    """
    return _batch(graphs, np_=jnp)


def _batch(graphs, np_):
    """Returns batched graph given a list of graphs and a numpy-like module."""
    # Calculates offsets for sender and receiver arrays, caused by concatenating
    # the nodes arrays.
    # offsets = np_.cumsum(np_.array([0] + [np_.sum(g.num_nodes) for g in graphs[:-1]]))

    # def _map_concat(nests):
    #     def concat(*args):
    #         return np_.concatenate(args)

    #     return tree.tree_map(concat, *nests)

    # return Data(
    #     x=_map_concat([g.x for g in graphs]),
    #     edge_attr=_map_concat([g.edge_attr for g in graphs]),
    #     glob=_map_concat([g.glob for g in graphs]),
    #     senders=np_.concatenate([g.senders + o for g, o in zip(graphs, offsets)]),
    #     receivers=np_.concatenate([g.receivers + o for g, o in zip(graphs, offsets)]),
    # )

    return Batch.from_data_list(graphs)


def pad_with_graphs(graphs: Batch, n_node: int, n_edge: int, n_graph: int = 2) -> Batch:
    """Pads a ``Batch`` to size by adding computation preserving graphs.

    The ``Batch`` is padded by first adding a dummy graph which contains the
    padding nodes and edges, and then empty graphs without nodes or edges.

    The empty graphs and the dummy graph do not interfer with the graphnet
    calculations on the original graph, and so are computation preserving.

    The padding graph requires at least one node and one graph.

    This function does not support jax.jit, because the shape of the output
    is data-dependent.

    Args:
      graph: ``Batch`` padded with dummy graph and empty graphs.
      n_node: the number of nodes in the padded ``Batch``.
      n_edge: the number of edges in the padded ``Batch``.
      n_graph: the number of graphs in the padded ``Batch``. Default is 2,
        which is the lowest possible value, because we always have at least one
        graph in the original ``GraphsTuple`` and we need one dummy graph for the
        padding.

    Raises:
      ValueError: if the passed ``n_graph`` is smaller than 2.
      RuntimeError: if the given ``GraphsTuple`` is too large for the given
        padding.

    Returns:
      A padded ``Batch``.
    """

    assert type(graphs) == Batch, "graphs must be a Batch object"
    if n_graph < 2:
        raise ValueError(
            f"n_graph is {n_graph}, which is smaller than minimum value of 2."
        )
    graph = jax.device_get(graphs)
    graph.glob = graphs.glob
    # device_get is not copyinng graphs.glob, which is a dict. Fix this.

    pad_n_node = int(n_node - graph.num_nodes) - 1  # TODO why?
    pad_n_edge = int(n_edge - graph.num_edges)
    pad_n_graph = int(n_graph - graph.num_graphs)
    if pad_n_node <= 0 or pad_n_edge < 0 or pad_n_graph <= 0:
        raise RuntimeError(
            "Given graph is too large for the given padding. difference: "
            f"n_node {pad_n_node}, n_edge {pad_n_edge}, n_graph {pad_n_graph}"
        )

    pad_n_empty_graph = pad_n_graph - 1

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
    batch.num_graphs = batch.compute_num_graphs()
    return batch


if __name__ == "__main__":

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

    graph1 = _get_random_graph()
    graph2 = _get_random_graph()
    graph3 = _get_random_graph()
    batch = Batch([graph1, graph2])
    # print(batch)
    # batch._add_to_batch(graph3)
    # print(graph3)
    # print(batch)

    print(batch)
    print(batch.num_edges)
    pad_with_graphs(batch, 100, 200, 3)
    print(batch)
    print(batch.num_edges)
