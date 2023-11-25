from __future__ import annotations

import jax.numpy as jnp
import jax.tree_util as tree

import jeometric

from typing import Sequence

import jax
import numpy as np


def get_graph_padding_mask(num_graphs: int) -> jnp.ndarray:
    """Returns a mask for the graphs of a padded graph. For now, padded graphs only have 1 additional graph.

    Args:
      num_graphs: number of graphs in ``Batch`` padded using ``pad_with_graph``.

    Returns:
      Boolean array of shape [total_num_graphs] containing True for real graphs,
      and False for padding graph.
    """
    n_padding_graph = 1
    total_num_graphs = num_graphs
    return _get_mask(padding_length=n_padding_graph, full_length=total_num_graphs)


def _get_mask(padding_length, full_length):
    valid_length = full_length - padding_length
    return jnp.arange(full_length, dtype=jnp.int32) < valid_length


def batch(graphs: Sequence[jeometric.data.Data]) -> jeometric.data.Batch:  # TODO test
    """Returns a batched graph given a list of graphs.

    This method wraps `_batch`.

    Args:
      graphs: sequence of ``GraphsTuple``s which will be batched into a single
        graph.
    """
    from jeometric.data import Batch

    return Batch(*_batch(graphs))


def _batch(graphs: Sequence[jeometric.data.Data]):
    """
    Returns batched graph given a list of graphs.
    This function will concatenate the ``x``, ``senders``, ``receivers``, ``edge_attr``, ``y`` and ``globals``.

    For ``senders`` and ``receivers``, offsets are computed so that connectivity
    remains valid for the new node indices.

    A ``batch`` array is also created, which maps each node to its graph index.


    Adapated from https://github.com/google-deepmind/jraph/blob/master/jraph/_src/utils.py#L424
    """
    # Calculates offsets for sender and receiver arrays, caused by concatenating
    # the nodes arrays.
    offsets = jnp.cumsum(jnp.array([0] + [jnp.sum(g.num_nodes) for g in graphs[:-1]]))

    def _map_concat(nests):
        def concat(*args):
            return jnp.concatenate(args)

        return tree.tree_map(concat, *nests)

    def _map_concat_dict(nests):
        def concat_dict(*args):
            return {k: jnp.concatenate([a[k] for a in args]) for k in args[0].keys()}

        return tree.tree_map(concat_dict, *nests)

    x = _map_concat([g.x for g in graphs])
    senders = jnp.concatenate([g.senders + o for g, o in zip(graphs, offsets)])
    receivers = jnp.concatenate([g.receivers + o for g, o in zip(graphs, offsets)])
    y = _map_concat([g.y for g in graphs])
    edge_attr = _map_concat([g.edge_attr for g in graphs])
    glob = _map_concat_dict(
        [g.glob for g in graphs]
    )  # TODO fix concatenation as this is a dict

    # self.batch should contain indices that map nodes to graph indices. They should be integers.
    batch = jnp.concatenate(
        [jnp.ones(g.num_nodes, dtype=jnp.int32) * i for i, g in enumerate(graphs)]
    )

    return x, senders, receivers, edge_attr, y, glob, batch


def pad_with_graph(
    graphs: jeometric.data.Batch, n_node: int, n_edge: int, task: str = "node"
) -> jeometric.data.Batch:
    """Adapted from https://github.com/google-deepmind/jraph.

    Pads a ``Batch`` to size by adding a single computation preserving graph.

    The ``Batch`` is padded by adding a dummy graph which contains the
    padding nodes and edges.

    The dummy graph do not interfer with the graphnet
    calculations on the original graph, and so is computation preserving.


    The padding graph requires at least one node.

    If task is "node", the dummy graph ``y`` will have shape [n_node, 1].
    If task is "graph", the dummy graph ``y`` will have shape [1,].

    This function does not support jax.jit, because the shape of the output
    is data-dependent.

    Args:
      graphs: ``Batch`` to be padded with dummy graph.
      n_node: the number of nodes in the padded ``Batch``.
      n_edge: the number of edges in the padded ``Batch``.

    Raises:
      ValueError: if the passed ``n_graph`` is smaller than 2.
      RuntimeError: if the given ``GraphsTuple`` is too large for the given
        padding.

    Returns:
      A padded ``Batch``.
    """

    assert type(graphs) == jeometric.data.Batch, "graphs must be a Batch object"
    assert task in ["node", "graph"], "task must be either 'node' or 'graph'"
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

    def tree_y_pad(leaf):
        if task == "node":
            return np.zeros((pad_n_node, 1), dtype=leaf.dtype)
        elif task == "graph":
            return np.zeros((1,), dtype=leaf.dtype)

    # for globs, leaf is a dict. we want to create zero-value dicts with the same keys
    def tree_globs_pad(leaf):
        return np.zeros(1)

    graph = jeometric.data.Data(
        x=tree.tree_map(tree_nodes_pad, graph.x),
        edge_attr=tree.tree_map(tree_edges_pad, graph.edge_attr),
        senders=jnp.zeros(pad_n_edge, dtype=jnp.int32),
        receivers=jnp.zeros(pad_n_edge, dtype=jnp.int32),
        y=tree.tree_map(tree_y_pad, graph.y),
        glob=tree.tree_map(tree_globs_pad, graph.glob),
    )

    batch = graphs._add_to_batch(graph)
    return batch
