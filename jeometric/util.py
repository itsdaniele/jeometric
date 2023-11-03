import jax.numpy as jnp
import jax.tree_util as tree


from jeometric.data import Data, Batch

from typing import Sequence


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

    return Batch(graphs)
