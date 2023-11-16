from typing import Optional, Sequence
import jax.numpy as jnp
import jax

import jax.tree_util as tree

OptTensor = Optional[jnp.ndarray]


class Data:
    """
    Data class for graphs.

    Args:
        x: Node features.
        senders: Indices of the sender nodes.
        receivers: Indices of the receiver nodes.
        edge_attr: Edge features.
        y: Target values.
        glob: Global features.
    """

    def __init__(
        self,
        x: OptTensor = None,
        senders: OptTensor = None,
        receivers: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        glob: OptTensor = None,
        **kwargs,
    ):
        super().__init__()
        self._x = x
        self._senders = senders
        self._receivers = receivers
        self._edge_attr = edge_attr
        self._y = y  # this is never used by the rest of the library for now.
        self.glob = glob

    @property
    def x(self) -> jnp.ndarray:
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self) -> jnp.ndarray:
        return self._y

    @property
    def edge_attr(self) -> jnp.ndarray:
        return self._edge_attr

    @edge_attr.setter
    def edge_attr(self, edge_attr: jnp.ndarray):
        self._edge_attr = edge_attr

    @property
    def senders(self) -> jnp.ndarray:
        return self._senders

    @property
    def receivers(self) -> jnp.ndarray:
        return self._receivers

    @senders.setter
    def senders(self, senders: jnp.ndarray):
        self._senders = senders

    @receivers.setter
    def receivers(self, receivers: jnp.ndarray):
        self._receivers = receivers

    @property
    def num_nodes(self) -> int:
        return len(self._x)

    @property
    def num_edges(self) -> int:
        return len(self._senders)

    def _tree_flatten(self):
        return (
            self.x,
            self.senders,
            self.receivers,
            self.edge_attr,
            None,
            self.glob,
        ), None

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __repr__(self) -> str:
        info = f"num_nodes: {self.num_nodes}"
        info += " | "
        info += f"x: {self.x.shape}"
        return info


def _batch(graphs: Sequence[Data]):
    """
    Returns batched graph given a list of graphs.
    Adapated from https://github.com/google-deepmind/jraph/blob/master/jraph/_src/utils.py#L424
    """
    # Calculates offsets for sender and receiver arrays, caused by concatenating
    # the nodes arrays.
    offsets = jnp.cumsum(jnp.array([0] + [jnp.sum(g.num_nodes) for g in graphs[:-1]]))

    def _map_concat(nests):
        def concat(*args):
            return jnp.concatenate(args)

        return tree.tree_map(concat, *nests)

    x = _map_concat([g.x for g in graphs])
    edge_attr = _map_concat([g.edge_attr for g in graphs])
    glob = _map_concat([g.glob for g in graphs])
    senders = jnp.concatenate([g.senders + o for g, o in zip(graphs, offsets)])

    receivers = jnp.concatenate([g.receivers + o for g, o in zip(graphs, offsets)])

    # self.batch should contain indices that map nodes to graph indices. They should be integers.
    batch = jnp.concatenate(
        [jnp.ones(g.num_nodes, dtype=jnp.int32) * i for i, g in enumerate(graphs)]
    )

    return x, senders, receivers, edge_attr, glob, batch


class Batch(Data):

    def _add_to_batch(self, graph: Data):
        """
        Returns a new Batch instance with the single graph added. Does not modify the current object.
        """
        offset = jnp.sum(self.x.shape[0])
        new_x = jnp.concatenate([self.x, graph.x])
        new_edge_attr = (
            None
            if self.edge_attr is None
            else jnp.concatenate([self.edge_attr, graph.edge_attr])
        )

        new_glob = (
            None
            if self.glob is None
            else {k: jnp.concatenate([v, graph.glob[k]]) for k, v in self.glob.items()}
        )

        new_senders = jnp.concatenate([self.senders, graph.senders + offset])
        new_receivers = jnp.concatenate([self.receivers, graph.receivers + offset])

        new_batch = jnp.concatenate(
            [self.batch, jnp.ones(graph.num_nodes, dtype=jnp.int32) * self.num_graphs]
        )

        # Create a new Batch object with the updated values
        new_batch_obj = Batch(new_x, new_senders, new_receivers, new_edge_attr, new_glob, new_batch)
        return new_batch_obj

    def __init__(self, x, senders, receivers, edge_attr, glob, batch):
        super().__init__(x, senders, receivers, edge_attr, None, glob)  # TODO fix
        self.batch = batch

        # When using jax.jit, `self.num_graphs` is treated as a dynamic value to trace. I think this is not optimal. 
        self.num_graphs = self.compute_num_graphs() 

    @classmethod
    def from_data_list(cls, graphs: Sequence[Data]):
        """
        Returns a `Batch` instance from a list of `Data` instances.
        """
        x, senders, receivers, edge_attr, glob, batch = _batch(graphs)
        return cls(x, senders, receivers, edge_attr, glob, batch)

    def _tree_flatten(self):
        """
        This is needed to be able to use jax.jit.
        """
        return (
            self.x,
            self.senders,
            self.receivers,
            self.edge_attr,
            self.glob,
            self.batch,
            ), None

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        """
        This is needed to be able to use jax.jit.
        """
        return cls(*children)

    def compute_num_graphs(self):
        return jnp.max(self.batch) + 1


if __name__ == "__main__":
    seed = 42
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, shape=(10, 100))
    senders = jnp.array(list(range(10)))
    receivers = jnp.array(list(range(10)))
    graph = Data(x=x, senders=senders, receiver=receivers)

    x2 = x.copy()
    senders2 = senders.copy()
    receivers2 = receivers.copy()
    graph2 = Data(x=x2, senders=senders2, receiver=receivers2)

    batch = Batch.from_data_list([graph, graph2])

    print(graph)
