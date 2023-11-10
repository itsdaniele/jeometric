from typing import Optional, Sequence
import jax.numpy as jnp
import jax

import jax.tree_util as tree

OptTensor = Optional[jnp.ndarray]


class Data:
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
        self._y = y
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

    def _tree_flatten(self):
        return (self.x, self.senders, self.receivers, self.edge_attr, self.glob), None

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __repr__(self) -> str:
        info = f"num_nodes: {self.num_nodes}"
        info += " | "
        info += f"x: {self.x.shape}"
        return info


class Batch(Data):
    def _batch(self, graphs):
        """Returns batched graph given a list of graphs and a numpy-like module."""
        # Calculates offsets for sender and receiver arrays, caused by concatenating
        # the nodes arrays.
        offsets = jnp.cumsum(
            jnp.array([0] + [jnp.sum(g.num_nodes) for g in graphs[:-1]])
        )

        def _map_concat(nests):
            def concat(*args):
                return jnp.concatenate(args)

            return tree.tree_map(concat, *nests)

        self.x = _map_concat([g.x for g in graphs])
        self.edge_attr = _map_concat([g.edge_attr for g in graphs])
        self.glob = _map_concat([g.glob for g in graphs])
        self.senders = jnp.concatenate([g.senders + o for g, o in zip(graphs, offsets)])

        self.receivers = jnp.concatenate(
            [g.receivers + o for g, o in zip(graphs, offsets)]
        )

        # self.batch should contain indices that map nodes to graph indices. They should be integers.
        self.batch = jnp.concatenate(
            [jnp.ones(g.num_nodes, dtype=jnp.int32) * i for i, g in enumerate(graphs)]
        )

    def __init__(self, graphs: Sequence[Data] = None):
        super().__init__()
        # self.graphs = graphs
        self._batch(graphs)
        self.num_graphs = self.compute_num_graphs()

    def _tree_flatten(self):
        return (self.graphs), None

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(children)

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

    batch = Batch([graph, graph2])

    print(graph)
