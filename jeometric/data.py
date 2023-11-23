from typing import Optional, Sequence, Dict, Any

import jax.numpy as jnp
import jax

from jeometric.util import _batch

OptTensor = Optional[jnp.ndarray]


class Data:
    """
    Data class for graphs.

    Args:
        x: Node features.
        senders: Sender indices.
        receivers: Receiver indices.
        edge_attr: Edge attributes.
        y: Target values.
        glob: Global attributes.

    """

    def __init__(
        self,
        x: OptTensor = None,
        senders: OptTensor = None,
        receivers: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        glob: Dict[str, Any] = None,
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

    @y.setter
    def y(self, y):
        self._y = y

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
            self.y,
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


class Batch(Data):
    def _add_to_batch(self, graph: Data):
        """
        Returns a new Batch instance with a single graph added. Does not modify the current object.
        """
        offset = jnp.sum(self.x.shape[0])
        new_x = jnp.concatenate([self.x, graph.x])

        new_senders = jnp.concatenate([self.senders, graph.senders + offset])
        new_receivers = jnp.concatenate([self.receivers, graph.receivers + offset])
        new_edge_attr = (
            None
            if self.edge_attr is None
            else jnp.concatenate([self.edge_attr, graph.edge_attr])
        )

        new_y = None if self.y is None else jnp.concatenate([self.y, graph.y])

        new_glob = (
            None
            if self.glob is None
            else {k: jnp.concatenate([v, graph.glob[k]]) for k, v in self.glob.items()}
        )

        new_batch = jnp.concatenate(
            [self.batch, jnp.ones(graph.num_nodes, dtype=jnp.int32) * self.num_graphs]
        )

        # Create a new Batch object with the updated values
        new_batch_obj = Batch(
            new_x, new_senders, new_receivers, new_edge_attr, new_y, new_glob, new_batch
        )
        return new_batch_obj

    def __init__(
        self,
        x: OptTensor = None,
        senders: OptTensor = None,
        receivers: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        glob: Optional[Dict[str, Any]] = None,
        batch: OptTensor = None,
    ):
        super().__init__(x, senders, receivers, edge_attr, y, glob)
        self.batch = batch

        # When using jax.jit, `self.num_graphs` is treated as a dynamic value to trace. I think this is not optimal.
        self.num_graphs = self.compute_num_graphs()

    @classmethod
    def from_data_list(cls, graphs: Sequence[Data]):
        """
        Returns a `Batch` instance from a list of `Data` instances.
        """
        x, senders, receivers, edge_attr, y, glob, batch = _batch(graphs)
        return cls(x, senders, receivers, edge_attr, y, glob, batch)

    def _tree_flatten(self):
        """
        This is needed to be able to use jax.jit.
        """
        return (
            self.x,
            self.senders,
            self.receivers,
            self.edge_attr,
            self.y,
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
