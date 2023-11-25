from typing import Optional, Sequence, Dict, Any

import jax.numpy as jnp

from jeometric.util import _batch

OptTensor = Optional[jnp.ndarray]


class Data:
    """
    Class for a homogeneous graph.

    Args:
        - x: An ndarray containing the node features. The shape is `(num_nodes, num_node_features)`.

        - senders: An ndarray containing the sender node indices for every edge. The shape is `(num_edges,)`.

        - receivers: An ndarray containing the receiver node indices for every edge. The shape is `(num_edges,)`.

        - edge_attr: An ndarray containing the edge features. The shape is `(num_edges, num_edge_features)`.

        - y: An ndarray containing the labels. The shape depends on whether the task is node-level, edge-level or graph-level.
        For node-level tasks, the shape is `(num_nodes, num_node_labels)`. For edge-level tasks, the shape is `(num_edges, num_edge_labels)`.

        - glob: A dictionary containing global attributes. The keys are strings and the values are ndarrays. The shape of the values is `(num_graphs, num_global_features)`.

    """

    def __init__(
        self,
        x: OptTensor = None,
        senders: OptTensor = None,
        receivers: OptTensor = None,
        edge_attr: OptTensor = None,
        y: OptTensor = None,
        glob: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._x = x
        self._senders = senders
        self._receivers = receivers
        self._edge_attr = edge_attr
        self._y = y
        self.glob = glob

    @property
    def x(self) -> OptTensor:
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self) -> OptTensor:
        return self._y

    @y.setter
    def y(self, y):
        self._y = y

    @property
    def edge_attr(self) -> OptTensor:
        return self._edge_attr

    @edge_attr.setter
    def edge_attr(self, edge_attr: jnp.ndarray):
        self._edge_attr = edge_attr

    @property
    def senders(self) -> OptTensor:
        return self._senders

    @property
    def receivers(self) -> OptTensor:
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

    def __str__(self) -> str:
        info = f"num_nodes: {self.num_nodes}"
        info += " | "
        info += f"x: {self.x.shape}"
        return info


class Batch(Data):
    """
    Class for a batch of homogeneous graphs. Inherits from `Data`. The single graphs are batched into a single, larger graph.

    ``Batch`` object are very similar to ``Data`` objects,
    but they have an additional attribute ``batch`` that maps each node to its corresponding graph in the batch.
    """

    def _add_to_batch(self, graph: Data):
        """
        Returns a new Batch instance with a single graph added. Does not modify the current object.

        Args:
            - graph: A ``Data`` instance to be added to the batch.
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

    def compute_num_graphs(self) -> int:
        return jnp.max(self.batch) + 1
