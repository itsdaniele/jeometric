from typing import Any, NamedTuple, Iterable, Mapping, Union, Optional, List
import jax.numpy as jnp
import jax


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

    @property
    def y(self) -> jnp.ndarray:
        return self._y

    @property
    def edge_attr(self) -> jnp.ndarray:
        return self._edge_attr

    @property
    def senders(self) -> jnp.ndarray:
        return self._senders

    @property
    def receivers(self) -> jnp.ndarray:
        return self._receivers

    @property
    def num_nodes(self) -> int:
        return len(self._x)

    def __repr__(self) -> str:
        info = f"num_nodes: {self.num_nodes}"
        info += " | "
        info += f"x: {self.x.shape}"

        return info


class Batch(Data):
    def __init__(self, graphs: List[Data] = None):

        super().__init__()

        self.x = jnp.concatenate([g.x for g in graphs], axis=0)
        self.senders = jnp.concatenate([g.senders for g in graphs], axis=-1)
        self.receivers = jnp.concatenate([g.receivers for g in graphs], axis=-1)
        self.edge_attr = jnp.concatenate([g.edge_attr for g in graphs], axis=0)
        self.y = [g.y for g in graphs]


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

