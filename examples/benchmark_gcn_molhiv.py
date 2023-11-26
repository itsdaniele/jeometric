from typing import List
import time
import functools

import jax
import jax.numpy as jnp
from jax import tree_util

import optax

from flax import linen as nn

from jeometric.data import Data, Batch
from jeometric.gnn import GCNLayer
from jeometric.data_reader import DataReader
from jeometric.util import pad_with_graph, get_graph_padding_mask


tree_util.register_pytree_node(Data, Data._tree_flatten, Data._tree_unflatten)
tree_util.register_pytree_node(Batch, Batch._tree_flatten, Batch._tree_unflatten)


DATA_PATH = "./ogbg-molhiv"

BATCH_SIZE = 1
INPUT_DIM = 9
HIDDEN_DIMS = [64, 128]
OUTPUT_DIM = 2


def _nearest_bigger_power_of_two(x: int) -> int:
    """Computes the nearest power of two greater than x for padding."""
    y = 2
    while y < x:
        y *= 2
    return y


def pad_graph_to_nearest_power_of_two(batch: Batch):
    """
    Adapted from https://github.com/google-deepmind/jraph/tree/master.

    Pads a `Batch` to the nearest power of two.

    For example, if a `batch` has 7 nodes, 5 edges and 3 graphs, this method
    would pad the `GraphsTuple` nodes and edges:
      7 nodes --> 8 nodes (2^3)
      5 edges --> 8 edges (2^3)

    And since padding is accomplished using `pad_with_graph`, an extra
    graph and node is added:
      8 nodes --> 9 nodes
      3 graphs --> 4 graphs

    Args:
      - batch: a `Batch` (can be batch size 1).

    Returns:
      A ``Batch`` object batched to the nearest power of two.
    """
    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_two(batch.num_nodes) + 1
    pad_edges_to = _nearest_bigger_power_of_two(batch.num_edges)
    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    return pad_with_graph(batch, pad_nodes_to, pad_edges_to, task="graph")


def compute_loss(params, graph, label, net, num_graphs):
    """Computes loss."""
    pred_graph = net.apply(params, graph, num_graphs)
    preds = jax.nn.log_softmax(pred_graph)
    targets = jax.nn.one_hot(label, 2)

    # Since we have an extra 'dummy' graph in our batch due to padding, we want
    # to mask out any loss associated with the dummy graph.
    # Since we padded with `pad_with_graphs` we can recover the mask by using
    # get_graph_padding_mask.

    mask = get_graph_padding_mask(num_graphs)

    loss = -jnp.sum(preds * targets * mask[:, None]) / num_graphs

    accuracy = jnp.mean(jnp.argmax(preds, axis=1) == label)
    return loss, accuracy


class GraphConvolutionalNetwork(nn.Module):
    input_dim: int
    hidden_dims: List[int]  # list of dimensions for each hidden layer
    output_dim: int
    add_self_edges: bool = False
    symmetric_normalization: bool = (True,)

    @nn.compact
    def __call__(self, graph: Data, num_graphs: int) -> Data:
        x, senders, receivers = graph.x, graph.senders, graph.receivers
        current_input_dim = self.input_dim

        for dim in self.hidden_dims:
            x = GCNLayer(
                input_dim=current_input_dim,
                output_dim=dim,
                add_self_edges=self.add_self_edges,
                symmetric_normalization=self.symmetric_normalization,
            )(x, senders, receivers)
            x = jax.nn.relu(x)
            current_input_dim = dim

        x = GCNLayer(
            input_dim=current_input_dim,
            output_dim=self.output_dim,
            add_self_edges=self.add_self_edges,
            symmetric_normalization=self.symmetric_normalization,
        )(
            x,
            senders,
            receivers,
        )

        x = jax.ops.segment_sum(x, graph.batch, num_graphs)

        return x


gcn_model = GraphConvolutionalNetwork(
    input_dim=INPUT_DIM,
    hidden_dims=HIDDEN_DIMS,
    output_dim=OUTPUT_DIM,
    add_self_edges=True,
    symmetric_normalization=True,
)

# Initialize model parameters
key = jax.random.PRNGKey(0)
train_reader = DataReader(
    data_path=DATA_PATH,
    master_csv_path=DATA_PATH + "/master.csv",
    split_path=DATA_PATH + "/train.csv.gz",
    batch_size=BATCH_SIZE,
)

params = gcn_model.init(key, next(iter(train_reader)), num_graphs=1)

# Initialize optimizer
optimizer = optax.adam(learning_rate=1e-3)
optimizer_state = optimizer.init(params)


# JIT-compiled version of train_step
def train_step_jit(
    params,
    optimizer_state: optax.OptState,
    batch: Data,
):
    batch = pad_graph_to_nearest_power_of_two(batch)
    compute_loss_fn = functools.partial(
        compute_loss, net=gcn_model, num_graphs=batch.num_graphs
    )
    compute_loss_fn = jax.jit(jax.value_and_grad(compute_loss_fn, has_aux=True))
    (loss, acc), grad = compute_loss_fn(params, batch, batch.y)
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state, loss, acc


def train_step_no_jit(params, optimizer_state: optax.OptState, batch: Data):
    compute_loss_fn = functools.partial(
        compute_loss, net=gcn_model, num_graphs=batch.num_graphs
    )
    compute_loss_fn = jax.value_and_grad(compute_loss_fn, has_aux=True)
    (loss, acc), grad = compute_loss_fn(params, batch, batch.y)
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state, loss, acc


def benchmark(train_step_fn, num_steps=100):
    start_time = time.time()
    for _ in range(num_steps):
        batch = next(train_reader)
        global params, optimizer_state
        params, optimizer_state, _, _ = train_step_fn(params, optimizer_state, batch)
    return time.time() - start_time


jit_time = benchmark(train_step_jit)
no_jit_time = benchmark(train_step_no_jit)

print(f"JIT Time: {jit_time} seconds")
print(f"Non-JIT Time: {no_jit_time} seconds")

"""
Results on my machine with RTX 3090 with 100 steps:

JIT Time: 35.69718885421753 seconds
Non-JIT Time: 60.09366488456726 seconds
"""
