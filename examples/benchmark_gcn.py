from jeometric.gnn import GCNLayer
from jeometric.data_reader import DataReader

import jax
import jax.numpy as jnp

from typing import List

from jeometric.data import Data, Batch

from flax import linen as nn

import optax

import time

import functools


from jax import tree_util

tree_util.register_pytree_node(Data, Data._tree_flatten, Data._tree_unflatten)
tree_util.register_pytree_node(Batch, Batch._tree_flatten, Batch._tree_unflatten)


def compute_loss(params, graph, label, net, num_graphs):
    """Computes loss."""
    pred_graph = net.apply(params, graph, num_graphs)
    preds = jax.nn.log_softmax(pred_graph)
    targets = jax.nn.one_hot(label, 2)

    # Cross entropy loss.
    loss = -jnp.sum(preds * targets) / num_graphs
    accuracy = jnp.mean(jnp.argmax(preds, axis=1) == label)
    return loss, accuracy


class GraphConvolutionalNetwork(nn.Module):
    input_dim: int
    hidden_dims: List[int]  # list of dimensions for each hidden layer
    output_dim: int
    aggregate_nodes_fn: str = "sum"
    add_self_edges: bool = False
    symmetric_normalization: bool = (True,)

    @nn.compact
    def __call__(self, graph: Data, num_graphs: int) -> Data:
        x, senders, receivers = graph.x, graph.senders, graph.receivers
        n_node = graph.num_nodes
        current_input_dim = self.input_dim

        for dim in self.hidden_dims:
            x = GCNLayer(
                input_dim=current_input_dim,
                output_dim=dim,
                aggregate_nodes_fn=self.aggregate_nodes_fn,
                add_self_edges=self.add_self_edges,
                symmetric_normalization=self.symmetric_normalization,
            )(x, senders, receivers, n_node)
            x = jax.nn.relu(x)
            current_input_dim = dim

        x = GCNLayer(
            input_dim=current_input_dim,
            output_dim=self.output_dim,
            aggregate_nodes_fn=self.aggregate_nodes_fn,
            add_self_edges=self.add_self_edges,
            symmetric_normalization=self.symmetric_normalization,
        )(x, senders, receivers, n_node)

        x = jax.ops.segment_sum(x, graph.batch, num_graphs)

        return x


# Example settings
batch_size = 32
input_dim = 9
hidden_dims = [64, 128]
output_dim = 2

gcn_model = GraphConvolutionalNetwork(
    input_dim=input_dim,
    hidden_dims=hidden_dims,
    output_dim=output_dim,
    aggregate_nodes_fn="sum",
    add_self_edges=True,
    symmetric_normalization=True,
)


# Initialize model parameters
key = jax.random.PRNGKey(0)
path = "/Users/danielepaliotta/Desktop/phd/projects/jax-geometric/jeometric/jeometric/ogbg-molhiv"
train_reader = DataReader(
    data_path=path,
    master_csv_path=path + "/master.csv",
    split_path=path + "/train.csv.gz",
    batch_size=32,
)

params = gcn_model.init(key, next(iter(train_reader)), 32)

# Initialize optimizer
optimizer = optax.adam(learning_rate=1e-3)
optimizer_state = optimizer.init(params)


# JIT-compiled version of train_step
def train_step_jit(params, optimizer_state: optax.OptState, batch: Data):
    compute_loss_fn = functools.partial(
        compute_loss, net=gcn_model, num_graphs=batch.num_graphs
    )
    compute_loss_fn = jax.jit(jax.value_and_grad(compute_loss_fn, has_aux=True))
    (loss, acc), grad = compute_loss_fn(params, batch, batch.glob["label"])
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state, loss, acc


# Non-JIT version of train_step (essentially the same as your original function)
def train_step_no_jit(params, optimizer_state: optax.OptState, batch: Data):
    compute_loss_fn = functools.partial(
        compute_loss, net=gcn_model, num_graphs=batch.num_graphs
    )
    compute_loss_fn = jax.value_and_grad(compute_loss_fn, has_aux=True)
    (loss, acc), grad = compute_loss_fn(params, batch, batch.glob["label"])
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    params = optax.apply_updates(params, updates)
    return params, optimizer_state, loss, acc


# Benchmarking function
def benchmark(train_step_fn, num_steps=100):
    start_time = time.time()
    for _ in range(num_steps):
        batch = next(train_reader)
        global params, optimizer_state  # Make sure to use the global variables
        params, optimizer_state, _, _ = train_step_fn(params, optimizer_state, batch)
    return time.time() - start_time


# Benchmark both versions
jit_time = benchmark(train_step_jit)
no_jit_time = benchmark(train_step_no_jit)

print(f"JIT Time: {jit_time} seconds")
print(f"Non-JIT Time: {no_jit_time} seconds")

"""
JIT Time: 47.3715717792511 seconds
Non-JIT Time: 71.37014412879944 seconds
"""
