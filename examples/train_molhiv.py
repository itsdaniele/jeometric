from jeometric.gnn import GCNLayer
from jeometric.data_reader import DataReader

import jax
import jax.numpy as jnp

from typing import List

from jeometric.data import Data, Batch
from jeometric.util import pad_with_graphs

from flax import linen as nn

import optax

import functools


from jax import tree_util

tree_util.register_pytree_node(Data, Data._tree_flatten, Data._tree_unflatten)
tree_util.register_pytree_node(Batch, Batch._tree_flatten, Batch._tree_unflatten)


def _nearest_bigger_power_of_two(x: int) -> int:
    """Computes the nearest power of two greater than x for padding."""
    y = 2
    while y < x:
        y *= 2
    return y


def pad_graph_to_nearest_power_of_two(batch: Batch):
    """Pads a batched `GraphsTuple` to the nearest power of two.

    For example, if a `GraphsTuple` has 7 nodes, 5 edges and 3 graphs, this method
    would pad the `GraphsTuple` nodes and edges:
      7 nodes --> 8 nodes (2^3)
      5 edges --> 8 edges (2^3)

    And since padding is accomplished using `jraph.pad_with_graphs`, an extra
    graph and node is added:
      8 nodes --> 9 nodes
      3 graphs --> 4 graphs

    Args:
      graphs_tuple: a batched `GraphsTuple` (can be batch size 1).

    Returns:
      A graphs_tuple batched to the nearest power of two.
    """
    # Add 1 since we need at least one padding node for pad_with_graphs.
    pad_nodes_to = _nearest_bigger_power_of_two(batch.num_nodes) + 1
    pad_edges_to = _nearest_bigger_power_of_two(batch.num_edges)
    # Add 1 since we need at least one padding graph for pad_with_graphs.
    # We do not pad to nearest power of two because the batch size is fixed.
    pad_graphs_to = batch.num_graphs + 1
    return pad_with_graphs(batch, pad_nodes_to, pad_edges_to, pad_graphs_to)


def compute_loss(params, graph, label, net, num_graphs):
    """Computes loss."""
    pred_graph = net.apply(params, graph, num_graphs)
    preds = jax.nn.log_softmax(pred_graph)
    targets = jax.nn.one_hot(label, 2)

    # Since we have an extra 'dummy' graph in our batch due to padding, we want
    # to mask out any loss associated with the dummy graph.
    # Since we padded with `pad_with_graphs` we can recover the mask by using
    # get_graph_padding_mask.
    # mask = jraph.get_graph_padding_mask(pred_graph)

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

        # Global average pooling. Graphs are batched so need to use scatter_add and such.
        # x = jax.ops.segment_sum(x, graph.batch, graph.num_graphs)  # TODO fix

        # the previous code doesn't work with jax.jit because of graph.num_graphs property. Fixing it with the following code
        # x = jax.jit(jax.ops.segment_sum, static_argnums=(2,))(
        #     x, graph.batch, graph.num_graphs
        # )
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
# key = jax.random.PRNGKey(0)
# path = "/Users/danielepaliotta/Desktop/phd/projects/jax-geometric/jeometric/jeometric/ogbg-molhiv"
# train_reader = DataReader(
#     data_path=path,
#     master_csv_path=path + "/master.csv",
#     split_path=path + "/train.csv.gz",
#     batch_size=32,
# )

# params = gcn_model.init(key, next(iter(train_reader)), 32)


# for data in iter(train_reader):
#     out = gcn_model.apply(params, data, )
#     break


# training loop

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


def train_step(
    params,
    optimizer_state: optax.OptState,
    batch: Data,
):
    """Train for a single step."""
    # Compute loss and accuracy

    compute_loss_fn = functools.partial(
        compute_loss, net=gcn_model, num_graphs=batch.num_graphs
    )

    compute_loss_fn = jax.jit(jax.value_and_grad(compute_loss_fn, has_aux=True))
    # loss, accuracy = compute_loss(params, batch, batch.y, gcn_model)

    # Compute gradients
    (loss, acc), grad = compute_loss_fn(params, batch, batch.glob["label"])
    # Update parameters
    updates, optimizer_state = optimizer.update(grad, optimizer_state)
    params = optax.apply_updates(params, updates)

    return params, optimizer_state, loss, acc


def evaluate(data_reader):
    """Evaluate the model on the test set."""
    loss = 0
    accuracy = 0
    num_batches = 0
    for batch in iter(data_reader):
        loss_, accuracy_ = compute_loss(params, batch, batch.glob["label"], gcn_model)
        loss += loss_
        accuracy += accuracy_
        num_batches += 1
    return loss / num_batches, accuracy / num_batches


test_reader = DataReader(
    data_path=path,
    master_csv_path=path + "/master.csv",
    split_path=path + "/test.csv.gz",
    batch_size=32,
)


train_reader.repeat()
num_training_steps = 1000
for idx in range(num_training_steps):
    batch = next(train_reader)
    batch = pad_graph_to_nearest_power_of_two(batch)
    params, optimizer_state, loss, accuracy = train_step(params, optimizer_state, batch)
    if idx % 100 == 0:
        print(f"step: {idx}, loss: {loss}, acc: {accuracy}")


print("Evaluating...")
test_loss, test_accuracy = evaluate(test_reader)
print(f"Test loss {test_loss} | Test accuracy {test_accuracy}")
