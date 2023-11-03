from jeometric.gnn import GCNLayer
from jeometric.data_reader import DataReader

import jax
import jax.numpy as jnp

from typing import List

from jeometric.data import Data

from flax import linen as nn


def compute_loss(params, graph, label, net):
    """Computes loss."""
    pred_graph = net.apply(params, graph)
    preds = jax.nn.log_softmax(pred_graph.globals)
    targets = jax.nn.one_hot(label, 2)

    # Since we have an extra 'dummy' graph in our batch due to padding, we want
    # to mask out any loss associated with the dummy graph.
    # Since we padded with `pad_with_graphs` we can recover the mask by using
    # get_graph_padding_mask.
    mask = jraph.get_graph_padding_mask(pred_graph)

    # Cross entropy loss.
    loss = -jnp.mean(preds * targets * mask[:, None])

    # Accuracy taking into account the mask.
    accuracy = jnp.sum(
        (jnp.argmax(pred_graph.globals, axis=1) == label) * mask
    ) / jnp.sum(mask)
    return loss, accuracy


class GraphConvolutionalNetwork(nn.Module):
    input_dim: int
    hidden_dims: List[int]  # list of dimensions for each hidden layer
    output_dim: int
    aggregate_nodes_fn: str = "sum"
    add_self_edges: bool = False
    symmetric_normalization: bool = True

    @nn.compact
    def __call__(self, graph: Data) -> Data:
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
        x = jax.ops.segment_sum(x, graph.batch, graph.num_graphs)

        return x


# Example settings
batch_size = 32
input_dim = 9
hidden_dims = [64, 128]
output_dim = 10

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

params = gcn_model.init(key, next(iter(train_reader)))


for data in iter(train_reader):
    out = gcn_model.apply(params, data)
    break

print(out.shape)
