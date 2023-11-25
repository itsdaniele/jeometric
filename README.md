
# Jeometric - GNNs in JAX.

<p align="center">
  <img src="images/logo.png" alt="jeometric logo" width="350">
</p>

# Installation

```bash
pip install jeometric
```

NOTE: this library is still in the very early stages of development. Breaking changes might appear every other day ❤️.

# Usage

## Create a batch of graphs and forward through a GCN layer.

```python
import jax

from jeometric.data import Data, Batch
from jeometric.gnn import GCNLayer


# generate random node features and edges
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (10, 5))
senders = jax.random.randint(key, (10,), 0, 10)
receivers = jax.random.randint(key, (10,), 0, 10)

# create two graphs
graph1 = Data(x=x, senders=senders, receivers=receivers)
graph2 = Data(x=x, senders=senders, receivers=receivers)

# batch the graphs together in a single graphs
batch = Batch.from_data_list([graph1, graph2])

# create a GCN layer
gcn_layer = GCNLayer(input_dim=5, output_dim=1)

# initialize the layer and apply it to the batch
params = gcn_layer.init(key, batch.x, batch.senders, batch.receivers)
out = gcn_layer.apply(params, batch.x, batch.senders, batch.receivers)
# out.shape == (20, 1)

```

## Define a GNN with multiple GCN layers and sum-pooling.

```python
import jax
from flax import linen as nn

from jeometric.data import Data
from jeometric.ops import segment_sum
from jeometric.gnn import GCNLayer

from typing import List


class GraphConvolutionalNetwork(nn.Module):
    input_dim: int
    hidden_dims: List[int]
    output_dim: int

    @nn.compact
    def __call__(self, graph: Data, num_graphs: int) -> Data:
        x, senders, receivers = graph.x, graph.senders, graph.receivers
        current_input_dim = self.input_dim

        for dim in self.hidden_dims:
            x = GCNLayer(
                input_dim=current_input_dim,
                output_dim=dim,
            )(x, senders, receivers)
            x = jax.nn.relu(x)
            current_input_dim = dim

        x = GCNLayer(
            input_dim=current_input_dim,
            output_dim=self.output_dim,
        )(x, senders, receivers)

        x = segment_sum(x, graph.batch, num_graphs)

        return x
```

# Examples

Some examples can be find in the `examples` directory.

* `examples/train_molhiv.py` provides an example of training a graph convolutional network on `molhiv`.
* `examples/benchmark_gcn_molhiv` provides code to benchmark the jit and non-jit version.