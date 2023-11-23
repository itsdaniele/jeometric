from jeometric.data import Data, Batch

import jax
import jax.numpy as jnp


def test_batch():
    seed = 42
    key = jax.random.PRNGKey(seed)
    x = jax.random.normal(key, shape=(10, 100))
    senders = jnp.array(list(range(10)))
    receivers = jnp.array(list(range(10)))
    graph = Data(x=x, senders=senders, receivers=receivers)

    x2 = x.copy()
    senders2 = senders.copy()
    receivers2 = receivers.copy()
    graph2 = Data(x=x2, senders=senders2, receivers=receivers2)

    batch = Batch.from_data_list([graph, graph2])

    assert batch.num_nodes == 20
    assert batch.num_edges == 20
    assert batch.num_graphs == 2

    assert jnp.all(batch.x == jnp.concatenate([x, x2]))
    assert batch.batch is not None
    assert batch.batch.shape == (20,)
    assert batch.batch.dtype == jnp.int32
    assert jnp.all(batch.batch == jnp.concatenate([jnp.zeros(10), jnp.ones(10)]))
    assert jnp.all(batch.senders == jnp.concatenate([senders, senders2 + 10]))
    assert jnp.all(batch.receivers == jnp.concatenate([receivers, receivers2 + 10]))

    # Test that the batch is not modified when adding a graph to it.
    batch2 = batch._add_to_batch(graph)
    assert batch.num_nodes == 20
    assert batch.num_edges == 20
    assert batch.num_graphs == 2
    assert jnp.all(batch.x == jnp.concatenate([x, x2]))
    assert batch.batch is not None
    assert batch.batch.shape == (20,)
    assert batch.batch.dtype == jnp.int32
    assert jnp.all(batch.batch == jnp.concatenate([jnp.zeros(10), jnp.ones(10)]))
    assert jnp.all(batch.senders == jnp.concatenate([senders, senders2 + 10]))
    assert jnp.all(batch.receivers == jnp.concatenate([receivers, receivers2 + 10]))

    # Test that the new batch is correct.
    assert batch2.num_nodes == 30
    assert batch2.num_edges == 30
    assert batch2.num_graphs == 3
    assert jnp.all(batch2.x == jnp.concatenate([x, x2, x]))
    assert batch2.batch is not None
    assert batch2.batch.shape == (30,)
    assert batch2.batch.dtype == jnp.int32
    assert jnp.all(
        batch2.batch == jnp.concatenate([jnp.zeros(10), jnp.ones(10), 2 * jnp.ones(10)])
    )
    assert jnp.all(
        batch2.senders == jnp.concatenate([senders, senders2 + 10, senders + 20])
    )
    assert jnp.all(
        batch2.receivers
        == jnp.concatenate([receivers, receivers2 + 10, receivers + 20])
    )
