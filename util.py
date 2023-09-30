import os.path as osp
import ssl
import sys
import urllib
from typing import Optional

import errno


import warnings
from itertools import repeat

import torch

try:
    import cPickle as pickle
except ImportError:
    import pickle

import jax.numpy as jnp
import jax.tree_util as tree


from data import Data

from typing import Sequence


def makedirs(path: str):
    r"""Recursively creates a directory.

    Args:
        path (str): The path to create.
    """
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def download_url(
    url: str, folder: str, log: bool = True, filename: Optional[str] = None
):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (str): The URL.
        folder (str): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    if filename is None:
        filename = url.rpartition("/")[2]
        filename = filename if filename[0] == "?" else filename.split("?")[0]

    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log and "pytest" not in sys.modules:
            print(f"Using existing file {filename}", file=sys.stderr)
        return path

    if log and "pytest" not in sys.modules:
        print(f"Downloading {url}", file=sys.stderr)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        # workaround for https://bugs.python.org/issue42853
        while True:
            chunk = data.read(10 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return path


def read_planetoid_data(folder, prefix):
    names = ["x", "tx", "allx", "y", "ty", "ally", "graph", "test.index"]
    items = [read_file(folder, prefix, name) for name in names]
    x, tx, allx, y, ty, ally, graph, test_index = items
    train_index = torch.arange(y.size(0), dtype=torch.long)
    val_index = torch.arange(y.size(0), y.size(0) + 500, dtype=torch.long)
    sorted_test_index = test_index.sort()[0]

    if prefix.lower() == "citeseer":
        # There are some isolated nodes in the Citeseer graph, resulting in
        # none consecutive test indices. We need to identify them and add them
        # as zero vectors to `tx` and `ty`.
        len_test_indices = (test_index.max() - test_index.min()).item() + 1

        tx_ext = torch.zeros(len_test_indices, tx.size(1))
        tx_ext[sorted_test_index - test_index.min(), :] = tx
        ty_ext = torch.zeros(len_test_indices, ty.size(1))
        ty_ext[sorted_test_index - test_index.min(), :] = ty

        tx, ty = tx_ext, ty_ext

    if prefix.lower() == "nell.0.001":
        tx_ext = torch.zeros(len(graph) - allx.size(0), x.size(1))
        tx_ext[sorted_test_index - allx.size(0)] = tx

        ty_ext = torch.zeros(len(graph) - ally.size(0), y.size(1))
        ty_ext[sorted_test_index - ally.size(0)] = ty

        tx, ty = tx_ext, ty_ext

        x = torch.cat([allx, tx], dim=0)
        x[test_index] = x[sorted_test_index]

        # Creating feature vectors for relations.
        row, col, value = SparseTensor.from_dense(x).coo()
        rows, cols, values = [row], [col], [value]

        mask1 = index_to_mask(test_index, size=len(graph))
        mask2 = index_to_mask(torch.arange(allx.size(0), len(graph)), size=len(graph))
        mask = ~mask1 | ~mask2
        isolated_index = mask.nonzero(as_tuple=False).view(-1)[allx.size(0) :]

        rows += [isolated_index]
        cols += [torch.arange(isolated_index.size(0)) + x.size(1)]
        values += [torch.ones(isolated_index.size(0))]

        x = SparseTensor(
            row=torch.cat(rows), col=torch.cat(cols), value=torch.cat(values)
        )
    else:
        x = torch.cat([allx, tx], dim=0)
        x[test_index] = x[sorted_test_index]

    y = torch.cat([ally, ty], dim=0).max(dim=1)[1]
    y[test_index] = y[sorted_test_index]

    train_mask = index_to_mask(train_index, size=y.size(0))
    val_mask = index_to_mask(val_index, size=y.size(0))
    test_mask = index_to_mask(test_index, size=y.size(0))

    edge_index = edge_index_from_dict(graph, num_nodes=y.size(0))

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data


def read_file(folder, prefix, name):
    path = osp.join(folder, f"ind.{prefix.lower()}.{name}")

    if name == "test.index":
        return read_txt_array(path, dtype=torch.long)

    with open(path, "rb") as f:
        if sys.version_info > (3, 0):
            warnings.filterwarnings("ignore", ".*`scipy.sparse.csr` name.*")
            out = pickle.load(f, encoding="latin1")
        else:
            out = pickle.load(f)

    if name == "graph":
        return out

    out = out.todense() if hasattr(out, "todense") else out
    out = torch.from_numpy(out).to(torch.float)
    return out


def edge_index_from_dict(graph_dict, num_nodes=None):
    row, col = [], []
    for key, value in graph_dict.items():
        row += repeat(key, len(value))
        col += value
    edge_index = torch.stack([torch.tensor(row), torch.tensor(col)], dim=0)

    # NOTE: There are some duplicated edges and self loops in the datasets.
    #       Other implementations do not remove them!
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    return edge_index


def batch(graphs: Sequence[Data]) -> Data:
  """Returns a batched graph given a list of graphs.

  This method will concatenate the ``nodes``, ``edges`` and ``globals``,
  ``n_node`` and ``n_edge`` of a sequence of ``GraphsTuple`` along axis 0. For
  ``senders`` and ``receivers``, offsets are computed so that connectivity
  remains valid for the new node indices.

  For example::

    key = jax.random.PRNGKey(0)
    graph_1 = GraphsTuple(nodes=jax.random.normal(key, (3, 64)),
                      edges=jax.random.normal(key, (5, 64)),
                      senders=jnp.array([0,0,1,1,2]),
                      receivers=[1,2,0,2,1],
                      n_node=jnp.array([3]),
                      n_edge=jnp.array([5]),
                      globals=jax.random.normal(key, (1, 64)))
    graph_2 = GraphsTuple(nodes=jax.random.normal(key, (5, 64)),
                      edges=jax.random.normal(key, (10, 64)),
                      senders=jnp.array([0,0,1,1,2,2,3,3,4,4]),
                      receivers=jnp.array([1,2,0,2,1,0,2,1,3,2]),
                      n_node=jnp.array([5]),
                      n_edge=jnp.array([10]),
                      globals=jax.random.normal(key, (1, 64)))
    batch = graph.batch([graph_1, graph_2])

    batch.nodes.shape
    >> (8, 64)
    batch.edges.shape
    >> (15, 64)
    # Offsets computed on senders and receivers
    batch.senders
    >> DeviceArray([0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7], dtype=int32)
    batch.receivers
    >> DeviceArray([1, 2, 0, 2, 1, 4, 5, 3, 5, 4, 3, 5, 4, 6, 5], dtype=int32)
    batch.n_node
    >> DeviceArray([3, 5], dtype=int32)
    batch.n_edge
    >> DeviceArray([5, 10], dtype=int32)

  If a ``GraphsTuple`` does not contain any graphs, it will be dropped from the
  batch.

  This method is not compilable because it is data dependent.

  This functionality was implementation as  ``utils_tf.concat`` in the
  Tensorflow version of graph_nets.

  Args:
    graphs: sequence of ``GraphsTuple``s which will be batched into a single
      graph.
  """
  return _batch(graphs, np_=jnp)

def _batch(graphs, np_):
  """Returns batched graph given a list of graphs and a numpy-like module."""
  # Calculates offsets for sender and receiver arrays, caused by concatenating
  # the nodes arrays.
  offsets = np_.cumsum(
      np_.array([0] + [np_.sum(g.num_nodes) for g in graphs[:-1]]))

  def _map_concat(nests):
    concat = lambda *args: np_.concatenate(args)
    return tree.tree_map(concat, *nests)

  return Data(
      x=_map_concat([g.x for g in graphs]),
      edge_attr=_map_concat([g.edge_attr for g in graphs]),
      glob=_map_concat([g.glob for g in graphs]),
      senders=np_.concatenate([g.senders + o for g, o in zip(graphs, offsets)]),
      receivers=np_.concatenate(
          [g.receivers + o for g, o in zip(graphs, offsets)]))
