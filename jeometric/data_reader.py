import pathlib
import numpy as np
import pandas as pd
from jeometric.data import Data

from jeometric.util import batch


class DataReader:
    """
    Data Reader for Open Graph Benchmark datasets.

    Adapted from https://github.com/google-deepmind/jraph/blob/master/jraph/ogb_examples/data_utils.py
    """

    def __init__(
        self,
        data_path,
        master_csv_path,
        split_path,
        batch_size=1,
    ):
        """Initializes the data reader by loading in data."""
        with pathlib.Path(master_csv_path).open("rt") as fp:
            self._dataset_info = pd.read_csv(fp, index_col=0)["ogbg-molhiv"]
        self._data_path = pathlib.Path(data_path)
        # Load edge information, and transpose into (senders, receivers).
        with pathlib.Path(data_path, "edge.csv.gz").open("rb") as fp:
            sender_receivers = pd.read_csv(
                fp, compression="gzip", header=None
            ).values.T.astype(np.int64)
            self._senders = sender_receivers[0]
            self._receivers = sender_receivers[1]
        # Load n_node and n_edge
        with pathlib.Path(data_path, "num-node-list.csv.gz").open("rb") as fp:
            self._n_node = pd.read_csv(fp, compression="gzip", header=None)
            self._n_node = self._n_node.astype(np.int64)[0].tolist()
        with pathlib.Path(data_path, "num-edge-list.csv.gz").open("rb") as fp:
            self._n_edge = pd.read_csv(fp, compression="gzip", header=None)
            self._n_edge = self._n_edge.astype(np.int64)[0].tolist()
        # Load node features
        with pathlib.Path(data_path, "node-feat.csv.gz").open("rb") as fp:
            self._nodes = (
                pd.read_csv(fp, compression="gzip", header=None)
                .astype(np.float32)
                .values
            )
        with pathlib.Path(data_path, "edge-feat.csv.gz").open("rb") as fp:
            self._edges = (
                pd.read_csv(fp, compression="gzip", header=None)
                .astype(np.float32)
                .values
            )
        with pathlib.Path(data_path, "graph-label.csv.gz").open("rb") as fp:
            self._labels = pd.read_csv(fp, compression="gzip", header=None).values

        with pathlib.Path(split_path).open("rb") as fp:
            self._split_idx = pd.read_csv(fp, compression="gzip", header=None).values.T[
                0
            ]

        self._repeat = False
        self._batch_size = batch_size
        self._generator = self._make_generator()
        self._max_nodes = int(np.max(self._n_node))
        self._max_edges = int(np.max(self._n_edge))

        # If n_node = [1,2,3], we create accumulated n_node [0,1,3,6] for indexing.
        self._accumulated_n_nodes = np.concatenate(
            (np.array([0]), np.cumsum(self._n_node))
        )
        # Same for n_edge
        self._accumulated_n_edges = np.concatenate(
            (np.array([0]), np.cumsum(self._n_edge))
        )

    @property
    def total_num_graphs(self):
        return len(self._n_node)

    def repeat(self):
        self._repeat = True

    def __iter__(self):
        return self

    def __next__(self):
        graphs = []
        for _ in range(self._batch_size):
            graph = next(self._generator)
            graphs.append(graph)
        return batch(graphs)

    def get_graph_by_idx(self, idx):
        """Gets a graph by an integer index."""
        # Gather the graph information
        label = self._labels[idx]
        # n_node = self._n_node[idx]
        # n_edge = self._n_edge[idx]
        node_slice = slice(
            self._accumulated_n_nodes[idx], self._accumulated_n_nodes[idx + 1]
        )
        edge_slice = slice(
            self._accumulated_n_edges[idx], self._accumulated_n_edges[idx + 1]
        )
        nodes = self._nodes[node_slice]
        edges = self._edges[edge_slice]
        senders = self._senders[edge_slice]
        receivers = self._receivers[edge_slice]
        # Molecular graphs are bi directional, but the serialization only
        # stores one way so we add the missing edges.
        return Data(
            x=nodes,
            edge_attr=np.concatenate([edges, edges]),
            senders=np.concatenate([senders, receivers]),
            receivers=np.concatenate([receivers, senders]),
            y=label,
        )

    def _make_generator(self):
        """Makes a single example generator of the loaded OGB data."""
        idx = 0
        while True:
            # If not repeating, exit when we've cycled through all the graphs.
            # Only return graphs within the split.
            if not self._repeat:
                if idx == self.total_num_graphs:
                    return
            else:
                # This will reset the index to 0 if we are at the end of the dataset.
                idx = idx % self.total_num_graphs
            if idx not in self._split_idx:
                idx += 1
                continue
            graph = self.get_graph_by_idx(idx)
            idx += 1
            yield graph
