import warnings
import jraph
import numpy as np
import jax.numpy as jnp
from typing import Generator, Iterator, List, Optional
from random import shuffle

_NUMBER_FIELDS = ("n_node", "n_edge", "n_graph")

def _get_graph_size(graphs_tuple):
    n_node = np.sum(graphs_tuple.n_node)
    n_edge = len(graphs_tuple.senders)
    n_graph = len(graphs_tuple.n_node)
    return n_node, n_edge, n_graph

def _is_over_batch_size(graph, graph_batch_size):
    graph_size = _get_graph_size(graph)
    return any([x > y for x, y in zip(graph_size, graph_batch_size)])

def dynamically_batch(
    graphs_tuple_iterator: Iterator[jraph.GraphsTuple],
    n_node: int,
    n_edge: int,
    n_graph: int,
) -> Generator[jraph.GraphsTuple, None, None]:
    if n_graph < 2:
        raise ValueError(
            "The number of graphs in a batch size must be greater or "
            f"equal to `2` for padding with graphs, got {n_graph}."
        )
    valid_batch_size = (n_node - 1, n_edge, n_graph - 1)
    accumulated_graphs = []
    num_accumulated_nodes = 0
    num_accumulated_edges = 0
    num_accumulated_graphs = 0
    for element in graphs_tuple_iterator:
        element_nodes, element_edges, element_graphs = _get_graph_size(element)
        if _is_over_batch_size(element, valid_batch_size):
            if accumulated_graphs:
                yield jraph.batch_np(accumulated_graphs)

            graph_size = element_nodes, element_edges, element_graphs
            graph_size = {k: v for k, v in zip(_NUMBER_FIELDS, graph_size)}
            batch_size = {k: v for k, v in zip(_NUMBER_FIELDS, valid_batch_size)}
            raise RuntimeError(
                "Found graph bigger than batch size. Valid Batch "
                f"Size: {batch_size}, Graph Size: {graph_size}"
            )
        if not accumulated_graphs:
            accumulated_graphs = [element]
            num_accumulated_nodes = element_nodes
            num_accumulated_edges = element_edges
            num_accumulated_graphs = element_graphs
            continue
        else:
            if (
                (num_accumulated_graphs + element_graphs > n_graph - 1)
                or (num_accumulated_nodes + element_nodes > n_node - 1)
                or (num_accumulated_edges + element_edges > n_edge)
            ):
                yield jraph.batch_np(accumulated_graphs)
                accumulated_graphs = [element]
                num_accumulated_nodes = element_nodes
                num_accumulated_edges = element_edges
                num_accumulated_graphs = element_graphs
            else:
                accumulated_graphs.append(element)
                num_accumulated_nodes += element_nodes
                num_accumulated_edges += element_edges
                num_accumulated_graphs += element_graphs

    if accumulated_graphs:
        yield jraph.batch_np(accumulated_graphs)

class GraphDataLoader:
    def __init__(
        self,
        graphs: List[jraph.GraphsTuple],
        n_node: int,
        n_edge: int,
        n_graph: int,
        min_n_node: int = 1,
        min_n_edge: int = 1,
        min_n_graph: int = 1,
        shuffle: bool = True,
        n_mantissa_bits: Optional[int] = None,
    ):
        self.graphs = graphs
        self.n_node = n_node
        self.n_edge = n_edge
        self.n_graph = n_graph
        self.min_n_node = min_n_node
        self.min_n_edge = min_n_edge
        self.min_n_graph = min_n_graph
        self.shuffle = shuffle
        self.n_mantissa_bits = n_mantissa_bits
        self._length = None

        keep_graphs = [
            graph
            for graph in self.graphs
            if graph.n_node.sum() <= self.n_node - 1
            and graph.n_edge.sum() <= self.n_edge
        ]
        if len(keep_graphs) != len(self.graphs):
            warnings.warn(
                f"Discarded {len(self.graphs) - len(keep_graphs)} graphs due to size constraints."
            )
        self.graphs = keep_graphs

    def __iter__(self):
        graphs = self.graphs.copy()  # this is a shallow copy
        if self.shuffle:
            shuffle(graphs)

        for batched_graph in dynamically_batch(
            graphs,
            n_node=self.n_node,
            n_edge=self.n_edge,
            n_graph=self.n_graph,
        ):
            yield jraph.pad_with_graphs(
                    batched_graph, self.n_node, self.n_edge, self.n_graph
                )

    def __len__(self):
        if self.shuffle:
            raise NotImplementedError("Cannot compute length of shuffled data loader.")
        return self.approx_length()

    def approx_length(self):
        if self._length is None:
            self._length = 0
            for _ in self:
                self._length += 1
        return self._length

    def subset(self, i):
        graphs = self.graphs
        if isinstance(i, slice):
            graphs = graphs[i]
        if isinstance(i, int):
            graphs = graphs[:i]
        if isinstance(i, list):
            graphs = [graphs[j] for j in i]
        if isinstance(i, float):
            graphs = graphs[: int(len(graphs) * i)]

        return GraphDataLoader(
            graphs=graphs,
            n_node=self.n_node,
            n_edge=self.n_edge,
            n_graph=self.n_graph,
            min_n_node=self.min_n_node,
            min_n_edge=self.min_n_edge,
            min_n_graph=self.min_n_graph,
            shuffle=self.shuffle,
            n_mantissa_bits=self.n_mantissa_bits,
        )

    def replace_graphs(self, graphs: List[jraph.GraphsTuple]):
        return GraphDataLoader(
            graphs=graphs,
            n_node=self.n_node,
            n_edge=self.n_edge,
            n_graph=self.n_graph,
            min_n_node=self.min_n_node,
            min_n_edge=self.min_n_edge,
            min_n_graph=self.min_n_graph,
            shuffle=self.shuffle,
            n_mantissa_bits=self.n_mantissa_bits,
        )