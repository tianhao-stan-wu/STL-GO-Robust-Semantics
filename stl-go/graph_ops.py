"""
Graph operations: pure neighbor lookup, no algebra or recursion.

Graphs are represented as dense matrix
"""
from typing import Dict, List, Tuple
import numpy as np

def get_neighbors(
    graphs: Dict[str, np.ndarray],
    graph_type: str,
    t: int,
    agent_id: int,
    W: Tuple[float, float],
) -> List[int]:
    """
    Return neighbors of agent_id at time t in an undirected graph.

    The graph is represented as a dense matrix G[t, i, j].
    For an undirected graph, we treat i--j as present if either orientation
    carries a nonzero weight. The effective weight is the max of the two.
    """
    w_low, w_high = W
    mat = graphs[graph_type][t]
    n = mat.shape[0]

    neighbors = []
    for j in range(n):
        if j == agent_id:
            continue
        w = max(float(mat[agent_id, j]), float(mat[j, agent_id]))
        if w != 0.0 and (w_low <= w <= w_high):
            neighbors.append(j)

    return neighbors