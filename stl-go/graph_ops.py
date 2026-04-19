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
    Return neighbors of agent_id at time t in an undirected graph, 
    if the weight falls in range W

    The graph is represented as a dense matrix G[t, i, j] of shape (T, N, N)
    """
    w_low, w_high = W
    mat = graphs[graph_type][t]
    n = mat.shape[0]

    neighbors = []
    for j in range(n):
        if j == agent_id:
            continue
        w = float(mat[agent_id, j])
        # w = 0 means there is no edge
        if w != 0.0 and (w_low <= w <= w_high):
            neighbors.append(j)

    return neighbors