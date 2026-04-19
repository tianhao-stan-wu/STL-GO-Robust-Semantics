"""
Graph operations: pure neighbor lookup, no algebra or recursion.

Graphs are represented as:
  dict { graph_type: list_of_dicts }
  where each dict maps (i, j) -> weight for that timestep.
  e.g. {"comm": [{(0,1): 1.0, (1,0): 2.0}, ...]}
"""


def get_neighbors(graphs, graph_type, t, agent_id, W, direction):
    """
    Return list of neighbor agent ids connected to agent_id at time t
    via graph_type, with edge weight inside interval W=[w1,w2].

    direction='in'  : edges (j -> agent_id), returns j's
    direction='out' : edges (agent_id -> j), returns j's
    """
    w1, w2 = W
    edges = graphs[graph_type][t]   # dict {(i,j): weight}
    neighbors = []

    for (i, j), w in edges.items():
        in_weight  = w1 <= w <= w2
        if direction == 'in'  and j == agent_id and in_weight:
            neighbors.append(i)
        elif direction == 'out' and i == agent_id and in_weight:
            neighbors.append(j)

    return neighbors