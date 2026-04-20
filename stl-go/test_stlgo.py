"""
Simplest STL-GO test for generated 2D trajectories and graphs.

What this script does:
  1. Loads the generated trajectory and graphs from a .npz file.
  2. Evaluates one STL-GO specifications

Assumptions:
  - Graphs are undirected.
  - The generated file contains keys:
      positions, dist_matrix, G_dist, G_sense, G_comm
  - Graph arrays are dense adjacency / weight matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Tuple

import numpy as np

from syntax import Formula, TrueF, Predicate, MultiAgentPredicate, Neg, And, Until, In, Out, AgentFormula, EXV, FAV
from algebra import MinMaxAlgebra, BooleanAlgebra
from graph_ops import get_neighbors
from evaluator import evaluate


algebras = {"minmax": MinMaxAlgebra(),
            "bool": BooleanAlgebra()
}

# ---------------------------------------------------------------------------
# Atomic predicate
# ---------------------------------------------------------------------------

def y_nonneg(state: np.ndarray) -> float:
    """Robustness of y >= 0."""
    return float(state[1])


ATOMIC_PREDICATES = {
    "y_nonneg": y_nonneg,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_generated_data(path: str) -> Dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    required = {"positions", "dist_matrix", "G_dist", "G_sense", "G_comm"}
    missing = required.difference(data.keys())
    if missing:
        raise KeyError(f"Missing keys in {path}: {sorted(missing)}")
    return {k: data[k] for k in required}


# ---------------------------------------------------------------------------
# Example specification
# ---------------------------------------------------------------------------

def build_In(predicate, agent_id, graph_type, weight_interval, count_interval, quantifier, aggregator):
    pred = Predicate(mu=ATOMIC_PREDICATES[predicate], label=predicate)

    phi_in = AgentFormula(
        agent_id=agent_id,
        child=In(
            graph_types=[graph_type],
            W=weight_interval,
            E=count_interval,
            quantifier=quantifier,
            aggregator=aggregator,
            child=pred,
        ),
    )

    return phi_in


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_path = "../trajectory_data/2D_graphs.npz"
    data = load_generated_data(data_path)

    positions = data["positions"]  # (T, N, 2)
    T, N, _ = positions.shape

    trajs = {i: positions[:, i, :] for i in range(N)}
    graphs = {
        "dist": data["dist_matrix"],
        "sense": data["G_sense"],
        "comm": data["G_comm"],
    }

    # supported aggreator: "min_max", "counting", "averaging"
    phi_in = In(
        graph_types=["dist"],
        W=[0.0, 5.0], 
        E=[1, 4], 
        quantifier="exists", 
        aggregator="hybrid",
        child=Predicate(mu=ATOMIC_PREDICATES["y_nonneg"], label="y_nonneg"))

    algebra = algebras["minmax"]

    rob_in = evaluate(trajs, graphs, phi_in, algebra, t=0, agent_id=0)
    print(f"In robustness : {rob_in}")

    phi_exv = EXV(child=phi_in)
    rob_exv = evaluate(trajs, graphs, phi_exv, algebra, t=0, agent_id=0)
    print(f"EXV In robustness : {rob_exv}")

    phi_fav = FAV(child=phi_in)
    rob_fav = evaluate(trajs, graphs, phi_fav, algebra, t=0, agent_id=0)
    print(f"FAV In robustness : {rob_fav}")


if __name__ == "__main__":
    main()


