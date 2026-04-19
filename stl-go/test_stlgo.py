"""
Simplest STL-GO test for generated 2D trajectories and graphs.

What this script does:
  1. Loads the generated trajectory and graphs from a .npz file.
  2. Evaluates one STL-GO specification with the In graph operator.
  3. Evaluates the same specification with the Out graph operator.
  4. Uses the min-max algebra and min_max graph aggregation.

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

from syntax import Predicate, In, Out, AgentFormula
from algebra import MinMaxAlgebra
from graph_ops import get_neighbors
from evaluator import evaluate


@dataclass
class TestConfig:
    data_path: str = "../trajectory_data/2D_graphs.npz"
    agent_id: int = 0
    time_index: int = 0

    # Use the distance graph for the simplest possible test.
    graph_type: str = "dist"

    # For weighted distance graph: keep edges whose distance is in this interval.
    # Since the graph is undirected, In and Out will be equivalent.
    weight_interval: Tuple[float, float] = (0.0, 5.0)

    # Count interval for the graph operator.
    # For N agents, the maximum number of eligible neighbors is N-1.
    count_interval: Tuple[int, int] = (1, 4)

    # Choose a simple geometric predicate over each neighbor state.
    # Robustness is y-coordinate: positive means y >= 0.
    predicate_name: str = "y_nonneg"


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

def build_example_specs(config: TestConfig):
    pred = Predicate(mu=ATOMIC_PREDICATES[config.predicate_name], label=config.predicate_name)

    phi_in = AgentFormula(
        agent_id=config.agent_id,
        child=In(
            graph_types=[config.graph_type],
            W=config.weight_interval,
            E=config.count_interval,
            quantifier="exists",
            child=pred,
        ),
    )

    phi_out = AgentFormula(
        agent_id=config.agent_id,
        child=Out(
            graph_types=[config.graph_type],
            W=config.weight_interval,
            E=config.count_interval,
            quantifier="exists",
            child=pred,
        ),
    )

    return phi_in, phi_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> float:
    config = TestConfig()
    data = load_generated_data(config.data_path)

    positions = data["positions"]  # (T, N, 2)
    T, N, _ = positions.shape

    trajs = {i: positions[:, i, :] for i in range(N)}
    graphs = {
        "dist": data["dist_matrix"],
        "sense": data["G_sense"],
        "comm": data["G_comm"],
    }

    phi_in, phi_out = build_example_specs(config)
    algebra = MinMaxAlgebra()
    aggregator = "min_max"

    t = int(config.time_index)
    agent_id = int(config.agent_id)

    rob_in = evaluate(trajs, graphs, phi_in, algebra, t=t, agent_id=agent_id, aggregator=aggregator)
    rob_out = evaluate(trajs, graphs, phi_out, algebra, t=t, agent_id=agent_id, aggregator=aggregator)

    print(f"Loaded positions shape: {positions.shape}")
    print(f"Agent: {agent_id}, time: {t}")
    print(f"Graph type: {config.graph_type}")
    print(f"Weight interval W: {config.weight_interval}")
    print(f"Count interval E: {config.count_interval}")
    print()
    print(f"In robustness : {rob_in}")
    print(f"Out robustness: {rob_out}")

    return float(rob_in)


if __name__ == "__main__":
    main()
