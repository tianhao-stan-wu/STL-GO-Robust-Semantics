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
from algebra import MinMaxAlgebra, BooleanAlgebra
from graph_ops import get_neighbors
from evaluator import evaluate


algebras = {"minmax": MinMaxAlgebra(),
            "bool": BooleanAlgebra()
}

aggregators = {"minmax": "min_max",
                "count": "counting",
                "avg": "averaging",
                "hybrid": "hybrid",
                "bool": "boolean"
}


@dataclass
class TestConfig:
    traj_path: str = "../trajectory_data/2D_data/data_0/trajectory.npz"
    graph_path: str = "../trajectory_data/2D_data/data_0/graphs.npz"
    agent_id: int = 26
    time_index: int = 0

    # Use the distance graph for the simplest possible test.
    graph_type: str = "dist"

    # For weighted distance graph: keep edges whose distance is in this interval.
    # Since the graph is undirected, In and Out will be equivalent.
    weight_interval: Tuple[float, float] = (0.0, 10.0) #(0.0, 5.0)

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

def load_generated_data(graph_path: str, traj_path: str) -> Dict[str, np.ndarray]:
    """Load trajectories and graphs from separate .npz files.

    Args:
        graph_path: path to graphs.npz with keys 'G_dist', 'G_sense', 'G_comm'
        traj_path: path to trajectory.npz with key 'trajectories' (T, N, 3)

    Returns:
        Dictionary with keys: trajectories, G_dist, G_sense, G_comm
    """
    traj_data = np.load(traj_path, allow_pickle=False)
    if "trajectories" not in traj_data.keys():
        raise KeyError(f"Missing 'trajectories' key in {traj_path}")
    trajectories = traj_data["trajectories"]

    graph_data = np.load(graph_path, allow_pickle=False)
    required = {"G_dist", "G_sense", "G_comm"}
    missing = required.difference(graph_data.keys())
    if missing:
        raise KeyError(f"Missing keys in {graph_path}: {sorted(missing)}")

    return {
        "trajectories": trajectories,
        "G_dist": graph_data["G_dist"],
        "G_sense": graph_data["G_sense"],
        "G_comm": graph_data["G_comm"],
    }


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

    print(phi_in)
    return phi_in, phi_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = TestConfig()
    data = load_generated_data(config.graph_path, config.traj_path)

    trajectories = data["trajectories"]  # (T, N, 3) with [x, y, theta]
    T, N, _ = trajectories.shape

    trajs = {i: trajectories[:, i, :] for i in range(N)}
    graphs = {
        "dist": data["G_dist"],
        "sense": data["G_sense"],
        "comm": data["G_comm"],
    }

    phi_in, phi_out = build_example_specs(config)
    algebra = algebras["minmax"]
    aggregator = aggregators["hybrid"]

    t = int(config.time_index)
    agent_id = int(config.agent_id)

    rob_in = evaluate(trajs, graphs, phi_in, algebra, t=t, agent_id=agent_id, aggregator=aggregator)
    rob_out = evaluate(trajs, graphs, phi_out, algebra, t=t, agent_id=agent_id, aggregator=aggregator)

    print(f"Agent: {agent_id}, time: {t}")
    print(f"Graph type: {config.graph_type}")
    print(f"Weight interval W: {config.weight_interval}")
    print(f"Count interval E: {config.count_interval}")
    print()
    print(f"In robustness : {rob_in}")
    print(f"Out robustness: {rob_out}")


if __name__ == "__main__":
    main()


