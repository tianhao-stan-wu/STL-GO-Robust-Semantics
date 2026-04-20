#!/usr/bin/env python3
"""Monitoring system: evaluate STL-GO formulas on trajectories and compute robustness."""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict

# Add stl-go directory to path
stl_go_path = Path(__file__).parent / "stl-go"
sys.path.insert(0, str(stl_go_path))

from evaluator import evaluate
from algebra import MinMaxAlgebra, BooleanAlgebra
from syntax import Formula, Predicate, AgentFormula, In, Out

# Add specifications directory to path
spec_path = Path(__file__).parent / "specifications"
sys.path.insert(0, str(spec_path))

from specifications.spec1 import build_spec

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(traj_path: str, graph_path: str) -> Dict[str, np.ndarray]:
    """Load trajectories and graphs from separate .npz files.

    Args:
        traj_path: path to trajectory.npz with key 'trajectories' (T, N, 3)
        graph_path: path to graphs.npz with keys 'G_dist', 'G_sense', 'G_comm'

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





def compute_robustness(
    trajectories: np.ndarray,
    graphs: dict,
    formula: Formula,
    agent_id: int,
    time: int,
    algebra,
    aggregator: str
) -> float:
    """
    Compute robustness of an STL-GO formula for a specific agent at a specific time.

    Args:
        trajectories: shape (T, N, 3) with [x, y, theta] for all agents over time
        graphs: dictionary with keys "G_dist", "G_sense", "G_comm", each shape (T, N, N)
        formula: STL-GO Formula object to evaluate
        agent_id: agent index (0 to N-1)
        time: time index (0 to T-1)
        algebra: Algebra for robustness computation (default: MinMaxAlgebra)
        aggregator: aggregation method for graph operators: "min_max", "counting", "averaging"

    Returns:
        Robustness value (float)

    Supports: temporal formulas (Always, Eventually, Until, And, Neg)
              and graph operators (In, Out) with specified aggregator
    """

    # if agent_id < 0 or agent_id >= trajectories.shape[1]:
    #     raise ValueError(f"Invalid agent_id {agent_id}, must be in [0, {trajectories.shape[1]-1}]")

    # if time < 0 or time >= trajectories.shape[0]:
    #     raise ValueError(f"Invalid time {time}, must be in [0, {trajectories.shape[0]-1}]")

    N = trajectories.shape[1]
    trajs = {i: trajectories[:, i, :] for i in range(N)}

    robustness = evaluate(trajs, graphs, formula, algebra, t=time, agent_id=agent_id, aggregator=aggregator)

    return robustness




algebras = {"minmax": MinMaxAlgebra(),
            "bool": BooleanAlgebra()
}

aggregators = {"minmax": "min_max",
                "count": "counting",
                "avg": "averaging",
                "hybrid": "hybrid",
                "bool": "boolean"
}


if __name__ == "__main__":
    data_path = Path("trajectory_data/2D_data/")
    
    algebra = algebras["minmax"]
    aggregator = aggregators["minmax"]
    agent_id: int = 25
    time_index: int = 0
    
    results = []

    # Load data
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    data_dirs = sorted([d for d in data_path.iterdir() if d.is_dir() and d.name.startswith("data_")])
    
    start_time = time.time()
    for data_dir in data_dirs:
        traj_path: str = str(data_dir / "trajectory.npz")
        graph_path: str = str(data_dir / "graphs.npz")
        data = load_data(traj_path, graph_path)
        # print(f"✓ Loaded trajectories: {data['trajectories'].shape}")
        # print(f"✓ Loaded graphs: {list(data.keys())}\n")

        trajectories = data["trajectories"]
        graphs = {
            "dist": data["G_dist"],
            "sense": data["G_sense"],
            "comm": data["G_comm"],
        }

        # Build spec and evaluate robustness
        phi = build_spec(agent_id)
        

        robustness = compute_robustness(trajectories, graphs, phi, agent_id, time_index, algebra, aggregator=aggregator)
        results.append({"data_dir": data_dir.name, "robustness": robustness})

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print summary
    print("="*60)
    print("ROBUSTNESS SUMMARY")
    print("="*60)
    for result in results:
        print(f"{result['data_dir']}: {result['robustness']:.4f}")

    # Count satisfying and violating traces
    satisfying = sum(1 for r in results if r['robustness'] >= 0)
    violating = sum(1 for r in results if r['robustness'] < 0)

    print("\n" + "="*60)
    print("SPECIFICATION STATUS")
    print("="*60)
    print(f"Total traces: {len(results)}")
    print(f"Satisfying: {satisfying}")
    print(f"Violating: {violating}")

    print("\n" + "="*60)
    print("COMPUTATION TIME")
    print("="*60)
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Data path: {data_path}")
    print(f"Agent ID: {agent_id}")
    print(f"Time index: {time_index}")
    print(f"Algebra: minmax")
    print(f"Aggregator: {aggregator}")
    print(f"Total traces processed: {len(results)}")
    print(f"Success rate: {satisfying}/{len(results)} ({100*satisfying/len(results):.1f}%)")
    print(f"Average robustness: {np.mean([r['robustness'] for r in results]):.4f}")
    print(f"Min robustness: {min(r['robustness'] for r in results):.4f}")
    print(f"Max robustness: {max(r['robustness'] for r in results):.4f}")

