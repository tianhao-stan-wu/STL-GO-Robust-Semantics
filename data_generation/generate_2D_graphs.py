#!/usr/bin/env python3
"""
Generate distance, sensing, and communication graphs from saved 2D trajectories.

Expected input:
  - A .npz file containing an array named `trajectories` with shape (T, N, 2)

Output:
  - A .npz file containing:
      positions[t]
      dist_matrix[t]
      G_dist[t]
      G_sense[t]
      G_comm[t]

Graph conventions:
  - Distance graph: weighted adjacency matrix. Entry (i, j) is the Euclidean
    distance if i != j, else 0.0.
  - Sensing graph: unweighted adjacency matrix with 1 if distance <= threshold.
  - Communication graph: unweighted adjacency matrix with 1 if distance <= threshold.

This script only generates the graphs. It does not define STL-GO formulas or
perform robustness monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class GraphConfig:
    """Configuration for graph generation from a single 2D trajectory."""
    input_path: str = "trajectory_data/2D_trajectories.npz"
    output_path: str = "trajectory_data/2D_graphs.npz"
    trajectory_key: str = "trajectories"
    trajectory_index: int = 0
    sensing_threshold: float = 5.0
    communication_threshold: float = 7.0


def load_trajectory(config: GraphConfig) -> np.ndarray:
    """
    Load trajectories from a .npz file and select one trajectory.

    Assumes the saved array shape is fixed:
      - (T, N, 2) for a single trajectory
      - (M, T, N, 2) for a batch of trajectories
    """
    data = np.load(config.input_path, allow_pickle=False)
    trajectories = data[config.trajectory_key]

    if trajectories.ndim == 3:
        return trajectories

    return trajectories[config.trajectory_index]


def pairwise_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise Euclidean distance matrix.

    positions: shape (N, 2)
    returns:   shape (N, N)
    """
    diff = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def weighted_distance_graph(dist_matrix: np.ndarray) -> np.ndarray:
    """
    Distance graph as a weighted adjacency matrix:
      - off-diagonal entry (i, j) = Euclidean distance
      - diagonal = 0.0
    """
    g = dist_matrix.copy()
    np.fill_diagonal(g, 0.0)
    return g


def threshold_graph(dist_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Build an unweighted adjacency matrix:
      edge (i, j) exists iff i != j and dist(i, j) <= threshold
    """
    adj = (dist_matrix <= threshold).astype(np.int8)
    np.fill_diagonal(adj, 0)
    return adj

def build_sensing_graph(dist_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Build an unweighted adjacency matrix:
      edge (i, j) exists iff i != j and dist(i, j) <= threshold
    """
    adj = (dist_matrix <= threshold).astype(np.int8)
    np.fill_diagonal(adj, 0)
    return adj


def generate_graphs_for_trajectory(
    trajectory: np.ndarray,
    sensing_threshold: float,
    communication_threshold: float,
) -> Dict[str, np.ndarray]:
    """
    Generate graphs for each time step.

    trajectory: shape (T, N, 2)
    returns:
      positions:   (T, N, 2)
      dist_matrix:  (T, N, N)
      G_dist:      (T, N, N)
      G_sense:     (T, N, N)
      G_comm:      (T, N, N)
    """
    T, N, _ = trajectory.shape

    dist_matrices = np.zeros((T, N, N), dtype=float)
    g_dist = np.zeros((T, N, N), dtype=float)
    g_sense = np.zeros((T, N, N), dtype=np.int8)
    g_comm = np.zeros((T, N, N), dtype=np.int8)

    for t in range(T):
        positions_t = trajectory[t]
        dist_t = pairwise_distance_matrix(positions_t)

        dist_matrices[t] = dist_t
        g_dist[t] = weighted_distance_graph(dist_t)
        g_sense[t] = threshold_graph(dist_t, sensing_threshold)
        g_comm[t] = threshold_graph(dist_t, communication_threshold)

    return {
        "positions": trajectory,
        "dist_matrix": dist_matrices,
        "G_dist": g_dist,
        "G_sense": g_sense,
        "G_comm": g_comm,
    }


def save_graphs(output_path: str, graphs: Dict[str, np.ndarray]) -> None:
    """Save graph arrays to a compressed .npz file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, **graphs)


def plot_graph_over_time(positions, graph_sequence, title_prefix, weighted=False):
    """
    Plot a graph sequence over time with consistent colors per agent.

    positions: (T, N, 2)
    graph_sequence: (T, N, N)
    weighted: if True, draw edges for distance graph
    """
    T, N, _ = positions.shape

    # assign a fixed color per agent
    colors = np.random.rand(N, 3)

    for t in range(T):
        plt.figure(figsize=(6, 6))
        pos = positions[t]
        G = graph_sequence[t]

        # plot nodes with fixed colors
        for i in range(N):
            plt.scatter(pos[i, 0], pos[i, 1], color=colors[i], s=40)

        # plot edges
        for i in range(N):
            for j in range(i + 1, N):
                if weighted:
                    if G[i, j] > 0:
                        plt.plot(
                            [pos[i, 0], pos[j, 0]],
                            [pos[i, 1], pos[j, 1]],
                            linewidth=0.5,
                            color="gray",
                            alpha=0.5
                        )
                else:
                    if G[i, j] == 1:
                        plt.plot(
                            [pos[i, 0], pos[j, 0]],
                            [pos[i, 1], pos[j, 1]],
                            linewidth=1.0,
                            color="black",
                            alpha=0.7
                        )

        plt.title(f"{title_prefix} (t={t})")
        plt.xlim(-11, 11)
        plt.ylim(-11, 11)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)

        plt.pause(0.5)
        plt.clf()

    plt.close()


def main() -> None:
    config = GraphConfig()

    trajectory = load_trajectory(config)
    graphs = generate_graphs_for_trajectory(
        trajectory=trajectory,
        sensing_threshold=config.sensing_threshold,
        communication_threshold=config.communication_threshold,
    )
    save_graphs(config.output_path, graphs)

    print(f"Loaded trajectory shape: {trajectory.shape}")
    print(f"Saved graphs to: {config.output_path}")
    
    print("\nPlotting Distance Graph over time...")
    plot_graph_over_time(
        graphs["positions"],
        graphs["G_dist"],
        title_prefix="Distance Graph",
        weighted=True
    )

    print("\nPlotting Sensing Graph over time...")
    plot_graph_over_time(
        graphs["positions"],
        graphs["G_sense"],
        title_prefix="Sensing Graph",
        weighted=False
    )

    print("\nPlotting Communication Graph over time...")
    plot_graph_over_time(
        graphs["positions"],
        graphs["G_comm"],
        title_prefix="Communication Graph",
        weighted=False
    )    
    


if __name__ == "__main__":
    main()
