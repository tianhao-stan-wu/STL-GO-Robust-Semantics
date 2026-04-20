#!/usr/bin/env python3
"""
Generate graphs from 3D trajectory data.

This module processes 3D agent trajectories and generates multi-agent interaction graphs
at each timestep. Graph types (to be defined):
  - Distance graph (G_dist): Weighted adjacency matrix with Euclidean distances
  - Sensing graph (G_sense): TBD
  - Communication graph (G_comm): TBD

Expected input:
  - trajectories array with shape (T, N, 3) where:
      T = timesteps
      N = number of agents
      3 = [x, y, z] state

Output files (.npz format) contain:
  - G_dist: (T, N, N) weighted distance graphs
  - G_sense: (T, N, N) graph (to be defined)
  - G_comm: (T, N, N) graph (to be defined)

Note: Graph definitions will be implemented based on 3D interaction requirements.
"""

from __future__ import annotations

import os
from typing import Dict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def pairwise_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise Euclidean distance matrix in 3D.

    Args:
        positions: shape (N, 3) with [x, y, z]

    Returns:
        shape (N, N) symmetric distance matrix
    """
    # TODO: Compute 3D pairwise distances
    pass

def geodesic_distance_matrix(positions: np.ndarray, sphere_radius: float) -> np.ndarray:
    """
    Compute geodesic distance matrix on sphere surface.

    For two points on a sphere: d_geodesic = R * arccos(dot(p1, p2) / R^2)

    Args:
        positions: shape (N, 3) with [x, y, z]
        sphere_radius: radius of the sphere

    Returns:
        shape (N, N) symmetric geodesic distance matrix
    """
    N = positions.shape[0]
    dist_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                dist_matrix[i, j] = 0
            else:
                dot_product = np.dot(positions[i], positions[j])
                cos_angle = dot_product / (sphere_radius * sphere_radius)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                dist_matrix[i, j] = sphere_radius * angle

    return dist_matrix


def free_agents_distance(positions: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix for free agents.

    Args:
        positions: shape (N, 3) with [x, y, z]

    Returns:
        shape (N, N) symmetric distance matrix
    """
    N = positions.shape[0]
    dist_matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i != j:
                dist_matrix[i, j] = np.linalg.norm(positions[i] - positions[j])

    return dist_matrix
    

def build_communication_graph(dist_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Create binary communication graph based on distance threshold.

    Args:
        dist_matrix: shape (N, N) pairwise distance matrix
        threshold: distance threshold for connectivity

    Returns:
        shape (N, N) binary adjacency matrix
    """
    # TODO: Define communication graph for 3D
    pass


def build_sensing_graph(positions: np.ndarray, dist_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Create sensing graph with distance and/or spatial constraints.

    Args:
        positions: shape (N, 3) with [x, y, z]
        dist_matrix: shape (N, N) pairwise distances
        threshold: distance threshold

    Returns:
        shape (N, N) binary adjacency matrix
    """
    # TODO: Define sensing graph for 3D (e.g., angular constraints in spherical coords)
    pass


def generate_graphs_for_trajectory(
    trajectory: np.ndarray,
    sensing_threshold: float,
    communication_threshold: float,
) -> Dict[str, np.ndarray]:
    """
    Generate distance, sensing, and communication graphs for all timesteps.

    Args:
        trajectory: shape (T, N, 3) with [x, y, z]
        sensing_threshold: distance threshold for sensing graph
        communication_threshold: distance threshold for communication graph

    Returns:
        Dictionary with keys:
            - G_dist: (T, N, N) weighted distance graphs
            - G_sense: (T, N, N) binary sensing graphs
            - G_comm: (T, N, N) binary communication graphs
    """
    if trajectory.ndim != 3 or trajectory.shape[2] != 3:
        raise ValueError(f"Expected trajectory shape (T, N, 3) with [x, y, z], got {trajectory.shape}")

    Tplus1, N, _ = trajectory.shape

    g_dist = np.zeros((Tplus1, N, N), dtype=float)
    g_sense = np.zeros((Tplus1, N, N), dtype=np.int8)
    g_comm = np.zeros((Tplus1, N, N), dtype=np.int8)

    for t in range(Tplus1):
        positions_t = trajectory[t]
        dist_t = pairwise_distance_matrix(positions_t)

        g_dist[t] = dist_t
        g_sense[t] = build_sensing_graph(positions_t, dist_t, sensing_threshold)
        g_comm[t] = build_communication_graph(dist_t, communication_threshold)

    return {
        "G_dist": g_dist,
        "G_sense": g_sense,
        "G_comm": g_comm,
    }


def save_graphs(output_path: str, graphs: Dict[str, np.ndarray]) -> None:
    """
    Save graph data to compressed .npz file.

    Args:
        output_path: path to save the .npz file
        graphs: dictionary of graph arrays to save
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez_compressed(output_path, **graphs)


def plot_graph_over_time(
    positions: np.ndarray,
    graph_sequence: np.ndarray,
    title_prefix: str,
    weighted: bool,
) -> None:
    """
    Visualize graph sequence over time in 3D with consistent agent colors.

    Args:
        positions: shape (T, N, 3) with [x, y, z]
        graph_sequence: shape (T, N, N) adjacency matrices
        title_prefix: title for plot frames
        weighted: if True, draw weighted edges; if False, draw binary edges
    """
    T, N, _ = positions.shape

    colors = np.random.rand(N, 3)

    for t in range(T):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        pos = positions[t]
        G = graph_sequence[t]

        # Plot nodes with fixed colors
        for i in range(N):
            ax.scatter(pos[i, 0], pos[i, 1], pos[i, 2], color=colors[i], s=40)

        # Plot edges
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if weighted:
                    if G[i, j] > 0:
                        ax.plot(
                            [pos[i, 0], pos[j, 0]],
                            [pos[i, 1], pos[j, 1]],
                            [pos[i, 2], pos[j, 2]],
                            linewidth=0.5,
                            color="gray",
                            alpha=0.5
                        )
                else:
                    if G[i, j] == 1:
                        ax.plot(
                            [pos[i, 0], pos[j, 0]],
                            [pos[i, 1], pos[j, 1]],
                            [pos[i, 2], pos[j, 2]],
                            linewidth=1.0,
                            color="black",
                            alpha=0.7
                        )

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"{title_prefix} (t={t})")
        plt.pause(0.5)
        plt.clf()

    plt.close('all')


def main() -> None:
    """Load 3D trajectories and generate graphs."""
    # Load trajectory data from generate_3D_trajectories output
    trajectory_file = "trajectory_data/sphere_trajectories.npz"
    if not os.path.exists(trajectory_file):
        raise FileNotFoundError(f"Trajectory file not found: {trajectory_file}")

    data = np.load(trajectory_file)
    sphere_trajectories = data['sphere_trajectories']
    free_trajectories = data['free_trajectories']

    # Define threshold parameters
    sensing_threshold = 15.0
    communication_threshold = 20.0

    # Generate graphs for sphere agents
    print("Generating graphs for sphere agents...")
    sphere_graphs = generate_graphs_for_trajectory(
        sphere_trajectories,
        sensing_threshold=sensing_threshold,
        communication_threshold=communication_threshold,
    )

    # Generate graphs for free agents
    print("Generating graphs for free agents...")
    free_graphs = generate_graphs_for_trajectory(
        free_trajectories,
        sensing_threshold=sensing_threshold,
        communication_threshold=communication_threshold,
    )

    # Save results
    sphere_output = "graph_data/sphere_graphs.npz"
    free_output = "graph_data/free_graphs.npz"

    print(f"Saving sphere graphs to {sphere_output}")
    save_graphs(sphere_output, sphere_graphs)

    print(f"Saving free agent graphs to {free_output}")
    save_graphs(free_output, free_graphs)


if __name__ == '__main__':
    main()
