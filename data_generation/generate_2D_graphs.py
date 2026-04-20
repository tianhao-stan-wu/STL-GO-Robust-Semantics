#!/usr/bin/env python3
"""
Generate distance, sensing, and communication graphs from 2D trajectory data with orientation.

This module processes agent trajectories (position + orientation) and generates three types of
multi-agent interaction graphs at each timestep:

Graph types:
  - Distance graph (G_dist): Weighted adjacency matrix with Euclidean distances (x,y only)
  - Sensing graph (G_sense): Binary adjacency with distance AND field-of-view constraints
  - Communication graph (G_comm): Binary adjacency with distance threshold only

Expected input:
  - trajectories array with shape (T, N, 3) where:
      T = timesteps
      N = number of agents
      3 = [x, y, theta] state

Output files (.npz format) contain:
  - G_dist: (T, N, N) weighted distance graphs
  - G_sense: (T, N, N) binary sensing graphs (with FOV)
  - G_comm: (T, N, N) binary communication graphs

Note: This module only generates graphs. It does not evaluate STL-GO formulas or
compute robustness metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np


def pairwise_distance_matrix(positions: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise Euclidean distance matrix (using x,y only).

    positions: shape (N, 3) with [x, y, theta]
    returns:   shape (N, N)
    """
    xy_positions = positions[:, :2]
    diff = xy_positions[:, None, :] - xy_positions[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def build_communication_graph(dist_matrix: np.ndarray, threshold: float) -> np.ndarray:
    """
    Create binary communication graph based on distance threshold.

    Args:
        dist_matrix: shape (N, N) pairwise distance matrix
        threshold: distance threshold for connectivity

    Returns:
        shape (N, N) binary adjacency matrix (1 if distance <= threshold)
    """
    adj = (dist_matrix <= threshold).astype(np.int8)
    np.fill_diagonal(adj, 0)
    return adj

def build_sensing_graph(positions: np.ndarray, dist_matrix: np.ndarray, threshold: float, fov_angle: float) -> np.ndarray:
    """
    Create binary sensing graph with distance and field-of-view constraints.

    Edge (i, j) exists iff:
      - dist(i, j) <= threshold AND
      - agent j is within agent i's field of view

    Args:
        positions: shape (N, 3) with [x, y, theta]
        dist_matrix: shape (N, N) pairwise distances
        threshold: distance threshold for sensing
        fov_angle: field of view half-angle in radians

    Returns:
        shape (N, N) binary adjacency matrix
    """
    N = positions.shape[0]
    adj = np.zeros((N, N), dtype=np.int8)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            if dist_matrix[i, j] > threshold:
                continue

            # Get agent i's orientation and position
            theta_i = positions[i, 2]
            x_i, y_i = positions[i, 0], positions[i, 1]
            x_j, y_j = positions[j, 0], positions[j, 1]

            # Compute angle from i to j
            angle_to_j = np.arctan2(y_j - y_i, x_j - x_i)

            # Compute angular difference (handle wrapping)
            angle_diff = angle_to_j - theta_i
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

            # Check if j is within agent i's field of view
            if abs(angle_diff) <= fov_angle:
                adj[i, j] = 1

    return adj


def generate_graphs_for_trajectory(
    trajectory: np.ndarray,
    sensing_threshold: float,
    communication_threshold: float,
    fov_angle: float,
) -> Dict[str, np.ndarray]:
    """
    Generate distance, sensing, and communication graphs for all time steps.

    Args:
        trajectory: shape (T, N, 3) with [x, y, theta]
        sensing_threshold: distance threshold for sensing graph
        communication_threshold: distance threshold for communication graph
        fov_angle: field of view half-angle in radians for sensing graph

    Returns:
        Dictionary with keys:
            - G_dist: (T, N, N) weighted distance graphs
            - G_sense: (T, N, N) binary sensing graphs (with FOV)
            - G_comm: (T, N, N) binary communication graphs
    """
    if trajectory.ndim != 3 or trajectory.shape[2] != 3:
        raise ValueError(f"Expected trajectory shape (T, N, 3) with [x, y, theta], got {trajectory.shape}")

    Tplus1, N, _ = trajectory.shape

    g_dist = np.zeros((Tplus1, N, N), dtype=float)
    g_sense = np.zeros((Tplus1, N, N), dtype=np.int8)
    g_comm = np.zeros((Tplus1, N, N), dtype=np.int8)

    for t in range(Tplus1):
        positions_t = trajectory[t]
        dist_t = pairwise_distance_matrix(positions_t)

        g_dist[t] = dist_t
        g_sense[t] = build_sensing_graph(positions_t, dist_t, sensing_threshold, fov_angle)
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
    bounds: Tuple[float, float, float, float],
) -> None:
    """
    Visualize graph sequence over time with consistent agent colors.

    Args:
        positions: shape (T, N, 3) with [x, y, theta]
        graph_sequence: shape (T, N, N) adjacency matrices
        title_prefix: title for plot frames
        weighted: if True, draw weighted edges; if False, draw binary edges
        bounds: tuple (x_min, x_max, y_min, y_max) for plot limits
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

        # plot edges (handle both directed and undirected graphs)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
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
        plt.xlim(bounds[0] - 1, bounds[1] + 1)
        plt.ylim(bounds[2] - 1, bounds[3] + 1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)

        plt.pause(0.5)
        plt.clf()

    plt.close('all')
