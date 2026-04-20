#!/usr/bin/env python3
"""Visualize distance, sensing, and communication graphs from generated trajectories."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def visualize_distance_graph(trajectories, graph, time_idx, ax=None):
    """Visualize weighted distance graph at a specific time.

    Args:
        trajectories: (T, N, 3) array with [x, y, theta]
        graph: (T, N, N) distance matrix
        time_idx: which timestep to visualize
        ax: matplotlib axis (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    positions = trajectories[time_idx]  # (N, 3)
    G = graph[time_idx]  # (N, N)
    N = positions.shape[0]

    # Plot nodes
    ax.scatter(positions[:, 0], positions[:, 1], s=100, c='red', zorder=3, label='Agents')

    # Plot edges with alpha based on distance
    max_dist = np.max(G[G > 0]) if np.any(G > 0) else 1.0
    for i in range(N):
        for j in range(i + 1, N):
            if G[i, j] > 0:
                dist = G[i, j]
                alpha = 1.0 - (dist / max_dist) * 0.8  # Closer = darker
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    color='blue', alpha=alpha, linewidth=0.5, zorder=1
                )

    # Annotations
    for i in range(N):
        ax.annotate(f'{i}', (positions[i, 0], positions[i, 1]),
                   fontsize=8, ha='center', va='center', color='white', zorder=4)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Distance Graph (t={time_idx})')

    return ax


def visualize_sensing_graph(trajectories, graph, time_idx, ax=None):
    """Visualize binary sensing graph at a specific time (with FOV).

    Args:
        trajectories: (T, N, 3) array with [x, y, theta]
        graph: (T, N, N) binary adjacency matrix (directed)
        time_idx: which timestep to visualize
        ax: matplotlib axis (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    positions = trajectories[time_idx]  # (N, 3)
    G = graph[time_idx]  # (N, N)
    N = positions.shape[0]

    # Plot nodes
    ax.scatter(positions[:, 0], positions[:, 1], s=100, c='green', zorder=3, label='Agents')

    # Plot directed edges (arrows)
    for i in range(N):
        for j in range(N):
            if i != j and G[i, j] == 1:
                # Draw arrow from i to j
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                ax.arrow(
                    positions[i, 0], positions[i, 1],
                    dx * 0.9, dy * 0.9,
                    head_width=0.3, head_length=0.2,
                    fc='green', ec='green', alpha=0.6, zorder=1, length_includes_head=True
                )

    # Annotations
    for i in range(N):
        ax.annotate(f'{i}', (positions[i, 0], positions[i, 1]),
                   fontsize=8, ha='center', va='center', color='white', zorder=4)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Sensing Graph (t={time_idx}, with FOV)')

    return ax


def visualize_communication_graph(trajectories, graph, time_idx, ax=None):
    """Visualize binary communication graph at a specific time.

    Args:
        trajectories: (T, N, 3) array with [x, y, theta]
        graph: (T, N, N) binary adjacency matrix (undirected)
        time_idx: which timestep to visualize
        ax: matplotlib axis (creates new if None)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    positions = trajectories[time_idx]  # (N, 3)
    G = graph[time_idx]  # (N, N)
    N = positions.shape[0]

    # Plot nodes
    ax.scatter(positions[:, 0], positions[:, 1], s=100, c='purple', zorder=3, label='Agents')

    # Plot edges
    for i in range(N):
        for j in range(i + 1, N):
            if G[i, j] == 1:
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    color='purple', alpha=0.7, linewidth=1.0, zorder=1
                )

    # Annotations
    for i in range(N):
        ax.annotate(f'{i}', (positions[i, 0], positions[i, 1]),
                   fontsize=8, ha='center', va='center', color='white', zorder=4)

    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Communication Graph (t={time_idx})')

    return ax


def main():
    # Load data
    traj_data = np.load("trajectory_data/2D_data/data_0/trajectory.npz")
    trajectories = traj_data["trajectories"]

    graph_data = np.load("trajectory_data/2D_data/data_0/graphs.npz")
    G_dist = graph_data["G_dist"]
    G_sense = graph_data["G_sense"]
    G_comm = graph_data["G_comm"]

    print(f"Trajectories shape: {trajectories.shape}")
    print(f"G_dist shape: {G_dist.shape}")
    print(f"G_sense shape: {G_sense.shape}")
    print(f"G_comm shape: {G_comm.shape}\n")

    time_idx = 0
    print(f"Visualizing graphs at t={time_idx}")

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    visualize_distance_graph(trajectories, G_dist, time_idx, ax=axes[0])
    visualize_sensing_graph(trajectories, G_sense, time_idx, ax=axes[1])
    visualize_communication_graph(trajectories, G_comm, time_idx, ax=axes[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
