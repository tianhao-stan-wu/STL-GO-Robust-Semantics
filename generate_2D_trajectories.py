"""2D agent trajectory simulation.

Simulates N agents moving in 2D space with stochastic dynamics. Each agent samples
random velocities and angles to move. Positions are clipped to stay within bounds.
Computes pairwise distance graphs at each timestep. Saves trajectories and graphs
to file and visualizes results.
"""

from dataclasses import dataclass
from typing import Tuple
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig:
    """Configuration for trajectory simulation."""
    num_agents: int = 5 #100
    time_horizon: int = 10
    dim: int = 2
    x_bounds: Tuple[float, float] = (-10.0, 10.0)
    y_bounds: Tuple[float, float] = (-10.0, 10.0)
    velocity_bounds: Tuple[float, float] = (0.0, 10.0)
    theta_bounds: Tuple[float, float] = (0.0, 2 * np.pi)
    random_seed: int = 100

    save_path: str = "trajectory_data/2D_trajectories.npz"


def generate_graph(positions: np.ndarray) -> np.ndarray:
    """Compute symmetric pairwise distance matrix (Euclidean metric).

    positions: shape (N, dim)
    returns: shape (N, N) distance matrix
    """
    distances = pdist(positions, metric='euclidean')
    return squareform(distances)


def one_step_stochastic_dynamics(
    positions: np.ndarray,
    velocity_min: float,
    velocity_max: float,
    theta_min: float,
    theta_max: float,
    bounds: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Take one stochastic step for N agents.

    positions: shape (N, dim)
    velocity_min, velocity_max: velocity bounds for stochastic sampling
    theta_min, theta_max: angle bounds for stochastic sampling
    bounds: tuple of (x_min, x_max, y_min, y_max) for clipping positions

    returns: (new_positions, distance_graph)
    """
    num_agents = positions.shape[0]
    velocities = velocity_min + (velocity_max - velocity_min) * np.random.rand(num_agents)
    thetas = theta_min + (theta_max - theta_min) * np.random.rand(num_agents)
    displacement = np.column_stack((velocities * np.cos(thetas), velocities * np.sin(thetas)))
    new_positions = positions + displacement

    # Clip positions to stay within bounds
    x_min, x_max, y_min, y_max = bounds
    new_positions[:, 0] = np.clip(new_positions[:, 0], x_min, x_max)
    new_positions[:, 1] = np.clip(new_positions[:, 1], y_min, y_max)

    distance_graph = generate_graph(new_positions)
    return new_positions, distance_graph


def save_results(trajectories: np.ndarray, graphs: np.ndarray, filepath: str) -> None:
    """Save trajectories and graphs to a compressed numpy file.

    trajectories: shape (T, N, dim)
    graphs: shape (T, N, N)
    filepath: output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, trajectories=trajectories, graphs=graphs)
    logger.info(f"Results saved to {filepath}")


def plot_trajectories(trajectories: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    """Plot agent trajectories with boundary rectangle.

    trajectories: shape (T, N, dim)
    x_min, x_max, y_min, y_max: boundary limits
    """
    num_agents = trajectories.shape[1]
    fig, ax = plt.subplots()
    rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, edgecolor="black", linewidth=2, fill=False)
    ax.add_patch(rect)
    ax.set_facecolor('white')

    for nn in range(num_agents):
        color = np.random.rand(3,)
        ax.plot(trajectories[:, nn, 0], trajectories[:, nn, 1], color=color)

    ax.set_xlim((x_min - 1, x_max + 1))
    ax.set_ylim((y_min - 1, y_max + 1))
    ax.set_aspect('equal', adjustable='box')
    plt.title('Agent trajectories')
    plt.tight_layout()
    plt.show()


def main() -> None:
    logger.info("Starting trajectory simulation")
    config = SimulationConfig()
    logger.debug(f"Config: {config}")
    np.random.seed(config.random_seed)

    x_min, x_max = config.x_bounds
    y_min, y_max = config.y_bounds
    velocity_min, velocity_max = config.velocity_bounds
    theta_min, theta_max = config.theta_bounds

    # Initialize positions randomly for all agents within the specified bounds
    initial_positions = np.random.uniform(
        low=[x_min, y_min],
        high=[x_max, y_max],
        size=(config.num_agents, config.dim)
    )
    logger.info(f"Initialized {config.num_agents} agents in 2D space")

    # Initialize trajectories and graphs
    trajectories = np.zeros((config.time_horizon + 1, config.num_agents, config.dim), dtype=float)
    trajectories[0, :, :] = initial_positions
    graphs = np.zeros((config.time_horizon + 1, config.num_agents, config.num_agents), dtype=float)

    # Compute initial graph
    graphs[0, :, :] = generate_graph(initial_positions)

    # Simulate trajectories
    for tt in range(1, config.time_horizon + 1):
        prev_positions = trajectories[tt - 1, :, :]
        current_positions, distance_graph = one_step_stochastic_dynamics(
            prev_positions, velocity_min, velocity_max, theta_min, theta_max,
            bounds=(x_min, x_max, y_min, y_max)
        )
        trajectories[tt, :, :] = current_positions
        graphs[tt, :, :] = distance_graph

    logger.info(f"Completed simulation for {config.time_horizon} timesteps")

    # Save results
    save_results(trajectories, graphs, config.save_path)

    # Plot trajectories
    plot_trajectories(trajectories, x_min, x_max, y_min, y_max)
    logger.info("Simulation complete")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    main()
