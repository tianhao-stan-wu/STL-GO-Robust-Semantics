"""2D agent trajectory simulation with orientation.

Generates multiple independent 2D agent trajectories. Each trajectory simulates N agents
moving in 2D space (x, y) with orientation (theta) using stochastic Dubins car dynamics. Agents sample
random velocities and angular velocities to move and rotate, with positions clipped to stay
within bounds. Theta wraps around [0, 2π). Computes pairwise distance graphs (using x,y only)
at each timestep. Saves each trajectory and graph to separate files in a folder, and
visualizes the first trajectory as a sample.
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
    num_agents: int = 30 #100
    time_horizon: int = 50
    dim: int = 3 # x, y, theta
    x_bounds: Tuple[float, float] = (-25.0, 25.0)
    y_bounds: Tuple[float, float] = (-25.0, 25.0)
    theta_bounds: Tuple[float, float] = (0.0, 2 * np.pi)
    velocity_bounds: Tuple[float, float] = (0.0, 10.0)
    angular_velocity_bounds: Tuple[float, float] = (-np.pi / 4, np.pi / 4)
    random_seed: int = 100
    num_trajectories: int = 1

    save_path: str = "trajectory_data/2D_trajectories"


def generate_graph(positions: np.ndarray) -> np.ndarray:
    """Compute symmetric pairwise distance matrix (Euclidean metric).

    positions: shape (N, dim) - uses only x,y coordinates
    returns: shape (N, N) distance matrix
    """
    xy_positions = positions[:, :2]
    distances = pdist(xy_positions, metric='euclidean')
    return squareform(distances)


def one_step_stochastic_dynamics(
    positions: np.ndarray,
    velocity_min: float,
    velocity_max: float,
    angular_velocity_min: float,
    angular_velocity_max: float,
    bounds: Tuple[float, float, float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Take one stochastic step for N agents.

    positions: shape (N, 3) with [x, y, theta]
    velocity_min, velocity_max: velocity bounds for stochastic sampling
    angular_velocity_min, angular_velocity_max: angular velocity bounds for dtheta
    bounds: tuple of (x_min, x_max, y_min, y_max) for clipping x,y positions

    returns: (new_positions, distance_graph)
    """
    num_agents = positions.shape[0]
    velocities = velocity_min + (velocity_max - velocity_min) * np.random.rand(num_agents)
    angular_velocities = angular_velocity_min + (angular_velocity_max - angular_velocity_min) * np.random.rand(num_agents)

    old_thetas = positions[:, 2]
    new_thetas = (old_thetas + angular_velocities) % (2 * np.pi) 

    displacement = np.column_stack((velocities * np.cos(new_thetas), velocities * np.sin(new_thetas), np.zeros(num_agents)))
    new_positions = positions + displacement

    # Clip x,y positions to stay within bounds, wrap theta
    x_min, x_max, y_min, y_max = bounds
    new_positions[:, 0] = np.clip(new_positions[:, 0], x_min, x_max)
    new_positions[:, 1] = np.clip(new_positions[:, 1], y_min, y_max)
    new_positions[:, 2] = new_thetas

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
    
    x_min, x_max = config.x_bounds
    y_min, y_max = config.y_bounds
    theta_min, theta_max = config.theta_bounds
    velocity_min, velocity_max = config.velocity_bounds
    angular_velocity_min, angular_velocity_max = config.angular_velocity_bounds

    # Generate multiple trajectories
    for traj_idx in range(config.num_trajectories):
        np.random.seed(config.random_seed + traj_idx)

        # Initialize positions randomly for all agents within the specified bounds
        initial_positions = np.random.uniform(
            low=[x_min, y_min, theta_min],
            high=[x_max, y_max, theta_max],
            size=(config.num_agents, config.dim)
        )
        # logger.info(f"Initialized {config.num_agents} agents in 2D space")
        
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
                prev_positions, velocity_min, velocity_max, angular_velocity_min, angular_velocity_max,
                bounds=(x_min, x_max, y_min, y_max)
            )
            trajectories[tt, :, :] = current_positions
            graphs[tt, :, :] = distance_graph

        # Save results with unique filename in folder
        filepath = os.path.join(config.save_path, f"trajectory_{traj_idx:03d}.npz")
        save_results(trajectories, graphs, filepath)

        # Plot first trajectory as sample
        if traj_idx == 0:
            plot_trajectories(trajectories, x_min, x_max, y_min, y_max)

        # Log progress
        if (traj_idx + 1) % 10 == 0 or traj_idx == config.num_trajectories - 1:
            logger.info(f"Generated {traj_idx + 1}/{config.num_trajectories} trajectories")

    logger.info("All simulations complete")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    main()
