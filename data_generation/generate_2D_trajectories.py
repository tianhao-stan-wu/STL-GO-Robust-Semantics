"""2D agent trajectory simulation with orientation.

Generates multiple independent 2D agent trajectories. Each trajectory simulates N agents
moving in 2D space (x, y) with orientation (theta) using stochastic Dubins car dynamics. Agents sample
random velocities and angular velocities to move and rotate, with positions clipped to stay
within bounds. Theta wraps around [0, 2π). Saves trajectories to .npz files and automatically
generates distance, sensing, and communication graphs. Displays the first trajectory as a sample.
"""

import logging
import os
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config import SimulationConfig2D
from generate_2D_graphs import generate_graphs_for_trajectory, save_graphs, plot_graph_over_time

logger = logging.getLogger(__name__)


def one_step_stochastic_dynamics(
    positions: np.ndarray,
    velocity_min: float,
    velocity_max: float,
    angular_velocity_min: float,
    angular_velocity_max: float,
    bounds: Tuple[float, float, float, float],
) -> np.ndarray:
    """Take one stochastic step for N agents.

    Args:
        positions: shape (N, 3) with [x, y, theta]
        velocity_min, velocity_max: velocity bounds for stochastic sampling
        angular_velocity_min, angular_velocity_max: angular velocity bounds for dtheta
        bounds: tuple of (x_min, x_max, y_min, y_max) for clipping x,y positions

    Returns:
        shape (N, 3) new agent positions and orientations
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

    return new_positions


def save_results(trajectories: np.ndarray, filepath: str) -> None:
    """Save trajectories to a compressed numpy file.

    Args:
        trajectories: shape (T, N, dim) agent trajectories
        filepath: output file path (.npz format)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, trajectories=trajectories)
    # logger.info(f"Results saved to {filepath}")


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
    plt.show(block=False)


def main() -> None:
    logger.info("Starting 2D trajectory simulation")
    config = SimulationConfig2D()

    print(f"#Agents: {config.num_agents}")
    print(f"Time horizon: {config.time_horizon}")
    print(f"#Trajectories: {config.num_trajectories}\n")
    
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
        
        # Initialize trajectories
        trajectories = np.zeros((config.time_horizon + 1, config.num_agents, config.dim), dtype=float)
        trajectories[0, :, :] = initial_positions

        # Simulate trajectories
        for tt in range(1, config.time_horizon + 1):
            prev_positions = trajectories[tt - 1, :, :]
            current_positions = one_step_stochastic_dynamics(
                prev_positions, velocity_min, velocity_max, angular_velocity_min, angular_velocity_max,
                bounds=(x_min, x_max, y_min, y_max)
            )
            trajectories[tt, :, :] = current_positions

        # Save results with unique filename in folder
        filepath = os.path.join(config.save_path, f"data_{traj_idx}/trajectory.npz")
        save_results(trajectories, filepath)

        # Generate and save graphs
        graph_dict = generate_graphs_for_trajectory(
            trajectory=trajectories,
            sensing_threshold=config.sensing_threshold,
            communication_threshold=config.communication_threshold,
            fov_angle=config.fov_angle
        )

        # Debug: verify shapes
        if traj_idx == 0:
            logger.info(f"Trajectories shape: {trajectories.shape}, Expected: ({config.time_horizon + 1}, {config.num_agents}, {config.dim})")
            print("\nGraphs generated:")
            for graph_name, graph_data in graph_dict.items():
                print(f"  {graph_name}: {graph_data.shape}")
            print("\n" + "="*60)

        graph_filepath = os.path.join(config.save_path, f"data_{traj_idx}/graphs.npz")
        save_graphs(graph_filepath, graph_dict)
        
        # logger.info(f"Saved graphs to: {graph_filepath}")

        # Plot first trajectory as sample
        if traj_idx == 0:
            plot_trajectories(trajectories, x_min, x_max, y_min, y_max)

            # print("\nPlotting Distance Graph over time...")
            # plot_graph_over_time(
            #     trajectories,
            #     graph_dict["G_dist"],
            #     title_prefix="Distance Graph",
            #     weighted=True,
            #     bounds=(x_min, x_max, y_min, y_max)
            # )

            # print("\nPlotting Sensing Graph over time...")
            # plot_graph_over_time(
            #     trajectories,
            #     graph_dict["G_sense"],
            #     title_prefix="Sensing Graph",
            #     weighted=False,
            #     bounds=(x_min, x_max, y_min, y_max)
            # )

            # print("\nPlotting Communication Graph over time...")
            # plot_graph_over_time(
            #     trajectories,
            #     graph_dict["G_comm"],
            #     title_prefix="Communication Graph",
            #     weighted=False,
            #     bounds=(x_min, x_max, y_min, y_max)
            # )

        # Log progress
        if (traj_idx + 1) % 10 == 0 or traj_idx == config.num_trajectories - 1:
            logger.info(f"Generated {traj_idx + 1}/{config.num_trajectories} trajectories")

    logger.info("All simulations complete")
    print(f"Data saved to: {config.save_path}")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    main()
