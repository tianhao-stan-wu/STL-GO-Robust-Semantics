"""3D agent trajectory simulation.

Simulates two types of agents:
- Agents constrained to move on a sphere surface with random angular walks
- Free agents in 3D space with bounded radius and angular changes

Visualizes trajectories in 3D and saves results to file.
"""

from dataclasses import dataclass
from typing import Tuple
import logging
import os
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class SimulationConfig3D:
    """Configuration for trajectory and graphs generation"""
    num_sphere_agents: int = 10
    num_free_agents: int = 5
    time_horizon: int = 50


    sphere_radius: float = 10.0
    angle_change_range: Tuple[float, float] = (-np.pi / 8, np.pi / 8)
    

    free_space_lower_bound: float = 20.0
    free_space_bound: float = 30.0
    free_max_angle_change: float = np.pi/10 #np.pi / 100
    free_max_radius_change: float = 0.5 #0.1

    random_seed: int = 300
    save_path: str = "trajectory_data/sphere_trajectories.npz"


def rand_sample_unit_sphere(n: int) -> np.ndarray:
    """Generate N random unit vectors on a sphere.

    returns: shape (N, 3) array of unit vectors
    """
    X = np.random.randn(n, 3)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X


def cart2sph(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Cartesian to spherical coordinates.

    returns: (phi, theta, r) where phi is azimuth, theta is elevation
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / r)
    return phi, theta, r


def sph2cart(phi: np.ndarray, theta: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert spherical to Cartesian coordinates.

    returns: (x, y, z)
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def simulate_sphere_agents(config: SimulationConfig3D) -> np.ndarray:
    """Simulate agents constrained to move on a sphere surface.

    returns: trajectories shape (T+1, N, 3) - (time, agents, dimensions)
    """
    
    np.random.seed(config.random_seed)

    # Initialize positions on sphere
    init_positions = config.sphere_radius * rand_sample_unit_sphere(config.num_sphere_agents)

    # Convert to spherical angles
    phis, thetas, _ = cart2sph(init_positions[:, 0], init_positions[:, 1], init_positions[:, 2])
    polar_trajectories = np.zeros((config.time_horizon + 1, config.num_sphere_agents, 2))
    polar_trajectories[0, :, 0] = thetas
    polar_trajectories[0, :, 1] = phis

    # Simulate random walk in angles
    angle_min, angle_max = config.angle_change_range
    for tt in range(1, config.time_horizon + 1):
        dthetas = angle_min + (angle_max - angle_min) * np.random.rand(config.num_sphere_agents)
        dphis = angle_min + (angle_max - angle_min) * np.random.rand(config.num_sphere_agents)

        old_angles = polar_trajectories[tt - 1, :, :]
        new_angles = old_angles + np.column_stack((dthetas, dphis))
        polar_trajectories[tt, :, :] = new_angles

    # Convert back to Cartesian
    trajectories = np.zeros((config.time_horizon + 1, config.num_sphere_agents, 3))
    for tt in range(config.time_horizon + 1):
        thetas = polar_trajectories[tt, :, 0]
        phis = polar_trajectories[tt, :, 1]
        x, y, z = sph2cart(phis, thetas, config.sphere_radius)
        trajectories[tt, :, 0] = x
        trajectories[tt, :, 1] = y
        trajectories[tt, :, 2] = z

    return trajectories


def simulate_free_agents(config: SimulationConfig3D) -> np.ndarray:
    """Simulate free agents moving in 3D space with constrained radius changes.

    returns: trajectories shape (T+1, N, 3) - (time, agents, dimensions)
    """
    np.random.seed(config.random_seed)

    # Initialize positions in bounded space with radius in [lower_bound, bound]
    unit_vectors = rand_sample_unit_sphere(config.num_free_agents)
    random_radii = np.random.uniform(config.free_space_lower_bound, config.free_space_bound, config.num_free_agents)
    init_positions = unit_vectors * random_radii[:, np.newaxis]
    trajectories = np.zeros((config.time_horizon + 1, config.num_free_agents, 3))
    trajectories[0, :, :] = init_positions

    # Simulate with constrained radius changes
    for tt in range(1, config.time_horizon + 1):
        old_pos = trajectories[tt - 1, :, :]
        phis, thetas, rs = cart2sph(old_pos[:, 0], old_pos[:, 1], old_pos[:, 2])

        # Random angle changes
        dphis = -config.free_max_angle_change + 2 * config.free_max_angle_change * np.random.rand(config.num_free_agents)
        dthetas = -config.free_max_angle_change + 2 * config.free_max_angle_change * np.random.rand(config.num_free_agents)

        # Random radius changes
        drs = -config.free_max_radius_change + 2 * config.free_max_radius_change * np.random.rand(config.num_free_agents)

        new_phis = phis + dphis
        new_thetas = thetas + dthetas
        new_rs = np.clip(rs + drs, config.free_space_lower_bound, config.free_space_bound)

        x, y, z = sph2cart(new_phis, new_thetas, new_rs)
        trajectories[tt, :, 0] = x
        trajectories[tt, :, 1] = y
        trajectories[tt, :, 2] = z

    return trajectories


def plot_3d_trajectories(sphere_traj: np.ndarray, free_traj: np.ndarray, sphere_radius: float) -> None:
    """Plot 3D trajectories with sphere boundary.

    sphere_traj: shape (T+1, N1, 3) trajectories on sphere
    free_traj: shape (T+1, N2, 3) trajectories in free space
    sphere_radius: radius of constraint sphere
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Plot sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = sphere_radius * np.outer(np.cos(u), np.sin(v))
    y_sphere = sphere_radius * np.outer(np.sin(u), np.sin(v))
    z_sphere = sphere_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.2, color='blue')

    # Plot sphere agent trajectories
    num_sphere_agents = sphere_traj.shape[1]
    for kk in range(num_sphere_agents):
        traj_x = sphere_traj[:, kk, 0]
        traj_y = sphere_traj[:, kk, 1]
        traj_z = sphere_traj[:, kk, 2]

        # Plot start point
        ax.plot([traj_x[0]], [traj_y[0]], [traj_z[0]], marker='s', markersize=10, color='blue')

        # Plot trajectory
        color = np.random.rand(3)
        ax.plot(traj_x, traj_y, traj_z, linestyle=':', marker='.', color=color, linewidth=1.5)

    # Plot free agent trajectories
    num_free_agents = free_traj.shape[1]
    for kk in range(num_free_agents):
        traj_x = free_traj[:, kk, 0]
        traj_y = free_traj[:, kk, 1]
        traj_z = free_traj[:, kk, 2]

        # Plot start point
        ax.plot([traj_x[0]], [traj_y[0]], [traj_z[0]], marker='o', markersize=10,
                markerfacecolor='none', markeredgecolor='red')

        # Plot trajectory
        ax.plot(traj_x, traj_y, traj_z, linestyle=':', marker='.', color='green', linewidth=1.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Agent Trajectories')
    ax.grid(True)
    ax.view_init(elev=20, azim=45)
    plt.tight_layout()
    plt.show()


def save_results(sphere_traj: np.ndarray, free_traj: np.ndarray, filepath: str) -> None:
    """Save both trajectory sets to file.

    sphere_traj: shape (T+1, N1, 3)
    free_traj: shape (T+1, N2, 3)
    filepath: output file path
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    np.savez_compressed(filepath, sphere_trajectories=sphere_traj, free_trajectories=free_traj)
    logger.info(f"Results saved to {filepath}")


def main() -> None:
    logger.info("Starting 3D trajectory simulation")

    simconfig = SimulationConfig3D()
    # sphere_config = SphereAgentConfig()
    # free_config = FreeAgentConfig()

    logger.info(f"Simulating {simconfig.num_sphere_agents} agents on sphere")
    sphere_trajectories = simulate_sphere_agents(simconfig)

    logger.info(f"Simulating {simconfig.num_free_agents} free agents in 3D")
    free_trajectories = simulate_free_agents(simconfig)

    # Save results
    save_results(sphere_trajectories, free_trajectories, simconfig.save_path)

    # Plot results
    plot_3d_trajectories(sphere_trajectories, free_trajectories, simconfig.sphere_radius)

    logger.info("Simulation complete")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    main()
