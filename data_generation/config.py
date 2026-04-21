"""Configuration for 2D trajectory simulation."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class SimulationConfig2D:
    """Configuration for trajectory simulation."""
    num_agents: int = 100
    time_horizon: int = 50
    dim: int = 3  # x, y, theta

    x_bounds: Tuple[float, float] = (-50.0, 50.0)
    y_bounds: Tuple[float, float] = (-50.0, 50.0)
    theta_bounds: Tuple[float, float] = (0.0, 2 * np.pi)

    velocity_bounds: Tuple[float, float] = (0.0, 10.0)
    angular_velocity_bounds: Tuple[float, float] = (-np.pi / 4, np.pi / 4)

    sensing_threshold: float = 5.0
    communication_threshold: float = 7.0
    fov_angle: float = np.pi / 4  # ±45° field of view

    random_seed: int = 100
    num_trajectories: int = 1000
    save_path: str = "trajectory_data/2D_data"
