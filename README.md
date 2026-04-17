# Multi-Agent Trajectory Simulation Module

Stochastic multi-agent trajectory simulation in 2D and 3D spaces.

## Scripts

### `generate_2D_trajectories.py`
Simulates N agents moving in 2D space with stochastic dynamics.

### `generate_3D_trajectories.py`
Simulates agents in 3D space with two types:
1. **Sphere agents**: Constrained to move on a sphere surface (random angular walks)
2. **Free agents**: Move freely in 3D space with constrained radius changes


## Installation

### Using Conda (Recommended)

Create the environment from the provided `setup-env.yml`:

```bash
conda env create -f setup-env.yml
```

Activate the environment:

```bash
conda activate stlgo-rob-env
```

### Manual Installation

If not using conda, install required packages:

```bash
pip install numpy matplotlib scipy
```

### Requirements
- Python 3.11+
- NumPy 2.4+
- Matplotlib 3.10+
- SciPy 1.17+

## Running the Scripts

### 2D Simulation
```bash
python generate_2D_trajectories.py
```

**Output files:**
- `trajectory_data/2D_trajectories.npz` - Contains trajectories and distance graphs (compressed)
  - `trajectories`: shape (T+1, N, 2) - positions over time
  - `graphs`: shape (T+1, N, N) - pairwise distance matrices
- Displays 2D plot of agent trajectories

**Configuration:** Edit `SimulationConfig` in the script to change:
- `num_agents`: Number of agents (default: 5)
- `time_horizon`: Number of timesteps (default: 10)
- `x_bounds`, `y_bounds`: Position bounds (default: [-10, 10])
- `velocity_bounds`: Velocity range (default: [0, 10])
- `theta_bounds`: Angle range (default: [0, 2 pi])

### 3D Simulation
```bash
python generate_3D_trajectories.py
```

**Output files:**
- `trajectory_data/sphere_trajectories.npz` - Contains trajectories for sphere and free agents
- Displays 3D visualization with sphere boundary

**Configuration:** Edit `SphereAgentConfig` and `FreeAgentConfig` to change parameters

