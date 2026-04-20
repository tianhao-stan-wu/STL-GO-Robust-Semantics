
import sys
from pathlib import Path
import numpy as np

# Add stl-go directory to path
stl_go_path = Path(__file__).parent.parent / "stl-go"
sys.path.insert(0, str(stl_go_path))

from syntax import Predicate, AgentFormula, In, Out, Eventually, Always

# Define goal region (circle)
GOAL_CENTER = (5, 5)
GOAL_RADIUS = 5.0


def distance_to_goal(position, goal_center, goal_radius):
    """Compute distance from position to goal boundary.

    Positive = inside goal, Negative = outside goal
    """
    dist_to_center = np.linalg.norm(position[:2] - np.array(goal_center))
    return goal_radius - dist_to_center


def y_nonneg(state: np.ndarray) -> float:
    """Robustness of y >= 0."""
    return float(state[1])


ATOMIC_PREDICATES = {
    "y_nonneg": y_nonneg,
    "goal": lambda state: distance_to_goal(state, GOAL_CENTER, GOAL_RADIUS),
}

def build_spec(agent_id: int) -> AgentFormula:
    """Build STL-GO specification for an agent."""
    graph_type = "dist"
    weight_interval = (0.0, 20.0)
    count_interval = (1, 4)
    predicate_name = "goal"

    pred = Predicate(mu=ATOMIC_PREDICATES[predicate_name], label=predicate_name)

    phi_in = AgentFormula(
        agent_id=agent_id,
        child=In(
            graph_types=[graph_type],
            W=weight_interval,
            E=count_interval,
            quantifier="exists",
            child=pred,
        ),
    )

    phi = Eventually(
        child = phi_in,
        interval = (0,10)
    )

    # return phi_in
    return phi

  


