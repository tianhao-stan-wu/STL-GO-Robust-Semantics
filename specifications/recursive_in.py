
import sys
from pathlib import Path
import numpy as np

# Add stl-go directory to path
stl_go_path = Path(__file__).parent.parent / "stl-go"
sys.path.insert(0, str(stl_go_path))

from syntax import Predicate, AgentFormula, In, Out, Eventually, Always, EXV

# Define goal region (circle)
GOAL_CENTER = (5, 5)
GOAL_RADIUS = 10.0


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

def build_spec(agent_id: int, iteration_level: int):
    """Build STL-GO specification for an agent with nested IN operators.

    Args:
        agent_id: agent index
        iteration_level: number of nested IN operators (1 = IN(pred), 2 = IN(IN(pred)), etc.)

    Returns:
        AgentFormula with nested IN operators
    """
    graph_type = "sense"
    weight_interval = (1, 1)
    count_interval = (1, 4)
    predicate_name = "goal"

    pred = Predicate(mu=ATOMIC_PREDICATES[predicate_name], label=predicate_name)

    # Build nested IN operators from innermost to outermost
    child = pred
    for _ in range(iteration_level):
        child = In(
            graph_types=[graph_type],
            W=weight_interval,
            E=count_interval,
            quantifier="exists",
            child=child,
        )

    phi_eventually = Eventually(
        child=child,
        interval=(0, 10)
    )

    phi = EXV(child=phi_eventually)

    # print(phi)

    return phi

  


