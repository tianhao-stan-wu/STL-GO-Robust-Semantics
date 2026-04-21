
import sys
from pathlib import Path
import numpy as np

# Add stl-go directory to path
stl_go_path = Path(__file__).parent.parent / "stl-go"
sys.path.insert(0, str(stl_go_path))

from syntax import Predicate, In, Out, Eventually, EXV, And

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
    """Build EXV(F(Out AND In)) specification with nested operators.

    Ensures agent ego has both:
    - An outgoing edge to some agent K that satisfies the goal
    - An incoming edge from that same agent K that satisfies the goal

    The AND operator combines both conditions to find agents K in the intersection of:
    - {K : ego → K and K satisfies goal}
    - {K : K → ego and K satisfies goal}

    Args:
        agent_id: agent index
        iteration_level: nesting level (1 = Out(pred) AND In(pred), 2 = Out(Out(pred)) AND In(In(pred)), etc.)

    Returns:
        EXV with nested Out AND In operators
    """
    graph_types = ["sense", "comm"]
    weight_interval = (1, 1)
    count_interval = (1, 2)
    predicate_name = "goal"

    pred = Predicate(mu=ATOMIC_PREDICATES[predicate_name], label=predicate_name)

    # Build nested Out operators from innermost to outermost
    out_child = pred
    for _ in range(iteration_level):
        out_child = Out(
            graph_types=graph_types,
            W=weight_interval,
            E=count_interval,
            quantifier="exists",
            child=out_child,
        )

    # Build nested In operators from innermost to outermost
    in_child = pred
    for _ in range(iteration_level):
        in_child = In(
            graph_types=graph_types,
            W=weight_interval,
            E=count_interval,
            quantifier="exists",
            child=in_child,
        )

    # Combine with AND: Out AND In
    and_op = And(left=out_child, right=in_child)

    # Wrap in Eventually
    phi_eventually = Eventually(
        child=and_op,
        interval=(0, 10)
    )

    # Wrap in EXV
    phi = EXV(child=phi_eventually)

    return phi
