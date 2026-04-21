
import sys
from pathlib import Path
import numpy as np

# Add stl-go directory to path
stl_go_path = Path(__file__).parent.parent / "stl-go"
sys.path.insert(0, str(stl_go_path))

from syntax import Predicate, AgentFormula, In, Out, Eventually, Always, EXV

# Define goal region (circle)
GOAL_CENTER = (0, 0)
GOAL_RADIUS = 10.0


def distance_to_goal(position, goal_center, goal_radius):
    """Compute distance from position to goal boundary.

    Positive = inside goal, Negative = outside goal
    """
    dist_to_center = np.linalg.norm(position[:2] - np.array(goal_center))
    return goal_radius - dist_to_center


ATOMIC_PREDICATES = {
    "goal": lambda state: distance_to_goal(state, GOAL_CENTER, GOAL_RADIUS),
}

def build_spec(agent_id: int, iteration_level: int = 3) -> AgentFormula:
    """Build STL-GO specification alternating temporal and spatial operators.

    Alternates between:
    - Temporal operators: Eventually (F), Always (G)
    - Spatial operators: In, Out

    Example for iteration_level=3:
    F(In(G(Out(F(In(pred))))))

    Args:
        agent_id: agent index
        iteration_level: number of temporal-spatial alternations

    Returns:
        AgentFormula with alternating temporal and spatial operators
    """
    graph_types = ["dist", "sense", "comm"]
    weight_interval = (0.0, 20.0)
    count_interval = (1, 20)
    predicate_name = "goal"

    pred = Predicate(mu=ATOMIC_PREDICATES[predicate_name], label=predicate_name)

    # Build alternating temporal and spatial operators from innermost to outermost
    child = pred
    temporal_ops = [Eventually, Always]  # Alternate between F and G
    spatial_ops = [In, Out]              # Alternate between In and Out

    for level in range(iteration_level):
        # Alternate spatial operator (In, Out, In, Out, ...)
        spatial_op = spatial_ops[level % 2]
        child = spatial_op(
            graph_types=graph_types,
            W=weight_interval,
            E=count_interval,
            quantifier="exists",
            child=child,
        )

        # Alternate temporal operator (F, G, F, G, ...)
        temporal_op = temporal_ops[level % 2]
        child = temporal_op(
            child=child,
            interval=(0, 10)
        )

    phi = EXV(child=child)

    return phi

  


