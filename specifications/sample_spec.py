"""Sample STL-GO specification with graph operators."""

import sys
from pathlib import Path
from typing import Tuple
import numpy as np

# Add stl-go directory to path
stl_go_path = Path(__file__).parent.parent / "stl-go"
sys.path.insert(0, str(stl_go_path))

from syntax import Predicate, AgentFormula, In, Out

# ---------------------------------------------------------------------------
# Atomic predicate
# ---------------------------------------------------------------------------

def y_nonneg(state: np.ndarray) -> float:
    """Robustness of y >= 0."""
    return float(state[1])


ATOMIC_PREDICATES = {
    "y_nonneg": y_nonneg,
}

def build_example_specs(agent_id):
    graph_type: str = "dist"
    weight_interval: Tuple[float, float] = (0.0, 10.0)
    count_interval: Tuple[int, int] = (1, 4)
    predicate_name: str = "y_nonneg"

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

    phi_out = AgentFormula(
        agent_id = agent_id,
        child= Out(
            graph_types=[graph_type],
            W=weight_interval,
            E=count_interval,
            quantifier="exists",
            child=pred,
        ),
    )

    return phi_in, phi_out