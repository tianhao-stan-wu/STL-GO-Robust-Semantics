"""
Relay-chain specification generator for STL-GO.

Models multi-hop information propagation through a network:
  psi_0 := pi_source
  psi_{k+1} := In^{exists}_{G_c, [1,+∞)} (psi_k)
  phi_k := F_{[0,T]} psi_k

psi_k means an agent can receive information from a source through a k-hop relay chain.
phi_k means the agent eventually becomes connected to a source through a k-hop chain within time T.
"""

import sys
import os
from typing import Callable, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stl_go.syntax import Formula, Predicate, In, Eventually
from stl_go import pretty_print


def make_relay_formula(
    k: int,
    source_pred: Optional[Callable] = None,
    source_label: str = "source",
    graph_name: str = "G_c"
) -> Formula:
    """
    Generate psi_k: the k-hop relay chain formula.

    psi_0 := pi_source
    psi_{k+1} := In^{exists}_{G_c, [1,+∞)} (psi_k)

    Args:
        k: Number of hops (0 = direct source, 1 = 1-hop relay, etc.)
        source_pred: Callable that evaluates to robustness value. If None, uses default.
        source_label: Label for the base predicate node.
        graph_name: Name of the graph to use (e.g., "G_c" for comm graph).

    Returns:
        Formula representing psi_k.

    Raises:
        ValueError: if k < 0.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    # Default predicate: check if state has source attribute
    if source_pred is None:
        def default_source(state):
            if isinstance(state, dict):
                return 1.0 if state.get("source", False) else -1.0
            elif isinstance(state, bool):
                return 1.0 if state else -1.0
            return -1.0
        source_pred = default_source

    # Base case: psi_0 is just the source predicate
    if k == 0:
        return Predicate(mu=source_pred, label=source_label)

    # Recursive case: wrap previous formula in In operator
    child = make_relay_formula(k - 1, source_pred, source_label, graph_name)
    return In(
        graph_types=[graph_name],
        W=(0, float('inf')),
        E=(1, float('inf')),
        quantifier="exists",
        child=child
    )


def make_eventual_relay_formula(
    k: int,
    T: int,
    source_pred: Optional[Callable] = None,
    source_label: str = "source",
    graph_name: str = "G_c"
) -> Formula:
    """
    Generate phi_k: the eventual k-hop relay formula.

    phi_k := F_{[0,T]} psi_k

    Meaning: within time horizon T, the agent eventually becomes connected to a
    source through a k-hop relay chain.

    Args:
        k: Number of hops.
        T: Time horizon (in timesteps).
        source_pred: Callable for source predicate. If None, uses default.
        source_label: Label for base predicate.
        graph_name: Name of the graph.

    Returns:
        Formula representing phi_k.
    """
    psi_k = make_relay_formula(k, source_pred, source_label, graph_name)
    return Eventually(psi_k, (0, T))


if __name__ == "__main__":
    # Example: 2-hop relay chain
    print("=== psi_0 (source predicate) ===")
    psi_0 = make_relay_formula(0)
    print(pretty_print(psi_0))
    print()

    print("=== psi_1 (1-hop relay) ===")
    psi_1 = make_relay_formula(1)
    print(pretty_print(psi_1))
    print()

    print("=== psi_2 (2-hop relay) ===")
    psi_2 = make_relay_formula(2)
    print(pretty_print(psi_2))
    print()

    print("=== phi_2 (eventual 2-hop relay, T=10) ===")
    phi_2 = make_eventual_relay_formula(2, T=10)
    print(pretty_print(phi_2))
