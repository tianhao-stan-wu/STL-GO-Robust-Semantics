"""
STL-GO syntax: AST nodes for both STL-GO-S (agent-local) and STL-GO (multi-agent).

STL-GO-S: φ ::= ⊤ | π | ¬φ | φ∧φ | φ U_I φ | In^{W,#}_{G,E} φ | Out^{W,#}_{G,E} φ
STL-GO:   ϕ ::= ⊤ | π | i.φ | ¬ϕ | ϕ∧ϕ | ϕ U_I ϕ
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Interval helpers
# ---------------------------------------------------------------------------

Interval = Tuple[float, float]   # [lo, hi], use float('inf') for +∞


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class Formula:
    """Base class for all STL-GO formula nodes."""
    pass


# ---------------------------------------------------------------------------
# Shared by both STL-GO-S and STL-GO
# ---------------------------------------------------------------------------

@dataclass
class TrueF(Formula):
    """⊤ — always true."""
    pass


@dataclass
class Predicate(Formula):
    """Atomic predicate: true iff μ(state) ≥ 0."""
    mu: Callable        # μ: R^n → R  (agent-local)
    label: str = ""     # optional human-readable name


@dataclass
class Neg(Formula):
    """¬φ"""
    child: Formula


@dataclass
class And(Formula):
    """φ1 ∧ φ2"""
    left: Formula
    right: Formula


@dataclass
class Until(Formula):
    """φ1 U_[t1,t2] φ2"""
    left: Formula
    right: Formula
    interval: Interval  # [t1, t2]


# ---------------------------------------------------------------------------
# Derived temporal operators (convenience constructors)
# ---------------------------------------------------------------------------

def Eventually(child: Formula, interval: Interval) -> Until:
    """F_I φ  =  ⊤ U_I φ"""
    return Until(TrueF(), child, interval)


def Always(child: Formula, interval: Interval) -> Neg:
    """G_I φ  =  ¬(F_I ¬φ)"""
    return Neg(Eventually(Neg(child), interval))


def Or(left: Formula, right: Formula) -> Neg:
    """φ1 ∨ φ2  =  ¬(¬φ1 ∧ ¬φ2)"""
    return Neg(And(Neg(left), Neg(right)))


def Implies(left: Formula, right: Formula) -> Neg:
    """φ1 → φ2  =  ¬φ1 ∨ φ2"""
    return Or(Neg(left), right)


# ---------------------------------------------------------------------------
# STL-GO-S graph operators (agent-local)
# ---------------------------------------------------------------------------

@dataclass
class In(Formula):
    """
    In^{W,#}_{G,E} φ  — incoming graph operator.
    Counts neighbors j with edge (j,i) whose weight ∈ W and (j,t) |= φ.
    The count must fall in interval E.
    """
    graph_types: List[str]  # subset of graph keys, e.g. ["comm", "sense"]
    W: Interval             # edge-weight interval [w1, w2]
    E: Interval             # count interval [e1, e2]
    quantifier: str         # "exists" or "forall" over graph_types
    child: Formula


@dataclass
class Out(Formula):
    """
    Out^{W,#}_{G,E} φ  — outgoing graph operator.
    Counts neighbors j with edge (i,j) whose weight ∈ W and (j,t) |= φ.
    The count must fall in interval E.
    """
    graph_types: List[str]
    W: Interval
    E: Interval
    quantifier: str         # "exists" or "forall"
    child: Formula


# ---------------------------------------------------------------------------
# STL-GO multi-agent operator
# ---------------------------------------------------------------------------

@dataclass
class AgentFormula(Formula):
    """
    i.φ  — embeds an agent-local STL-GO-S formula φ into multi-agent context.
    Evaluates φ from the perspective of agent i.
    """
    agent_id: int
    child: Formula          # must be an STL-GO-S formula




# TODO: multi-agent predicate function operator

# TODO: EXV, FAV operators


