"""
Evaluator: computes robustness of an STL-GO formula given a trajectory,
an algebra, an agent id, and a time index.

For now handles STL operators only (no graph operators yet).
Trajectory is assumed to be a list/array where traj[t] is the state at time t.
Time interval bounds in Until/F/G are treated as integer time steps.
"""

from syntax import (
    Formula, TrueF, Predicate, Neg, And, Until, In, Out, AgentFormula
)
from algebra import Algebra


def evaluate(traj, formula: Formula, algebra: Algebra, t: int, agent_id: int = 0) -> float:
    """
    Recursively evaluate formula at time t for agent agent_id.
    Returns a scalar robustness value in the algebra's domain.
    """

    if isinstance(formula, TrueF):
        return algebra.top()

    elif isinstance(formula, Predicate):
        # μ(state) — state at time t for this agent
        return formula.mu(traj[t])

    elif isinstance(formula, Neg):
        return algebra.neg_op(evaluate(traj, formula.child, algebra, t, agent_id))

    elif isinstance(formula, And):
        l = evaluate(traj, formula.left,  algebra, t, agent_id)
        r = evaluate(traj, formula.right, algebra, t, agent_id)
        return algebra.and_op(l, r)

    elif isinstance(formula, Until):
        return _eval_until(traj, formula, algebra, t, agent_id)

    elif isinstance(formula, (In, Out, AgentFormula)):
        raise NotImplementedError("Graph operators not yet implemented.")

    else:
        raise ValueError(f"Unknown formula node: {type(formula)}")


def _eval_until(traj, formula: Until, algebra: Algebra, t: int, agent_id: int) -> float:
    """
    φ1 U_[t1,t2] φ2 at time t:
      max over t' in [t+t1, t+t2] of:
        min( ρ(φ2, t'),  min over t'' in [t, t'] of ρ(φ1, t'') )
    """
    t1, t2 = int(formula.interval[0]), int(formula.interval[1])
    T = len(traj) - 1  # last valid index

    result = algebra.bot()

    for tp in range(t + t1, min(t + t2, T) + 1):
        # robustness of φ2 at t'
        rhs = evaluate(traj, formula.right, algebra, tp, agent_id)

        # robustness of φ1 over [t, t']
        lhs = algebra.top()
        for tpp in range(t, tp + 1):
            v = evaluate(traj, formula.left, algebra, tpp, agent_id)
            lhs = algebra.and_op(lhs, v)

        result = algebra.or_op(result, algebra.and_op(rhs, lhs))

    return result