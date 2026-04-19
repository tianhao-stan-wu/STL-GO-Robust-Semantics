# """
# Evaluator: computes robustness of an STL-GO formula given a trajectory,
# an algebra, an agent id, and a time index.

# For now handles STL operators only (no graph operators yet).
# Trajectory is assumed to be a list/array where traj[t] is the state at time t.
# Time interval bounds in Until/F/G are treated as integer time steps.
# """

# from syntax import (
#     Formula, TrueF, Predicate, Neg, And, Until, In, Out, AgentFormula
# )
# from algebra import Algebra


# def evaluate(traj, formula: Formula, algebra: Algebra, t: int, agent_id: int = 0) -> float:
#     """
#     Recursively evaluate formula at time t for agent agent_id.
#     Returns a scalar robustness value in the algebra's domain.
#     """

#     if isinstance(formula, TrueF):
#         return algebra.top()

#     elif isinstance(formula, Predicate):
#         # μ(state) — state at time t for this agent
#         return algebra.predicate(formula.mu(traj[t]))

#     elif isinstance(formula, Neg):
#         return algebra.neg_op(evaluate(traj, formula.child, algebra, t, agent_id))

#     elif isinstance(formula, And):
#         l = evaluate(traj, formula.left,  algebra, t, agent_id)
#         r = evaluate(traj, formula.right, algebra, t, agent_id)
#         return algebra.and_op(l, r)

#     elif isinstance(formula, Until):
#         return _eval_until(traj, formula, algebra, t, agent_id)

#     elif isinstance(formula, (In, Out, AgentFormula)):
#         raise NotImplementedError("Graph operators not yet implemented.")

#     else:
#         raise ValueError(f"Unknown formula node: {type(formula)}")


# def _eval_until(traj, formula: Until, algebra: Algebra, t: int, agent_id: int) -> float:
#     """
#     φ1 U_[t1,t2] φ2 at time t:
#       max over t' in [t+t1, t+t2] of:
#         min( ρ(φ2, t'),  min over t'' in [t, t'] of ρ(φ1, t'') )
#     """
#     t1, t2 = int(formula.interval[0]), int(formula.interval[1])
#     T = len(traj) - 1  # last valid index

#     result = algebra.bot()

#     for tp in range(t + t1, min(t + t2, T) + 1):
#         # robustness of φ2 at t'
#         rhs = evaluate(traj, formula.right, algebra, tp, agent_id)

#         # robustness of φ1 over [t, t']
#         lhs = algebra.top()
#         for tpp in range(t, tp + 1):
#             v = evaluate(traj, formula.left, algebra, tpp, agent_id)
#             lhs = algebra.and_op(lhs, v)

#         result = algebra.or_op(result, algebra.and_op(rhs, lhs))

#     return result



"""
Evaluator: computes robustness of an STL / STL-GO formula.

Inputs:
  trajs      : dict {agent_id: list_of_states}
  graphs     : dict {graph_type: list_of_edge_dicts}  (empty {} for pure STL)
  formula    : STL-GO AST node
  algebra    : Algebra instance
  t          : discrete time index
  agent_id   : agent this formula is evaluated for (default 0)
  aggregator : 'min' | 'max' | 'counting' | 'averaging'  (for graph operators)
"""

from syntax import Formula, TrueF, Predicate, Neg, And, Until, In, Out, AgentFormula
from algebra import Algebra
from graph_ops import get_neighbors
from aggregators import aggregate


def evaluate(trajs, graphs, formula: Formula, algebra: Algebra,
             t: int, agent_id: int = 0, aggregator: str = 'counting'):
    """Recursively evaluate formula at time t for agent_id."""

    if isinstance(formula, TrueF):
        return algebra.top()

    elif isinstance(formula, Predicate):
        return algebra.predicate(formula.mu(trajs[agent_id][t]))

    elif isinstance(formula, Neg):
        return algebra.neg_op(
            evaluate(trajs, graphs, formula.child, algebra, t, agent_id, aggregator))

    elif isinstance(formula, And):
        l = evaluate(trajs, graphs, formula.left,  algebra, t, agent_id, aggregator)
        r = evaluate(trajs, graphs, formula.right, algebra, t, agent_id, aggregator)
        return algebra.and_op(l, r)

    elif isinstance(formula, Until):
        return _eval_until(trajs, graphs, formula, algebra, t, agent_id, aggregator)

    elif isinstance(formula, (In, Out)):
        return _eval_graph_op(trajs, graphs, formula, algebra, t, agent_id, aggregator)

    elif isinstance(formula, AgentFormula):
        return evaluate(trajs, graphs, formula.child, algebra, t, formula.agent_id, aggregator)

    else:
        raise ValueError(f"Unknown formula node: {type(formula)}")


# ---------------------------------------------------------------------------
# Temporal
# ---------------------------------------------------------------------------

def _eval_until(trajs, graphs, formula, algebra, t, agent_id, aggregator):
    """
    φ1 U_[t1,t2] φ2 :
      or  over t' in [t+t1, t+t2] of:
        and( ρ(φ2,t'),  and over t'' in [t,t'] of ρ(φ1,t'') )
    """
    t1, t2 = int(formula.interval[0]), int(formula.interval[1])
    T = len(next(iter(trajs.values()))) - 1

    result = algebra.bot()
    for tp in range(t + t1, min(t + t2, T) + 1):
        rhs = evaluate(trajs, graphs, formula.right, algebra, tp, agent_id, aggregator)
        lhs = algebra.top()
        for tpp in range(t, tp + 1):
            v = evaluate(trajs, graphs, formula.left, algebra, tpp, agent_id, aggregator)
            lhs = algebra.and_op(lhs, v)
        result = algebra.or_op(result, algebra.and_op(rhs, lhs))
    return result


# ---------------------------------------------------------------------------
# Graph operators
# ---------------------------------------------------------------------------

def _eval_graph_op(trajs, graphs, formula, algebra, t, agent_id, aggregator):
    """
    In/Out graph operator:
      for each graph type: get neighbors, evaluate child for each, aggregate
      then apply existential (or) or universal (and) over graph types
    """
    direction = 'in' if isinstance(formula, In) else 'out'

    if formula.quantifier == 'exists':
        q_init, q_op = algebra.bot(), algebra.or_op
    else:
        q_init, q_op = algebra.top(), algebra.and_op

    result = q_init
    for gtype in formula.graph_types:
        neighbors = get_neighbors(graphs, gtype, t, agent_id, formula.W, direction)

        neighbor_vals = [
            evaluate(trajs, graphs, formula.child, algebra, t, j, aggregator)
            for j in neighbors
        ]

        agg = aggregate(neighbor_vals, formula.E, aggregator, bot=algebra.bot())
        result = q_op(result, agg)

    return result