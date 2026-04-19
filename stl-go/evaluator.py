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
             t: int, agent_id: int = 0, aggregator: str = 'min_max'):
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

    # Undirected graph assumption: In and Out use the same neighbor set.
    q_values = []

    graph_types = list(formula.graph_types)
    for gtype in graph_types:
        neighbors = get_neighbors(
            graphs=graphs,
            graph_type=gtype,
            t=t,
            agent_id=agent_id,
            W=formula.W,
        )
        values = [evaluate(trajs, graphs, formula.child, algebra, t, j) for j in neighbors]

        # Use min_max aggregation as requested.
        q_values.append(aggregate(values, formula.E, aggregator))

    if formula.quantifier.lower() == "exists":
        return max(q_values) if q_values else algebra.bot()
    if formula.quantifier.lower() == "forall":
        return min(q_values) if q_values else algebra.top()
    raise ValueError(f"Unknown graph quantifier: {formula.quantifier}")


