"""
Aggregators: combine a list of neighbor robustness values into a single scalar.

Each function takes:
  values : list of robustness values from eligible neighbors
  E      : count interval (e1, e2)
  bot    : algebra's bottom value (threshold between sat and viol)
and returns a single scalar.
"""
import math


# aggregation options for neightbor values
def aggregate(values, E, method):
    """Dispatch to chosen aggregation method."""
    if not values:
        return _empty_result(E)

    if method == 'min_max':     return _min_max(values, E)
    elif method == 'counting':  return _counting(values, E)
    elif method == 'averaging': return _averaging(values)
    elif method == 'hybrid':    return _hybrid(values, E)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def _empty_result(E):
    """No eligible neighbors: satisfied iff 0 falls in E."""
    return float('inf') if E[0] == 0 else float('-inf')


def _min_max(values, E):
    """
    Order-statistic aggregator for a count interval E = [e1, e2].

    Robustness formula:
        min(r_(e1), -r_(e2+1))

    where values are sorted in descending order and r_(k) = -inf for k > len(values).
    """
    e1, e2 = E
    vals = sorted(values, reverse=True)

    def r(k: int) -> float:
        if k == 0:
            return float("inf")
        if 1 <= k <= len(vals):
            return float(vals[k - 1])
        return float("-inf")

    upper = float("inf") if math.isinf(e2) else -r(int(e2) + 1)
    return min(r(int(e1)), upper)


def _counting(values, E):
    """
    Signed-deficit aggregator for a count interval E = [e1, e2].

    Count satisfying neighbors (v > 0), for boolean algebra, True = 1 > 0; for min-max, robustness > 0
      c < e1        → c - e1          (negative: too few)
      e1 <= c <= e2 → min(c-e1, e2-c) (positive: margin from both bounds)
      c > e2        → e2 - c          (negative: too many)
    """
    e1, e2 = E
    c = sum(1 for v in values if v > 0)

    if c < e1:
        return float(c - e1)
    elif c > e2:
        return float(e2 - c)
    else:
        return float(min(c - e1, (e2 - c)))


def _averaging(values):
    """Average robustness over all eligible neighbors."""
    return sum(values) / len(values)


def _hybrid(values, E, alpha=10.0):
    """
    Hybrid aggregator: r_(e1) + alpha * (c_plus - e1)

    """
    e1, _e2 = E
    vals = sorted(values, reverse=True)

    def r(k):
        # 1-indexed order statistic
        if k <= 0:
            return float("inf")
        if k <= len(vals):
            return float(vals[k - 1])
        return float("-inf")

    c = sum(1 for v in values if v > 0)
    return float(r(e1) + alpha * (c - e1))



