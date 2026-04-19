"""
Aggregators: combine a list of neighbor robustness values into a single scalar.

Each function takes:
  values : list of robustness values from eligible neighbors
  E      : count interval (e1, e2)
  bot    : algebra's bottom value (threshold between sat and viol)
and returns a single scalar.
"""


def aggregate(values, E, method, bot):
    """Dispatch to chosen aggregation method."""
    if not values:
        return _empty_result(E)

    if   method == 'min':       return _min(values)
    elif method == 'max':       return _max(values)
    elif method == 'min_max':   return _min_max(values, E)
    elif method == 'counting':  return _counting(values, E, bot)
    elif method == 'averaging': return _averaging(values)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def _min_max(values, E):
    """
    Order-statistic aggregator for a count interval E = [e1, e2].

    Robustness formula:
        min(r_(e1), -r_(e2+1))

    where values are sorted in descending order and r_(k) = -inf for k > len(values).
    """
    e1, e2 = E
    vals = sorted(values, reverse=True)

    def r(k):
        # 1-indexed order statistic; convention r(k) = -inf for k > len(vals)
        return vals[k - 1] if 1 <= k <= len(vals) else float("-inf")

    return min(r(e1), -r(e2 + 1))
    

# def _empty_result(E):
#     """No eligible neighbors: satisfied iff 0 falls in E."""
#     return float('inf') if E[0] == 0 else float('-inf')


# def _min(values):
#     """Min robustness — 'all neighbors satisfy φ'."""
#     return min(values)


# def _max(values):
#     """Max robustness — 'at least one neighbor satisfies φ'."""
#     return max(values)


# def _counting(values, E, bot):
#     """
#     Count satisfying neighbors (v > bot), return margin w.r.t. interval E.
#       c < e1        → c - e1          (negative: too few)
#       e1 <= c <= e2 → min(c-e1, e2-c) (positive: margin from both bounds)
#       c > e2        → e2 - c          (negative: too many)
#     """
#     e1, e2 = E
#     c = sum(1 for v in values if v > bot)

#     if c < e1:
#         return float(c - e1)
#     elif c > e2:
#         return float(e2 - c)
#     else:
#         margin_hi = (e2 - c) if e2 < float('inf') else float('inf')
#         return float(min(c - e1, margin_hi))


# def _averaging(values):
#     """Average robustness over all eligible neighbors."""
#     return sum(values) / len(values)