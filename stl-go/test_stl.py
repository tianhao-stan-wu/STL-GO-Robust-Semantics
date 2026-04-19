import sys, math

from syntax import Predicate, TrueF, Neg, And, Always, Eventually, Until
from algebra import MinMaxAlgebra, BooleanAlgebra
from evaluator import evaluate

# ---------------------------------------------------------------------------
# Trajectory and formula definitions
# ---------------------------------------------------------------------------
traj = {0: [1.0, 2.0, -1.0, 3.0, 0.5]}
mu   = Predicate(mu=lambda x: x, label="x")
mu2  = Predicate(mu=lambda x: x - 2, label="x-2")   # satisfied iff x >= 2
INF  = float('inf')

tests = {
    # (formula, t, agent_id)
    "TrueF":              (TrueF(),                             0, 0),
    "Predicate t=2":      (mu,                                  2, 0),
    "Neg t=2":            (Neg(mu),                             2, 0),
    "Always[0,2] t=0":    (Always(mu, (0, 2)),                  0, 0),
    "Always[0,2] t=1":    (Always(mu, (0, 2)),                  1, 0),
    "Eventually[0,2] t=0":(Eventually(mu, (0, 2)),              0, 0),
    "Until[0,2] t=0":     (Until(mu, mu2, (0, 2)),              0, 0),
}

# ---------------------------------------------------------------------------
# Expected values per algebra
# ---------------------------------------------------------------------------
minmax_expected = {
    "TrueF":               INF,
    "Predicate t=2":      -1.0,
    "Neg t=2":             1.0,
    "Always[0,2] t=0":    -1.0,   # min(1, 2, -1) = -1
    "Always[0,2] t=1":    -1.0,   # min(2, -1, 3) = -1
    "Eventually[0,2] t=0": 2.0,   # max(1, 2, -1) = 2
    "Until[0,2] t=0":      0.0,   # best at t'=1: min(0, 1) = 0
}

bool_expected = {
    "TrueF":               True,
    "Predicate t=2":       False,  # -1 < 0
    "Neg t=2":             True,   # not False
    "Always[0,2] t=0":     False,  # traj[2]=-1 fails
    "Always[0,2] t=1":     False,  # traj[2]=-1 fails
    "Eventually[0,2] t=0": True,   # traj[1]=2 passes
    "Until[0,2] t=0":      True,   # at t'=1: rhs=True, lhs=True
}

# ---------------------------------------------------------------------------
# Checker functions — one per algebra
# ---------------------------------------------------------------------------
def minmax_check(name, result, expected):
    ok = math.isinf(expected) and math.isinf(result) and (result > 0) == (expected > 0) \
         or (not math.isinf(expected) and abs(result - expected) < 1e-9)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {result}, expected {expected}")
    assert ok, f"FAILED: {name}"

def bool_check(name, result, expected):
    ok = (result == expected)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: got {result}, expected {expected}")
    assert ok, f"FAILED: {name}"

# ---------------------------------------------------------------------------
# Run loop
# ---------------------------------------------------------------------------
algebras = [
    ("MinMaxAlgebra",  MinMaxAlgebra(),  minmax_check, minmax_expected),
    ("BooleanAlgebra", BooleanAlgebra(), bool_check,   bool_expected),
]

for alg_name, alg, checker, expected in algebras:
    print(f"\n=== {alg_name} ===")
    for test_name, (formula, t, agent_id) in tests.items():
        result = evaluate(traj, None, formula, alg, t=t, agent_id=agent_id, aggregator = None)
        checker(test_name, result, expected[test_name])

print("\nAll tests passed.")


