# """
# Simple STL robustness test.

# Trajectory: [1.0, 2.0, -1.0, 3.0, 0.5]
# Predicate:  mu(x) = x  →  robustness is the raw signal value

# Hand-computed expected values are listed next to each assert.
# """

# import sys
# sys.path.insert(0, "..")

# from syntax import Predicate, Always, Eventually, And, Neg
# from algebra import MinMaxAlgebra
# from evaluator import evaluate

# # -------------------------------------------------------------------
# # Setup
# # -------------------------------------------------------------------
# traj = [1.0, 2.0, -1.0, 3.0, 0.5]
# alg  = MinMaxAlgebra()
# mu   = Predicate(mu=lambda x: x, label="x")   # robustness = x


# def check(name, result, expected):
#     status = "PASS" if abs(result - expected) < 1e-9 else "FAIL"
#     print(f"[{status}] {name}: got {result}, expected {expected}")
#     assert abs(result - expected) < 1e-9, f"{name} failed"


# # -------------------------------------------------------------------
# # 1. Predicate only
# #    ρ(x, t=2) = traj[2] = -1.0
# # -------------------------------------------------------------------
# check("Predicate t=2", evaluate(traj, mu, alg, t=2), -1.0)


# # -------------------------------------------------------------------
# # 2. Negation
# #    ρ(¬x, t=2) = -(-1.0) = 1.0
# # -------------------------------------------------------------------
# check("Neg t=2", evaluate(traj, Neg(mu), alg, t=2), 1.0)


# # -------------------------------------------------------------------
# # 3. Always G[0,2](x)
# #    at t=0: min(traj[0], traj[1], traj[2]) = min(1, 2, -1) = -1.0
# #    at t=1: min(traj[1], traj[2], traj[3]) = min(2, -1, 3) = -1.0
# #    at t=2: min(traj[2], traj[3], traj[4]) = min(-1, 3, 0.5) = -1.0
# # -------------------------------------------------------------------
# G = Always(mu, interval=(0, 2))
# check("Always[0,2] t=0", evaluate(traj, G, alg, t=0), -1.0)
# check("Always[0,2] t=1", evaluate(traj, G, alg, t=1), -1.0)
# check("Always[0,2] t=2", evaluate(traj, G, alg, t=2), -1.0)


# # -------------------------------------------------------------------
# # 4. Eventually F[0,2](x)
# #    at t=0: max(traj[0], traj[1], traj[2]) = max(1, 2, -1) = 2.0
# #    at t=1: max(traj[1], traj[2], traj[3]) = max(2, -1, 3) = 3.0
# #    at t=2: max(traj[2], traj[3], traj[4]) = max(-1, 3, 0.5) = 3.0
# # -------------------------------------------------------------------
# F = Eventually(mu, interval=(0, 2))
# check("Eventually[0,2] t=0", evaluate(traj, F, alg, t=0), 2.0)
# check("Eventually[0,2] t=1", evaluate(traj, F, alg, t=1), 3.0)
# check("Eventually[0,2] t=2", evaluate(traj, F, alg, t=2), 3.0)


# # -------------------------------------------------------------------
# # 5. And: G[0,1](x) ∧ F[0,2](x)
# #    at t=0:
# #      G[0,1]: min(traj[0], traj[1]) = min(1, 2) = 1.0
# #      F[0,2]: max(traj[0], traj[1], traj[2]) = max(1, 2, -1) = 2.0
# #      And:    min(1.0, 2.0) = 1.0
# # -------------------------------------------------------------------
# phi = And(Always(mu, interval=(0, 1)), Eventually(mu, interval=(0, 2)))
# check("And(G[0,1], F[0,2]) t=0", evaluate(traj, phi, alg, t=0), 1.0)


# print("\nAll tests passed.")


"""
STL robustness/satisfaction tests.
Runs the same set of tests for each algebra in a loop.
Each algebra is paired with a checker function and a dict of expected values.

Trajectory: [1.0, 2.0, -1.0, 3.0, 0.5]
Predicate mu(x) = x
"""

import sys, math

from syntax import Predicate, TrueF, Neg, And, Always, Eventually, Until
from algebra import MinMaxAlgebra, BooleanAlgebra
from evaluator import evaluate

# ---------------------------------------------------------------------------
# Trajectory and formula definitions
# ---------------------------------------------------------------------------
traj = [1.0, 2.0, -1.0, 3.0, 0.5]
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
        result = evaluate(traj, formula, alg, t=t, agent_id=agent_id)
        checker(test_name, result, expected[test_name])

print("\nAll tests passed.")