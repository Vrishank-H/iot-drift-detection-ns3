"""Self-check: python test_ifdldd.py"""
import numpy as np

from ifdldd import best_split, evaluate, ifdldd

rng = np.random.default_rng(0)

# validation layer: Welch t-test localises a mean shift at 500
m, t, p = best_split(np.r_[rng.normal(0.4, 0.05, 500), rng.normal(0.5, 0.05, 500)], 30)
assert abs(m - 500) <= 10 and p < 1e-6, (m, t, p)
assert best_split(rng.normal(0.4, 0.05, 1000), 30)[2] > 1e-3  # no shift -> not significant

# metrics: first hit per segment is TP, extra hit FP, missing segment FN
assert evaluate([1010, 1500, 3100], [1000, 2000, 3000], 4000, tol=100) == dict(TP=2, FP=1, FN=1, Delay=55.0)

# full detector: abrupt shift of a 3-D Gaussian stream at 5000 and 10000
X = np.r_[rng.normal(0, 1, (5000, 3)), rng.normal(3, 1, (5000, 3)), rng.normal(-3, 1, (5000, 3))]
for kw in (dict(threshold=25), dict(adaptive=True)):
    found = ifdldd(X, 500, 250, **kw)["drifts"]
    assert evaluate(found, [5000, 10000], len(X), 500) == dict(TP=2, FP=0, FN=0, Delay=0.0), (kw, found)
print("all checks passed")
