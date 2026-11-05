"""Step-by-step evidence of every IF-DLDD stage on one dataset (for the demo / viva):
[1] Isolation Forest scores  [2] sliding-window anomaly count  [3] candidate drift
[4] t-test verification  [5] confirmed drift points  [6] benefit of the t-test layer.
Also saves results/explain_<dataset>.png.

usage: python explain.py [ns3-iot | cic-ids2017 | sea | stagger | sine1 | sine2]
"""
import sys

import numpy as np
from sklearn.ensemble import IsolationForest

from ifdldd import DATASETS, OUT, evaluate, ifdldd, ns3

name = sys.argv[1] if len(sys.argv) > 1 else "ns3-iot"
loader, window, step = DATASETS[name]
X, drifts = loader(np.random.default_rng(42))
edges = [0] + drifts + [len(X)]
print(f"DATASET {name}: {len(X)} samples x {X.shape[1]} features, true drift points {drifts}")
print(f"window = {window}, step = {step}, tau = 0.6, alpha = 0.01, adaptive threshold\n")

print("[1] ISOLATION FOREST  s(x) = 2^(-E[h(x)]/c(n))   (ifdldd.py line 61)")
print("    Forest trained on concept C1 only, then every sample scored:")
s = -IsolationForest(n_estimators=50, max_samples=256, random_state=0).fit(X[:drifts[0]]).score_samples(X)
for i, (a, b) in enumerate(zip(edges, edges[1:])):
    print(f"    C{i + 1} samples {a:>6}-{b:<6} mean score {s[a:b].mean():.3f}   anomalous (s > 0.6): {(s[a:b] > .6).mean():6.1%}")
if name == "ns3-iot":
    flows = ns3()[2]
    print("    by traffic type:")
    for lab, g in flows.groupby("label").groups.items():
        print(f"      {lab:18s} mean score {s[g].mean():.3f}   anomalous: {(s[g] > .6).mean():6.1%}")
print("    -> new behaviour gets higher scores without any labels.\n")

log = ifdldd(X, window, step, adaptive=True)
plain = ifdldd(X, window, step, adaptive=True, validate=False)
pos, D, thr = np.array(log["pos"]), np.array(log["D"]), np.array(log["thr"])

print("[2] SLIDING-WINDOW ANOMALY COUNT  D_k = #{x in window : s(x) > 0.6}   (line 68)")
print(f"    {len(D)} windows; D_k around each candidate (>> = candidate):")
for c in log["cand"]:
    k = int(np.searchsorted(pos, c[0]))
    for j in range(max(1, k - 2), k + 1):
        mark = ">>" if j == k else "  "
        print(f"    {mark} window {j:>3} samples {pos[j]:>6}-{pos[j] + window:<6} D = {D[j]:>4}   "
              f"D_k - D_k-1 = {D[j] - D[j - 1]:>+5}   threshold = {thr[j]:6.1f}")
    print()

print("[3] CANDIDATE DRIFT  when D_k - D_k-1 > threshold   (line 74)")
for c in log["cand"]:
    print(f"    window at {c[0]:>6}: jump {c[7]:+d} > threshold {c[8]:.1f} -> candidate")
print()

print("[4] T-TEST VERIFICATION  one-sided Welch t-test on scores, best split m   (lines 35-49, 78)")
for c in log["cand"]:
    verdict = "CONFIRMED" if c[2] else "REJECTED (noise)"
    print(f"    candidate {c[0]:>6}: split at {c[1]:>6}  mean before {c[5]:.3f}  after {c[6]:.3f}  "
          f"t = {c[3]:7.2f}  p = {c[4]:.1e}  -> {verdict}")
print()

print("[5] CONFIRMED DRIFT POINTS   (line 85)")
for d in log["drifts"]:
    near = min(drifts, key=lambda t: abs(t - d))
    print(f"    detected {d:>6}   nearest true drift {near:>6}   error {d - near:+d} samples")
r, rp = evaluate(log["drifts"], drifts, len(X), window), evaluate(plain["drifts"], drifts, len(X), window)
print(f"    TP {r['TP']}  FP {r['FP']}  FN {r['FN']}  mean delay {r['Delay']:.1f}\n")

def loc_err(found):  # mean over true drifts of |closest detection - true drift|
    return np.mean([min(abs(d - t) for d in found) for t in drifts]) if found else np.nan


print("[6] BENEFIT OF THE T-TEST LAYER (same run, verification switched off)")
print(f"    detection layer only : TP {rp['TP']}  FP {rp['FP']}  FN {rp['FN']}  "
      f"mean |error| {loc_err(plain['drifts']):.0f} samples")
print(f"    with t-test (IF-DLDD): TP {r['TP']}  FP {r['FP']}  FN {r['FN']}  "
      f"mean |error| {loc_err(log['drifts']):.0f} samples")
print(f"    -> {sum(not c[2] for c in log['cand'])} candidate(s) rejected as noise; drift point located at the "
      "t-test split m instead of a window boundary.")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

fig, ax = plt.subplots(3, 1, figsize=(11, 9))
ax[0].plot(s, ",", color="#4269d0", alpha=0.4)
ax[0].axhline(0.6, color="#d03b3b", lw=1, label="tau = 0.6")
ax[0].set_title("[1] Isolation Forest score per sample (forest trained on C1)", loc="left", fontsize=10)
ax[1].plot(pos + window, D, color="#4269d0", label="D_k anomaly count")
ax[1].plot(pos[1:] + window, D[:-1] + thr[1:], "--", color="#9a9a9a", lw=0.8, label="D_k-1 + threshold")
for c in log["cand"]:
    ax[1].plot(c[0] + window, D[np.searchsorted(pos, c[0])], "o", color="#2a9d55" if c[2] else "#efb118")
ax[1].set_title("[2][3] sliding-window anomaly count, candidates (green = confirmed, orange = rejected)",
                loc="left", fontsize=10)
for a in ax[:2]:
    for d in drifts:
        a.axvline(d, color="#2a9d55", alpha=0.5)
    for d in log["drifts"]:
        a.axvline(d, color="#d03b3b", ls="--", lw=1)
    a.legend(fontsize=8, loc="upper right")
c = log["cand"][0]
xs = np.arange(c[9], c[9] + len(c[10]))
ax[2].plot(xs, c[10], ".", ms=2, color="#4269d0")
ax[2].hlines([c[5], c[6]], [xs[0], c[1]], [c[1], xs[-1]], color="#d03b3b", lw=2, label="mean before / after split")
ax[2].axvline(c[1], color="#d03b3b", ls="--")
ax[2].set_title(f"[4][5] t-test on first candidate: split m = {c[1]}, t = {c[3]:.1f}, p = {c[4]:.1e}",
                loc="left", fontsize=10)
ax[2].legend(fontsize=8)
ax[2].set_xlabel("sample index")
fig.tight_layout()
fig.savefig(OUT / f"explain_{name}.png", dpi=120)
print(f"\nfigure -> {OUT / f'explain_{name}.png'}")
