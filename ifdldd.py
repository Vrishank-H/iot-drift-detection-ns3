"""IF-DLDD: double-layer concept drift detection based on Isolation Forest
(Chu et al., "Concept Drift Analysis Based on Isolation Forest for Effectively Detecting
Network Attack in IoT Scenarios", IEEE Access 14, 2026) + two lightweight IoT extensions.

  Detection layer  : Isolation Forest fitted on the reference window WS1 scores each sliding
                     window WS2; anomaly count D = #{s(x) > tau}. D_k - D_{k-1} > threshold
                     -> candidate drift (Algorithm 2).
  Validation layer : candidate window split at m into T(1), T(2); one-sided Welch t-test on the
                     anomaly scores (eq. 3-4). t > t_alpha(nu) -> confirmed drift at m (Alg. 3).
  Extension 1 (ours): adaptive detection threshold = median + k*std of recent D_k - D_{k-1}
                     within the current concept, so no per-dataset threshold tuning is needed
                     (the paper's stated limitation).
  Extension 2 (ours): per-node drift attribution - on each confirmed drift, rank IoT nodes by
                     their anomalous records after the drift point -> "suspect nodes".

usage: python ifdldd.py [dataset ...]   (default: every available dataset)
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).parent
OUT = ROOT / "results"
FEATURES = ["flow_bytes_s", "flow_pkts_s", "fwd_pkts_s", "flow_iat_mean", "flow_duration",
            "avg_pkt_size", "bwd_pkts_s", "pkt_len_mean", "flow_iat_max", "pkt_len_std"]


# ----------------------------------------------------------------------------- algorithm
def best_split(z, min_seg):
    """Welch t for every split m of z (vectorised with cumsums); return the split maximising
    t = (mean(T2) - mean(T1)) / se, with its one-sided p-value (eq. 3-4)."""
    n = len(z)
    c1, c2 = np.cumsum(z), np.cumsum(z * z)
    m = np.arange(min_seg, n - min_seg + 1)
    n1, n2 = m, n - m
    mu1, mu2 = c1[m - 1] / n1, (c1[-1] - c1[m - 1]) / n2
    v1 = np.maximum(c2[m - 1] - n1 * mu1**2, 0) / (n1 - 1) + 1e-12
    v2 = np.maximum(c2[-1] - c2[m - 1] - n2 * mu2**2, 0) / (n2 - 1) + 1e-12
    t = (mu2 - mu1) / np.sqrt(v1 / n1 + v2 / n2)
    i = int(np.argmax(t))
    a, b = v1[i] / n1[i], v2[i] / n2[i]
    nu = (a + b) ** 2 / (a * a / (n1[i] - 1) + b * b / (n2[i] - 1))
    return int(m[i]), float(t[i]), float(stats.t.sf(t[i], nu))


def ifdldd(X, window, step, threshold=None, adaptive=False, validate=True, tau=0.6, alpha=0.01,
           k=3.0, ref_max=4, n_trees=50, nodes=None, seed=0):
    """Run IF-DLDD over stream X (n x d). Returns a log dict with per-window anomaly counts,
    thresholds, candidates and confirmed drift points (+ suspect nodes if `nodes` given)."""
    rng = np.random.RandomState(seed)
    thr0 = threshold if threshold is not None else 0.05 * window

    def fit(a, b):
        a = max(a, b - ref_max * window)  # ponytail: reference capped at ref_max windows (bounded memory)
        f = IsolationForest(n_estimators=n_trees, max_samples=min(256, b - a), random_state=rng)
        return f.fit(X[a:b]), a

    model, ref_a = fit(0, window)
    d_prev, hist, pos = None, [], step
    log = dict(pos=[], D=[], thr=[], cand=[], drifts=[], suspects=[])
    while pos + window <= len(X):
        d = int((-model.score_samples(X[pos:pos + window]) > tau).sum())
        thr = thr0
        if adaptive and len(hist) >= 5:
            thr = max(0.02 * window, np.median(hist) + k * np.std(hist))
        log["pos"].append(pos), log["D"].append(d), log["thr"].append(thr)

        if d_prev is not None and d - d_prev > thr:  # detection layer: candidate drift
            a, b = max(ref_a, pos - window), pos + window
            z = -model.score_samples(X[a:b])
            if validate:  # validation layer
                m, t, p = best_split(z, max(30, len(z) // 10))
                ok = p < alpha
            else:
                m, t, p, ok = pos + window - step - a, np.nan, np.nan, True
            log["cand"].append((pos, a + m, ok, t, p, z[:m].mean(), z[m:].mean(), d - d_prev, thr, a, z))
            if ok:
                drift = a + m
                log["drifts"].append(drift)
                if nodes is not None:  # extension 2: who drifted?
                    post = pd.Series(nodes[drift:b])[z[m:] > tau]
                    log["suspects"].append(post.value_counts().head(8).to_dict())
                model, ref_a = fit(drift if b - drift >= window // 2 else pos, b)
                d_prev, hist, pos = None, [], pos + step
                continue
        elif d_prev is not None:
            hist = (hist + [d - d_prev])[-20:]
        model, ref_a = fit(ref_a, pos + window)  # WS1 = [WS1 . WS2]
        d_prev, pos = d, pos + step
    log["model_bytes"] = len(pickle.dumps(model))
    return log


# ---------------------------------------------------------------------------- evaluation
def evaluate(detected, true_drifts, n, tol):
    """TP: first detection in [d_i - tol, d_{i+1} - tol); later ones in the segment are FP."""
    edges = list(true_drifts) + [n + tol]
    tp, fp, delays = 0, sum(p < true_drifts[0] - tol for p in detected), []
    for i, d in enumerate(true_drifts):
        hits = sorted(p for p in detected if d - tol <= p < edges[i + 1] - tol)
        if hits:
            tp += 1
            delays.append(max(0, hits[0] - d))
            fp += len(hits) - 1
    return dict(TP=tp, FP=fp, FN=len(true_drifts) - tp, Delay=float(np.mean(delays)) if delays else np.nan)


def f1_paper(r, max_delay, a=0.6, b=1.0, g=0.3):
    """Eq. (6)-(7): F1 = (1+b^2)TP / ((1+b^2)TP + b^2 FN + a NormDelay + g FP)."""
    nd = 1.0 if np.isnan(r["Delay"]) else r["Delay"] / max_delay if max_delay else 0.0
    return (1 + b * b) * r["TP"] / ((1 + b * b) * r["TP"] + b * b * r["FN"] + a * nd + g * r["FP"])


# ------------------------------------------------------------------------------ datasets
def sea(rng, n=60000, every=15000):  # abrupt, 10% noise, theta in {8,9,7,9.5}
    X = rng.uniform(0, 10, (n, 3))
    y = X[:, 0] + X[:, 1] <= np.repeat([8, 9, 7, 9.5], every)[:n]
    y ^= rng.random(n) < 0.1
    return np.c_[X / 10, y], list(range(every, n, every))


def stagger(rng, n=100000, every=33334):  # abrupt, 3 categorical attributes
    size, color, shape = rng.integers(0, 3, (3, n))
    c = np.arange(n) // every
    y = np.choose(c, [(size == 0) & (color == 0), (color == 1) | (shape == 0), size >= 1])
    return np.c_[size / 2, color / 2, shape / 2, y], list(range(every, n, every))


def sine(rng, f, n=100000, every=20000):  # abrupt + reoccurring: labels reversed every concept
    X = rng.uniform(0, 1, (n, 2))
    y = (X[:, 1] < f(X[:, 0])) ^ ((np.arange(n) // every) % 2 == 1)
    return np.c_[X, y], list(range(every, n, every))


def slog(df):  # signed log1p: heavy-tailed traffic features -> comparable scales
    v = df.replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    return np.sign(v) * np.log1p(np.abs(v))


def ns3(path=OUT / "flows.csv"):
    f = pd.read_csv(path)
    drifts = [int(np.argmax(f.concept.to_numpy() >= c)) for c in range(2, 6)]
    return slog(f[FEATURES]), drifts, f


CIC_DIR = ROOT / "data" / "cicids2017"
CIC_FEATURES = ["Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s", "Flow IAT Mean", "Flow Duration",
                "Average Packet Size", "Bwd Packets/s", "Packet Length Mean", "Flow IAT Max",
                "Packet Length Std"]  # Table 9
CIC_CONCEPTS = [  # Table 10: label -> samples, 20000 per concept
    {"BENIGN": 20000},
    {"BENIGN": 8000, "SSH-Patator": 5000, "FTP-Patator": 7000},
    {"BENIGN": 5000, "DoS Hulk": 13687, "DoS GoldenEye": 598, "DoS slowloris": 346,
     "DoS Slowhttptest": 359, "Heartbleed": 10},
    {"BENIGN": 17970, "Web Attack Brute Force": 1384, "Web Attack Sql Injection": 18,
     "Web Attack XSS": 598, "Infiltration": 30},
    {"BENIGN": 2000, "Bot": 1000, "PortScan": 7000, "DDoS": 10000},
]


def cicids(rng):
    parts = []
    for p in sorted(CIC_DIR.glob("*.parquet")):
        d = pd.read_parquet(p)
        d.columns = d.columns.str.strip()
        parts.append(d[CIC_FEATURES + ["Label"]])
    d = pd.concat(parts, ignore_index=True)
    d["Label"] = d.Label.str.replace(r"[^\x20-\x7e]", "", regex=True).str.replace("  ", " ").str.strip()
    by = {k: g.index.to_numpy() for k, g in d.groupby(d.Label.str.lower())}
    idx = []
    for concept in CIC_CONCEPTS:
        c = np.concatenate([rng.choice(by[lab.lower()], k, replace=False) for lab, k in concept.items()])
        idx.append(rng.permutation(c))
    d = d.loc[np.concatenate(idx)].reset_index(drop=True)
    return slog(d[CIC_FEATURES]), [20000, 40000, 60000, 80000]


# --------------------------------------------------------------------------- experiments
# name -> (loader, window, step); window/step follow the paper's best settings (SEA 1000/600,
# Sine1 2000/1200, Sine2 2000/1000, CIC 1600/1300) or its heuristic (window 5-20% of the drift
# interval, step ~ half window).
DATASETS = {
    "ns3-iot": (lambda r: ns3()[:2], 300, 150),
    "cic-ids2017": (cicids, 1600, 1300),
    "sea": (sea, 1000, 600),
    "stagger": (stagger, 2000, 1000),
    "sine1": (lambda r: sine(r, np.sin), 2000, 1200),
    "sine2": (lambda r: sine(r, lambda x: 0.5 + 0.3 * np.sin(3 * np.pi * x)), 2000, 1000),
}
# Paper protocol: fixed threshold picked per dataset from a small candidate set by best F1.
THRESHOLD_GRID = [0.01, 0.02, 0.05, 0.1]  # x window


def plot(name, X, drifts, logs, window):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(logs), 1, figsize=(11, 2.6 * len(logs)), sharex=True)
    for ax, (method, log) in zip(axes, logs.items()):
        x = np.array(log["pos"]) + window
        ax.plot(x, log["D"], color="#4269d0", lw=1.2, label="anomaly count D (window)")
        ax.plot(x, np.r_[np.nan, np.array(log["D"][:-1]) + np.array(log["thr"][1:])], color="#9a9a9a",
                lw=0.8, ls="--", label="D(k-1) + threshold")
        for d in drifts:
            ax.axvline(d, color="#2a9d55", lw=1.5, alpha=0.7)
        for pos, pt, ok, *_ in log["cand"]:
            if not ok:
                ax.axvline(pt, color="#efb118", lw=1, ls=":")
        for d in log["drifts"]:
            ax.axvline(d, color="#d03b3b", lw=1.2, ls="--")
        ax.set_title(method, fontsize=10, loc="left")
        ax.set_ylabel("D")
    axes[0].plot([], [], color="#2a9d55", label="true drift")
    axes[0].plot([], [], color="#d03b3b", ls="--", label="confirmed drift")
    axes[0].plot([], [], color="#efb118", ls=":", label="rejected candidate")
    axes[0].legend(fontsize=8, ncol=5, loc="upper left")
    axes[-1].set_xlabel("sample index in stream")
    fig.suptitle(f"IF-DLDD drift detection - {name}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=120)
    plt.close(fig)


def run(name):
    loader, window, step = DATASETS[name]
    X, drifts = loader(np.random.default_rng(42))
    nodes = None
    if name == "ns3-iot":
        flows = ns3()[2]
        nodes, times = flows.node.to_numpy(), flows.start.to_numpy()
    grid = [evaluate(ifdldd(X, window, step, threshold=f * window)["drifts"], drifts, len(X), window)
            for f in THRESHOLD_GRID]
    worst = np.nanmax([g["Delay"] for g in grid] + [0])
    thr = THRESHOLD_GRID[int(np.argmax([f1_paper(g, worst) for g in grid]))] * window
    methods = {  # label -> ifdldd kwargs
        f"Detection layer only (thr={thr:g})": dict(threshold=thr, validate=False),
        f"IF-DLDD paper (tuned thr={thr:g})": dict(threshold=thr),
        "IF-DLDD + adaptive thr (ours)": dict(adaptive=True),
    }
    rows, logs = [], {}
    for method, kw in methods.items():
        t0 = time.perf_counter()
        log = ifdldd(X, window, step, nodes=nodes, **kw)
        us = (time.perf_counter() - t0) / len(X) * 1e6
        logs[method] = log
        r = evaluate(log["drifts"], drifts, len(X), tol=window)
        rows.append(dict(dataset=name, method=method, n=len(X), window=window, step=step, **r,
                         candidates=len(log["cand"]), us_per_sample=round(us, 1),
                         model_kB=round(log["model_bytes"] / 1024, 1)))
    max_delay = np.nanmax([r["Delay"] for r in rows] + [0])
    for r in rows:
        r["F1"] = round(f1_paper(r, max_delay), 3)
    plot(name, X, drifts, logs, window)

    print(f"\n=== {name}: n={len(X)}, true drifts at {drifts}")
    for method, log in logs.items():
        print(f"  {method:32s} confirmed drift points: {log['drifts']}")
    if nodes is not None:
        log = logs["IF-DLDD + adaptive thr (ours)"]
        print("  Per-node drift attribution (ours), * = node really attacking in the new concept:")
        attackers = flows[~flows.label.isin(["BENIGN", "ICMP-unreachable"])].groupby("concept").node.unique()
        for d, s in zip(log["drifts"], log["suspects"]):
            truth = set(attackers.get(flows.concept[d:d + window].mode()[0], []))
            top = list(s)[:6]
            tags = ", ".join(f"{n}{'*' * (n in truth)}({s[n]})" for n in top)
            k = min(len(top), len(truth)) or len(top)
            hits = sum(n in truth for n in top[:k])
            print(f"    drift @ record {d} (t = {times[d]:.1f} s) -> suspects: {tags}  "
                  f"[precision@{k} = {hits}/{k}, {len(truth)} attackers]")
    return rows


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    names = sys.argv[1:] or [n for n in DATASETS
                             if (n != "ns3-iot" or (OUT / "flows.csv").exists())
                             and (n != "cic-ids2017" or any(CIC_DIR.glob("*.parquet")))]
    res = pd.DataFrame([r for n in names for r in run(n)])
    res.to_csv(OUT / ("summary_" + "_".join(names) + ".csv" if sys.argv[1:] else "summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n" + res.drop(columns=["n", "window", "step"]).to_string(index=False))
    print(f"\nplots + summary.csv -> {OUT}")
