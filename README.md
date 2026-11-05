# Lightweight Drift Detection for IoT Nodes Using Isolation Forest in NS-3

Reference paper: R. Chu et al., *"Concept Drift Analysis Based on Isolation Forest for Effectively
Detecting Network Attack in IoT Scenarios"*, IEEE Access 14, 2026 (`RefPaper.pdf`) — **IF-DLDD**.

Pipeline: **NS-3 IoT traffic → flow records → Isolation Forest → sliding-window anomaly count →
candidate drift → t-test verification → confirmed drift point** (+ suspect IoT nodes), on 6 datasets.

## Run

```powershell
powershell -ExecutionPolicy Bypass -File run_demo.ps1            # full: NS-3 sim (~11 min) + detection
powershell -ExecutionPolicy Bypass -File run_demo.ps1 -SkipSim   # reuse results/packets.csv (~2 min)
powershell -ExecutionPolicy Bypass -File run_demo.ps1 -Run 2     # another random traffic realisation
python ifdldd.py ns3-iot sea                                     # selected datasets only
python explain.py ns3-iot                                        # step-by-step evidence of every stage (any dataset)
python test_ifdldd.py                                            # self-check
```

Outputs in `results/`: `<dataset>.png` (anomaly count D, threshold, candidates, confirmed drifts),
`summary.csv`, `flows.csv` (processed IoT stream), `packets.csv` (raw gateway log), `detection_log.txt`.

## Environment (installed / verified)

| Component | Where | Version |
|---|---|---|
| NS-3 (ns-3-dev, wifi/internet/mobility modules) | WSL Ubuntu-22.04, `~/ns-3` | 3.44-dev, g++ 11.4, CMake 3.22 |
| Python + numpy, pandas, scikit-learn, scipy, matplotlib, pyarrow | Windows | 3.12 |
| CIC-IDS2017 MachineLearningCVE (Tue–Fri) | `data/cicids2017/*.parquet` | HF mirror `bvsam/cic-ids-2017` |

The scenario `ns3/iot-drift-sim.cc` is symlinked into `~/ns-3/scratch/` by `run_demo.ps1`.

**CIC-IDS2017 data (~200 MB, not in the repo).** Without it the CIC dataset is skipped. To fetch it (PowerShell):

```powershell
mkdir data\cicids2017 -Force
foreach ($f in "Tuesday-WorkingHours","Wednesday-workingHours","Thursday-WorkingHours-Morning-WebAttacks",
  "Thursday-WorkingHours-Afternoon-Infilteration","Friday-WorkingHours-Morning",
  "Friday-WorkingHours-Afternoon-PortScan","Friday-WorkingHours-Afternoon-DDos") {
  curl.exe -L -o "data\cicids2017\$f.parquet" `
    "https://huggingface.co/datasets/bvsam/cic-ids-2017/resolve/main/machine_learning/$f.pcap_ISCX.csv.parquet"
}
```

Python packages: `pip install numpy pandas scikit-learn scipy matplotlib pyarrow`.

## Files

| File | What |
|---|---|
| `ns3/iot-drift-sim.cc` | 30 WiFi (802.11n) IoT sensors + gateway AP. 5 concepts × 100 s mirroring the paper's Table 10: C1 benign CoAP-like telemetry → C2 + SSH/Telnet brute force → C3 + DoS Hulk/Slowloris → C4 + small web attack → C5 + botnet (port scan, DDoS flood, C&C). Gateway logs every IP packet. |
| `process_traffic.py` | Packet log → bidirectional flow records (1 s active timeout) with the 10 Table 9 features (Flow Bytes/s, Flow Packets/s, Fwd/Bwd Packets/s, Flow IAT Mean/Max, Flow Duration, Avg Packet Size, Packet Length Mean/Std). |
| `ifdldd.py` | IF-DLDD + extensions, datasets, metrics, plots. |

## Algorithm (paper → code, `ifdldd.py`)

1. **Isolation Forest** (50 trees, ψ = 256) fitted on reference window WS1; score s(x) = 2^(−E[h(x)]/c(ψ)) (eq. 1).
2. **Sliding-window anomaly count** D = #{x ∈ WS2 : s(x) > τ = 0.6} (eq. 2).
3. **Candidate drift** when D_k − D_(k−1) > threshold; otherwise WS1 = [WS1 · WS2] (Algorithm 2).
4. **t-test verification**: candidate window split at every m, one-sided Welch t (eq. 3) with ν (eq. 4); max t, α = 0.01 (Algorithm 3).
5. **Confirmed drift point** = split index m; reference re-fitted on the new concept.

Metrics: TP / FP / FN, average delay (eq. 5) and the paper's modified F1 (eq. 6–7, (α, β, γ) = (0.6, 1, 0.3)).

## Our additions over the paper

1. **Self-calibrating detection threshold** — the paper hand-tunes `threshold` per dataset (its stated limitation).
   We set threshold_k = max(0.02·window, median + 3·std of recent D_k − D_(k−1) in the current concept), so it
   adapts to each stream with **no tuning**. The paper baseline below is given its *best* threshold from a grid
   {0.01, 0.02, 0.05, 0.1}×window (the paper's own protocol).
2. **Per-node drift attribution** — on every confirmed drift, IoT nodes are ranked by their anomalous flow records
   after the drift point, telling the operator *which devices* changed behaviour (likely compromised).
3. **Lightweight IoT deployment**: bounded reference memory (≤ 4 windows), 50-tree forest (~0.5–0.9 MB),
   label-free; per-record cost reported (`us_per_sample`).
4. NS-3 IoT network simulation as a new evaluation source (the paper only used offline datasets).

## Results (`results/summary.csv`, seed 42, NS-3 run 1)

| Dataset | Method | TP | FP | FN | Delay | F1 |
|---|---|---|---|---|---|---|
| **NS-3 IoT** (31 485 flows, 4 drifts) | Detection layer only | 3 | 3 | 1 | 0.3 | 0.706 |
| | IF-DLDD paper (tuned thr) | 4 | 1 | 0 | 0 | 0.964 |
| | **IF-DLDD + adaptive (ours)** | 4 | 1 | 0 | 0 | 0.964 |
| CIC-IDS2017 (Table 10 stream) | Detection layer only | 3 | 4 | 1 | 1233 | 0.682 |
| | IF-DLDD paper (tuned thr) | 3 | 2 | 1 | 0.3 | 0.789 |
| | **ours** | 3 | 0 | 1 | 0 | **0.857** |
| SEA (10 % noise) | paper (tuned) / **ours** | 2 / 2 | 1 / 0 | 1 / 1 | 51 / 0 | 0.754 / **0.800** |
| Stagger | paper (tuned) / ours | 2 / 2 | 0 / 1 | 0 / 0 | 0.5 / 1.5 | **1.000** / 0.930 |
| Sine1, Sine2 | paper (tuned) / ours | 4 / 4 | 0 / 0 | 0 / 0 | 0 / 0 | 1.000 / 1.000 |

- NS-3 drift points are located at exactly t = 100, 200, 300, 400 s; one false alarm inside the DoS phase (t ≈ 247 s).
- The t-test verification layer removes most false alarms of the detection layer alone (FP 3→1, 4→2/0).
- Without any tuning the adaptive threshold matches or beats the hand-tuned paper threshold on 5 of 6 datasets.
- Per-node attribution on NS-3 (precision@k vs. the real attacking nodes): C2 6/6, C3 6/6, C4 2/2, C5 6/6.
- Hard cases: SEA drifts only in the label (8 % of points change class), and CIC C1→C2 (Patator flows resemble
  benign flows) — the paper also reports large delays there.
