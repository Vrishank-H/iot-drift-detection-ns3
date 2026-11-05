"""NS-3 gateway packet log -> CIC-style bidirectional flow records with the 10 features of
Table 9 (Chu et al., IEEE Access 2026). One record per (node, node port, gw port, proto) per
1-second active timeout, ordered by flow start time = the data stream fed to IF-DLDD.

usage: python process_traffic.py results/packets.csv results/flows.csv [T]
"""
import sys

import numpy as np
import pandas as pd

GW = "10.1.1.1"
ACTIVE_TIMEOUT = 1.0  # seconds, flow record is cut at this boundary
FEATURES = ["flow_bytes_s", "flow_pkts_s", "fwd_pkts_s", "flow_iat_mean", "flow_duration",
            "avg_pkt_size", "bwd_pkts_s", "pkt_len_mean", "flow_iat_max", "pkt_len_std"]


def label(dport, proto, concept):
    if proto == 1:
        return "ICMP-unreachable"  # gateway answer to scan / flood on closed ports
    return {5683: "BENIGN", 22: "BruteForce-SSH", 23: "BruteForce-Telnet", 9: "DDoS", 8080: "Bot-C&C",
            80: "DoS" if concept == 3 else "WebAttack"}.get(dport, "PortScan")


def flows(pk, T):
    fwd = pk.src != GW
    pk = pk.assign(fwd=fwd,
                   node=np.where(fwd, pk.src, pk.dst),
                   nport=np.where(fwd, pk.sport, pk.dport),
                   gport=np.where(fwd, pk.dport, pk.sport),
                   epoch=(pk.t // ACTIVE_TIMEOUT).astype(int))
    key = ["node", "nport", "gport", "proto", "epoch"]
    pk = pk.sort_values(key + ["t"])
    pk["iat"] = pk.groupby(key).t.diff()
    g = pk.groupby(key, sort=False)
    f = g.agg(start=("t", "min"), end=("t", "max"), pkts=("t", "size"), fwd=("fwd", "sum"),
              bytes=("bytes", "sum"), size_std=("bytes", "std"),
              iat_mean=("iat", "mean"), iat_max=("iat", "max")).reset_index()
    dur = f.end - f.start
    rate_dur = dur.clip(lower=1e-3)  # single-packet flows: rate over 1 ms
    f["flow_duration"] = dur * 1e6  # microseconds, as in CIC-IDS2017
    f["flow_bytes_s"] = f.bytes / rate_dur
    f["flow_pkts_s"] = f.pkts / rate_dur
    f["fwd_pkts_s"] = f.fwd / rate_dur
    f["bwd_pkts_s"] = (f.pkts - f.fwd) / rate_dur
    f["flow_iat_mean"] = f.iat_mean.fillna(0) * 1e6
    f["flow_iat_max"] = f.iat_max.fillna(0) * 1e6
    f["avg_pkt_size"] = f.bytes / f.pkts
    f["pkt_len_mean"] = f.avg_pkt_size - 28  # L4 payload (IP 20 + UDP 8 header)
    f["pkt_len_std"] = f.size_std.fillna(0)
    f["concept"] = np.minimum(f.start // T, 4).astype(int) + 1
    f["label"] = [label(p, pr, c) for p, pr, c in zip(f.gport, f.proto, f.concept)]
    return f.sort_values("start", kind="stable").reset_index(drop=True)[
        ["start", "node", "nport", "gport", "proto", "concept", "label"] + FEATURES]


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    T = float(sys.argv[3]) if len(sys.argv) > 3 else 100.0
    f = flows(pd.read_csv(src), T)
    f.to_csv(dst, index=False)
    print(f"{len(f)} flow records -> {dst}")
    print(pd.crosstab(f.label, f.concept, margins=True).to_string())
