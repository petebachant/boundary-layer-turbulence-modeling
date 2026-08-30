#!/usr/bin/env python
"""Measure the tier-1 -> tier-2 discrepancy on the calibration plate.

The tiers are fidelities: the parabolic marching solver is the cheap one,
the elliptic OpenFOAM solve the one of record, and every closure that has
run on the plate in both gives one (cheap, expensive) pair. This stage
collects those pairs and characterizes the bias, which is the ingredient a
multi-fidelity optimization needs before it can decide when a fast-tier
evaluation is trustworthy and when an OpenFOAM one is worth its cost
(ideas-log 7.4).

Three honest limitations, recorded in the output rather than hidden: the
two tiers report slightly different skin-friction metrics (relative RMS
against the mean relative error); only a handful of closures exist in both
tiers, so this is a characterization, not a fitted surrogate; and the
pairing maps each Python closure to the OpenFOAM model that implements the
same equations, which for the clipping closure means tier-1
clip-k-omega-gamma against the OpenFOAM clipKGamma run. The
exact-formulation fast-tier cases (channel, temporal mixing layer) have no
tier-2 counterpart because they need none: their fast-tier solution is the
equation, not an approximation to it.

Outputs
-------
results/tier-discrepancy.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: tier-1 closure name -> (tier-2 label, results file that holds it). The
#: matched DNS domain is preferred; k-epsilon exists only in the
#: wall-resolved comparison.
PAIRS = {
    "clip-k-omega-gamma": ("clip-k-gamma", "results/dns-domain-vs-dns.json"),
    "laminar": ("laminar", "results/dns-domain-vs-dns.json"),
    "launder-sharma": ("k-epsilon", "results/openfoam-vs-dns.json"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tier-discrepancy.json")
    args = ap.parse_args()

    with open("results/benchmark.json") as f:
        tier1 = json.load(f)["results"]["jhtdb-transitional-bl"]
    pairs = []
    for t1_name, (t2_name, path) in PAIRS.items():
        with open(path) as f:
            t2 = json.load(f)[t2_name]
        cheap = tier1[t1_name]["cf_rel_rms"]
        expensive = t2["cf_rel_err_mean"]
        pairs.append({
            "closure": t1_name,
            "tier2_label": t2_name,
            "tier2_source": path,
            "cf_tier1": cheap,
            "cf_tier2": expensive,
            "ratio_tier2_over_tier1": expensive / cheap,
            "U_tier1": tier1[t1_name].get("U_rms"),
            "U_tier2": t2.get("U_rel_rms_mean"),
        })
    logs = [math.log(p["ratio_tier2_over_tier1"]) for p in pairs]
    mean = sum(logs) / len(logs)
    spread = (sum((v - mean) ** 2 for v in logs) / len(logs)) ** 0.5
    # Rank agreement is what a screening fidelity is for: does tier 1 order
    # the closures as tier 2 does?
    r1 = sorted(pairs, key=lambda p: p["cf_tier1"])
    r2 = sorted(pairs, key=lambda p: p["cf_tier2"])
    out = {
        "pairs": pairs,
        "n_pairs": len(pairs),
        "cf_bias_geometric_mean": math.exp(mean),
        "cf_bias_log_sd": spread,
        "ranking_tier1": [p["closure"] for p in r1],
        "ranking_tier2": [p["closure"] for p in r2],
        "rankings_agree": [p["closure"] for p in r1]
        == [p["closure"] for p in r2],
        "notes": (
            "Metrics differ between tiers (rel RMS against mean rel err); "
            "n is small; this characterizes the screening bias rather than "
            "fitting a surrogate. Exact-formulation fast-tier cases have no "
            "tier-2 counterpart by design."
        ),
    }
    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    for p in pairs:
        print(f"  {p['closure']:22s} cf tier1 {p['cf_tier1']:.3f} -> "
              f"tier2 {p['cf_tier2']:.3f}  ratio {p['ratio_tier2_over_tier1']:.2f}")
    print(f"geometric-mean bias {out['cf_bias_geometric_mean']:.2f} "
          f"(log-sd {spread:.2f}); rankings agree: {out['rankings_agree']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
