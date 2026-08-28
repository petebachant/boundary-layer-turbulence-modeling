#!/usr/bin/env python
"""Measure how much of a structure comparison is search noise.

The PDE-discovery method used in this project is two nested loops: an outer
loop over discrete model structure, an inner loop fitting that structure's
coefficients, with structures ranked by the best error their inner fit reaches.
That ranking is only meaningful if the inner fit is converged enough that
re-running it does not move the score by more than the gaps being ranked.

This script tests exactly that. Each structural variant is fitted several
times under an identical protocol, changing nothing but the random seed, and
the spread of the resulting scores is reported alongside the differences
between structures. If the within-structure spread is comparable to the
between-structure gaps, the ranking is not evidence and neither are any
conclusions drawn from it.

Outputs
-------
results/fit-noise.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypkg.closures import ClipKOmegaGamma
from pypkg.search import random_search, refine

# Coefficients common to every variant. Kept identical to
# scripts/fit-openfoam-coeffs.py so the spread measured here applies to the
# numbers that script produces.
COMMON_BOUNDS = {
    "Cgam": (1.0, 2000.0, "log"),
    "Lam_c": (150.0, 900.0),
    "Cnu": (0.01, 50.0, "log"),
    "gseed": (1e-4, 0.2, "log"),
    "beta": (0.030, 0.065),
    "Cd": (0.0, 60.0),
}

# Upper bound raised from 2.0. Cs sets where the length cap
# Cs*sqrt(k)/omega takes over from the wall distance y. With the old
# bound the cap sat near 0.5 and therefore bound EVERYWHERE in the
# pre-transitional layer, which forces nu_L = CL*sqrt(k)*Cs*sqrt(k)/omega
# -- proportional to k, not to sqrt(k). That welds the streak energy to
# the momentum transport it produces and defeats the two-reservoir
# behaviour the closure exists to represent (see clipping-closure.md
# 4.6). Allowing Cs above about 5 lets y bind at the production peak,
# restoring the sqrt(k) scaling the DNS shows, while the cap still limits
# nu_L in the free stream.
VARIANTS = [
    ("mixing",
     {"liftup_form": "mixing", "liftup_gate": False},
     {"CL": (0.005, 0.6, "log"), "Cs_cap": (0.05, 50.0, "log")}),
    ("mixing+gate",
     {"liftup_form": "mixing", "liftup_gate": True},
     {"CL": (0.005, 3.0, "log"), "Cs_cap": (0.05, 50.0, "log")}),
    ("komega+gate",
     {"liftup_form": "komega", "liftup_gate": True},
     {"CL": (0.001, 0.5, "log")}),
]

BASE = {"param": "Rev", "p": 1.0, "local_liftup": True,
        "log_layer_consistent": True, "freestream_decay": True,
        "x_virtual": -201.1, "x0": 30.2, "liftup_mode": "total",
        "gate_dissipation": False, "gate_omega": "exact"}


def fit_once(extra, bounds, seed, n_random, n_refine, x_stride):
    res = random_search(ClipKOmegaGamma, bounds, n=n_random, x_stride=x_stride,
                        seed=seed, extra=extra, log_every=0)
    total, coeffs, score = res[0]
    for i, shrink in enumerate([0.25, 0.12, 0.06, 0.03]):
        r2 = refine(ClipKOmegaGamma, bounds, coeffs, n=n_refine, shrink=shrink,
                    x_stride=x_stride, seed=seed * 10 + i, extra=extra)
        if r2[0][0] < total:
            total, coeffs, score = r2[0]
    return float(total), coeffs, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37, 51])
    ap.add_argument("--n-random", type=int, default=500)
    ap.add_argument("--n-refine", type=int, default=250)
    ap.add_argument("--x-stride", type=int, default=8)
    ap.add_argument("--out", default="results/fit-noise.json")
    args = ap.parse_args()

    rows = []
    for name, ex, bnd in VARIANTS:
        extra = dict(BASE)
        extra.update(ex)
        bounds = dict(COMMON_BOUNDS)
        bounds.update(bnd)
        # Cs only enters through the mixing-form length cap, so fitting it
        # under the komega form would search a dimension the model cannot see.
        if ex.get("liftup_form") == "komega":
            bounds.pop("Cs_cap", None)
        for seed in args.seeds:
            total, coeffs, score = fit_once(extra, bounds, seed, args.n_random,
                                            args.n_refine, args.x_stride)
            rows.append({
                "variant": name, "seed": seed, "total": total,
                "Lam_c": float(coeffs["Lam_c"]),
                "cf_rel_rms": float(score["cf_rel_rms"]),
                "k_log_rms_pre": float(score["k_log_rms_pre"]),
                "coeffs": {k: float(v) for k, v in coeffs.items()},
            })
            print(f"{name:<12} seed {seed:3d} total {total:8.3f} "
                  f"Lam_c {coeffs['Lam_c']:5.0f} "
                  f"cf {score['cf_rel_rms']:.4f} "
                  f"k_pre {score['k_log_rms_pre']:.3f}", flush=True)

    def stats(values):
        return {
            "mean": float(statistics.fmean(values)),
            "sd": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            "min": float(min(values)), "max": float(max(values)),
        }

    summary = {}
    for name, _, _ in VARIANTS:
        sel = [r for r in rows if r["variant"] == name]
        summary[name] = {
            "n": len(sel),
            "total": stats([r["total"] for r in sel]),
            "Lam_c": stats([r["Lam_c"] for r in sel]),
            "cf_rel_rms": stats([r["cf_rel_rms"] for r in sel]),
            "k_log_rms_pre": stats([r["k_log_rms_pre"] for r in sel]),
        }
        summary[name]["total"]["se"] = (
            summary[name]["total"]["sd"] / max(len(sel), 1) ** 0.5)

    names = [n for n, _, _ in VARIANTS]
    gaps = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = summary[a]["total"]["mean"] - summary[b]["total"]["mean"]
            se = (summary[a]["total"]["se"] ** 2
                  + summary[b]["total"]["se"] ** 2) ** 0.5
            gaps.append({"a": a, "b": b, "difference": float(d),
                         "combined_se": float(se),
                         "sigmas": float(abs(d) / se) if se > 0 else 0.0,
                         "significant_at_2se": bool(se > 0 and abs(d) > 2 * se)})

    payload = {
        "note": ("Within-structure spread of the inner coefficient fit, "
                 "against the between-structure gaps it is used to rank. "
                 "Only the random seed changes between repeats."),
        "protocol": {"n_random": args.n_random, "n_refine": args.n_refine,
                     "refine_shrinks": [0.25, 0.12, 0.06, 0.03],
                     "seeds": args.seeds, "x_stride": args.x_stride},
        "runs": rows,
        "summary": summary,
        "pairwise": gaps,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print()
    for name in names:
        s = summary[name]
        print(f"{name:<12} n={s['n']} total {s['total']['mean']:7.3f} "
              f"+- {s['total']['sd']:5.3f} "
              f"[{s['total']['min']:7.3f}, {s['total']['max']:7.3f}]  "
              f"Lam_c {s['Lam_c']['mean']:5.0f} +- {s['Lam_c']['sd']:4.0f}")
    print()
    for g in gaps:
        verdict = "SIGNIFICANT" if g["significant_at_2se"] else "not significant"
        print(f"{g['a']} vs {g['b']}: {g['difference']:+.3f} "
              f"({g['sigmas']:.1f} combined SE) -- {verdict}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
