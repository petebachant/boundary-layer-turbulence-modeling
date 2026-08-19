#!/usr/bin/env python
"""Ablate terms from the fitted closure to find out which ones earn their place.

A model that fits is not evidence that each of its terms matters. This removes
one ingredient at a time and re-scores, which is the cheapest honest test of
whether the central claim -- that a rectified threshold ("clipping") is what
makes transition predictable -- is load-bearing or decorative.

Outputs
-------
results/closure-ablation.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_package.closures import ClipKOmegaGamma
from py_package.dns_case import Case

ABLATIONS = [
    ("reference", {}, "the fitted model"),
    ("no_liftup", {"CL": 0.0},
     "drop the lift-up (streak forcing) production term"),
    ("no_streak_viscous_decay", {"Cnu": 0.0},
     "drop viscous decay of the un-activated energy"),
    ("no_liftup_no_decay", {"CL": 0.0, "Cnu": 0.0},
     "drop both auxiliary terms"),
    ("standard_beta", {"beta": 0.072},
     "use the standard Wilcox omega destruction coefficient"),
    ("classical_threshold", {"Lam_c": 440.0},
     "use the classical critical vorticity Reynolds number instead of the fit"),
    ("no_clip", {"Lam_c": 1.0},
     "REMOVE THE CLIP: threshold so low the source is always active"),
    ("always_active", {"gamma_fs": 1.0, "gseed": 1.0, "Lam_c": 1.0},
     "no activation gating at all, i.e. an ordinary k-omega model"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-stride", type=int, default=8)
    ap.add_argument("--coeffs", default="results/clip-k-gamma-coeffs.json")
    ap.add_argument("--out", default="results/closure-ablation.json")
    args = ap.parse_args()

    case = Case(root=".", x_stride=args.x_stride)
    with open(args.coeffs) as f:
        fitted = json.load(f)["internal_coeffs"]

    base = dict(param="Rev", p=1.0, local_liftup=True,
                log_layer_consistent=True, k_inf=case.kinf_fn())

    def score(**over):
        kw = dict(fitted)
        kw.update(over)
        kw = {k: v for k, v in kw.items() if k not in ("alpha", "betaStar")}
        return case.score(case.solve(ClipKOmegaGamma(**base, **kw))["U"])

    ref = score()
    rows = []
    for name, over, desc in ABLATIONS:
        sc = score(**over)
        rows.append({
            "name": name, "description": desc, "changes": over,
            "total": float(sc["total"]),
            "cf_rel_rms": float(sc["cf_rel_rms"]),
            "U_rms": float(sc["U_rms"]),
            "delta_total": float(sc["total"] - ref["total"]),
        })
        print(f"{name:26s} total={sc['total']:.4f} cf={sc['cf_rel_rms']:.4f} "
              f"U={sc['U_rms']:.4f}  delta={sc['total'] - ref['total']:+.4f}",
              flush=True)

    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"reference": {k: float(v) for k, v in ref.items()},
                   "ablations": rows}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
