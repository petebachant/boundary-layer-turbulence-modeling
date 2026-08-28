#!/usr/bin/env python
"""Search structural variants and coefficients for the clipping RANS closure.

Screens candidate closures against the JHTDB transitional-BL DNS using the
fast parabolic solver in pypkg/bl_solver.py, then writes the winning
structure and coefficients for the OpenFOAM implementation to pick up.

Outputs
-------
results/closure-search.json  ranked structural variants and their best coeffs
results/closure-params.json  the single best model (structure + coefficients)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from pypkg.closures import ClipKGamma, Laminar, LaunderSharma
from pypkg.dns_case import Case
from pypkg.search import evaluate, random_search, refine

# Coefficient bounds shared by every structural variant
BOUNDS = {
    "Cmu": (0.2, 1.5),
    "CL": (0.002, 0.3, "log"),
    "CD": (0.03, 0.6, "log"),
    "Cgam": (0.05, 20.0, "log"),
    "Lam_c": (150.0, 900.0),
    "Cl": (0.04, 0.20),
    "Cnu": (0.2, 20.0, "log"),
    "gseed": (1e-4, 0.1, "log"),
    "Cs_cap": (0.05, 1.0),
}

# Structural variants: which local parameter carries the clipping threshold,
# how sharp the rectifier is, and whether a hard stress limiter is imposed.
VARIANTS = [
    {"param": "Rev", "p": 1.0, "a1": 0.0},
    {"param": "Rev", "p": 0.5, "a1": 0.0},
    {"param": "Rev", "p": 2.0, "a1": 0.0},
    {"param": "Rev", "p": 1.0, "a1": 0.137},
    {"param": "Rek", "p": 1.0, "a1": 0.0},
    {"param": "Rek", "p": 1.0, "a1": 0.137},
    {"param": "Rks", "p": 1.0, "a1": 0.0},
    {"param": "Sk", "p": 1.0, "a1": 0.0},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=400)
    ap.add_argument("--n-refine", type=int, default=250)
    ap.add_argument("--x-stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    case = Case(root=".", x_stride=args.x_stride)

    # Baselines for context
    baselines = {}
    for name, closure in [
        ("laminar", Laminar()),
        ("launder-sharma-k-epsilon",
         LaunderSharma(k_inf=case.kinf_fn(), eps_inf=case.epsinf_fn())),
    ]:
        sc = case.score(case.solve(closure)["U"])
        baselines[name] = {k: float(v) for k, v in sc.items()}
        print(f"baseline {name}: total={sc['total']:.4f} "
              f"cf_rel_rms={sc['cf_rel_rms']:.4f}", flush=True)

    ranked = []
    for i, var in enumerate(VARIANTS):
        tag = f"{var['param']}_p{var['p']}" + ("_a1" if var["a1"] else "")
        print(f"\n[{i+1}/{len(VARIANTS)}] variant {tag}", flush=True)
        res = random_search(ClipKGamma, BOUNDS, n=args.n_random,
                            x_stride=args.x_stride, seed=args.seed + i,
                            extra=var, log_every=0)
        best_t, best_c, best_sc = res[0]
        for j, shrink in enumerate([0.25, 0.12, 0.06, 0.03]):
            r2 = refine(ClipKGamma, BOUNDS, best_c, n=args.n_refine,
                        shrink=shrink, x_stride=args.x_stride,
                        seed=1000 + 10 * i + j, extra=var)
            if r2[0][0] < best_t:
                best_t, best_c, best_sc = r2[0]
        print(f"  -> total={best_t:.4f} cf_rel_rms={best_sc.get('cf_rel_rms', float('nan')):.4f}",
              flush=True)
        ranked.append({
            "variant": var, "tag": tag, "total": float(best_t),
            "score": {k: float(v) for k, v in best_sc.items()},
            "coeffs": {k: float(v) for k, v in best_c.items()},
        })

    ranked.sort(key=lambda r: r["total"])
    with open(os.path.join(args.out_dir, "closure-search.json"), "w") as f:
        json.dump({"baselines": baselines, "ranked": ranked}, f, indent=2)
    with open(os.path.join(args.out_dir, "closure-params.json"), "w") as f:
        json.dump(ranked[0], f, indent=2)

    print("\n=== ranking ===")
    for r in ranked:
        print(f"  {r['total']:.4f}  {r['tag']}")
    print(f"\nbest: {ranked[0]['tag']}  total={ranked[0]['total']:.4f}")


if __name__ == "__main__":
    main()
