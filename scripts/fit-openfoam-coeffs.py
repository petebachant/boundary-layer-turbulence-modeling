#!/usr/bin/env python
"""Fit the portable k-omega-gamma closure and emit OpenFOAM coefficients.

The structural search in discover-closure.py fits ClipKGamma, which uses an
algebraic mixing length. The OpenFOAM model (clipKGamma) instead transports
omega, because a boundary-layer thickness is non-local and fragile in a
general CFD code. Those are different models, so coefficients fitted for one
must not be reused for the other. This script fits the transported-omega form
directly and writes the result under the names the OpenFOAM dictionary uses.

Outputs
-------
results/clip-k-gamma-coeffs.json   coefficients for constant/turbulenceProperties
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_package.closures import ClipKOmegaGamma
from py_package.dns_case import Case
from py_package.search import random_search, refine

BOUNDS = {
    "CL": (0.002, 0.3, "log"),
    "Cgam": (0.05, 200.0, "log"),
    "Lam_c": (150.0, 900.0),
    "Cnu": (0.2, 20.0, "log"),
    "gseed": (1e-4, 0.1, "log"),
    "Cs_cap": (0.05, 1.0),
    "alpha": (0.3, 0.8),
    "beta": (0.04, 0.11),
    "betaStar": (0.05, 0.14),
    "omega_fs_scale": (1.0, 60.0, "log"),
}

# Map internal names to the OpenFOAM dictionary entries
OF_NAMES = {
    "alpha": "alphaOmega",
    "beta": "beta",
    "betaStar": "betaStar",
    "CL": "CL",
    "Cgam": "Cgam",
    "Lam_c": "LambdaC",
    "Cnu": "Cnu",
    "Cs_cap": "Cs",
    "gseed": "gseed",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=500)
    ap.add_argument("--n-refine", type=int, default=250)
    ap.add_argument("--x-stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="results/clip-k-gamma-coeffs.json")
    args = ap.parse_args()

    case = Case(root=".", x_stride=args.x_stride)
    extra = {"param": "Rev", "p": 1.0, "local_liftup": True}

    # Reference: the defaults currently baked into run.py
    from py_package.search import evaluate
    ref, ref_sc = evaluate(ClipKOmegaGamma, {}, ".", args.x_stride, extra)
    print(f"defaults: total={ref:.4f} cf_rel_rms={ref_sc.get('cf_rel_rms', float('nan')):.4f}",
          flush=True)

    res = random_search(ClipKOmegaGamma, BOUNDS, n=args.n_random,
                        x_stride=args.x_stride, seed=args.seed, extra=extra,
                        log_every=100)
    best_t, best_c, best_sc = res[0]
    for i, shrink in enumerate([0.25, 0.12, 0.06, 0.03]):
        r2 = refine(ClipKOmegaGamma, BOUNDS, best_c, n=args.n_refine,
                    shrink=shrink, x_stride=args.x_stride, seed=200 + i,
                    extra=extra)
        if r2[0][0] < best_t:
            best_t, best_c, best_sc = r2[0]
        print(f"  refine {i} -> {best_t:.4f}", flush=True)

    of = {OF_NAMES[k]: float(v) for k, v in best_c.items() if k in OF_NAMES}
    of.update({"pExp": 1.0, "a1": 0.0, "c1": 10.0, "gammaFs": 0.02,
               "sigmak_ko": 2.0, "sigmaOmega": 2.0, "sigmaGamma": 1.0})

    payload = {
        "model": "clipKGamma (k-omega-gamma, transported length scale)",
        "score": {k: float(v) for k, v in best_sc.items()},
        "defaults_score": {k: float(v) for k, v in ref_sc.items()},
        "internal_coeffs": {k: float(v) for k, v in best_c.items()},
        "openfoam_coeffs": of,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nfitted: total={best_t:.4f} cf_rel_rms={best_sc['cf_rel_rms']:.4f} "
          f"U_rms={best_sc['U_rms']:.4f}  (defaults were {ref:.4f})")
    print(json.dumps(of, indent=2))


if __name__ == "__main__":
    main()
