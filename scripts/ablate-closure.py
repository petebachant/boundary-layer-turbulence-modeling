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
    # Sensitivity rows rather than ablations: these probe the trade-off
    # between streak energy and skin friction. Every route to more
    # pre-transitional k found so far costs c_f, which is what the
    # two-reservoir structure is supposed to avoid.
    ("weaker_cascade", {"Cd": 5.0},
     "stronger shear-gating of the cascade, i.e. less pre-transitional "
     "dissipation"),
    ("stronger_liftup", {"CL_scale": 2.0},
     "double the lift-up coefficient"),
]

# Structural ablations, applied to the solver settings rather than the
# coefficients.
STRUCTURE_ABLATIONS = [
    ("ungated_omega_production", {"gate_omega": False},
     "revert to the ungated strain production alpha*S^2 in the omega equation"),
    ("liftup_on_active_energy", {"liftup_mode": "active"},
     "build the lift-up amplitude on sqrt(gamma*k) instead of sqrt(k), which "
     "is what the OpenFOAM model did while the coefficients were fitted the "
     "other way"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-stride", type=int, default=8)
    ap.add_argument("--coeffs", default="results/clip-k-gamma-coeffs.json")
    ap.add_argument("--out", default="results/closure-ablation.json")
    args = ap.parse_args()

    case = Case(root=".", x_stride=args.x_stride)
    with open(args.coeffs) as f:
        payload = json.load(f)
    fitted = payload["internal_coeffs"]

    # These settings MUST match the ones the coefficients were fitted under
    # (scripts/fit-openfoam-coeffs.py). An ablation run against a differently
    # configured solver measures the configuration difference, not the term
    # being removed -- which is how an earlier run concluded that the lift-up
    # term earns nothing while ablating a version of it that was already an
    # order of magnitude weaker than the fitted one.
    base = dict(param="Rev", p=1.0, local_liftup=True,
                log_layer_consistent=True, freestream_decay=True,
                x_virtual=-201.1, x0=30.2, liftup_mode="total",
                gate_dissipation=False, k_inf=case.kinf_fn())
    base.update(payload.get("structure", {}))

    def score(structure=None, **over):
        kw = dict(fitted)
        # CL_scale multiplies the fitted value rather than replacing it, so the
        # row stays meaningful when the fitted coefficient changes.
        scale = over.pop("CL_scale", None)
        kw.update(over)
        if scale is not None:
            kw["CL"] = kw["CL"] * scale
        kw = {k: v for k, v in kw.items() if k not in ("alpha", "betaStar")}
        b = dict(base)
        if structure:
            b.update(structure)
        res = case.solve(ClipKOmegaGamma(**b, **kw))
        # Score k as well as the mean field. The turbulence energy is the
        # quantity an ablation of a turbulence term should move, and without
        # it the table cannot distinguish a term that does nothing from one
        # whose effect the mean-field metrics cannot see.
        return case.score(res["U"], k=res.get("k"))

    ref = score()
    rows = []
    cases = ([(n, o, d, None) for n, o, d in ABLATIONS]
             + [(n, {}, d, st) for n, st, d in STRUCTURE_ABLATIONS])
    for name, over, desc, structure in cases:
        sc = score(structure=structure, **over)
        rows.append({
            "name": name, "description": desc, "changes": over,
            "structure_changes": structure or {},
            "total": float(sc["total"]),
            "cf_rel_rms": float(sc["cf_rel_rms"]),
            "U_rms": float(sc["U_rms"]),
            "k_log_rms": float(sc.get("k_log_rms", float("nan"))),
            "k_log_rms_pre": float(sc.get("k_log_rms_pre", float("nan"))),
            "delta_total": float(sc["total"] - ref["total"]),
        })
        print(f"{name:26s} total={sc['total']:.4f} cf={sc['cf_rel_rms']:.4f} "
              f"U={sc['U_rms']:.4f} k_pre={sc.get('k_log_rms_pre', float('nan')):.3f} "
              f"delta={sc['total'] - ref['total']:+.4f}", flush=True)

    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"reference": {k: float(v) for k, v in ref.items()},
                   "ablations": rows}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
