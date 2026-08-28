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

from pypkg.closures import ClipKOmegaGamma
from pypkg.dns_case import Case
from pypkg.search import random_search, refine

# alpha is NOT fitted: it is derived from beta, betaStar and sigma_omega by
# the log-layer constraint (see ClipKOmegaGamma.log_layer_consistent).
# betaStar is held at its standard value, since it fixes the k-omega relation
# eps = betaStar*k*omega that the rest of the calibration rests on.
BOUNDS = {
    "CL": (0.005, 0.6, "log"),
    "Cgam": (1.0, 2000.0, "log"),
    "Lam_c": (150.0, 900.0),
    "Cnu": (0.01, 50.0, "log"),
    "gseed": (1e-4, 0.2, "log"),
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
    "Cs_cap": (0.05, 50.0, "log"),
    # Narrowed after a scan: with the free-stream decay constrained, beta is
    # optimal near 0.045 both with and without the constraint. The earlier
    # preference for 0.09-0.1 was an artifact of leaving the free-stream omega
    # a free parameter, which let the model compensate for excess free-stream
    # turbulence instead of decaying it correctly.
    "beta": (0.030, 0.065),
    # Shear-gated dissipation: suppresses the cascade where mean shear
    # organises the fluctuations, without touching the isotropic free stream.
    #
    # Bound raised from 8. The DNS fluctuation-energy budget has dissipation at
    # about 0.66 of production through the pre-transitional region, while the
    # fitted model runs at 2.0 there and the streak energy therefore decays
    # where the DNS grows (scripts/analyze-streak-reservoir.py). Suppressing
    # the pre-transitional cascade is the term that can close that gap, and 8
    # was not enough room to find out.
    "Cd": (0.0, 60.0),
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
    "Cd": "Cd",
    "gseed_omega": "gseedOmega",
}

# Structural variants of the omega equation.
#
# The strain-based production alpha*S^2 is the SST substitution for the
# textbook alpha*(omega/k)*P, and the two agree only when nu_t = k/omega. This
# closure gates the eddy viscosity, nu_t = gamma*k/omega, so the equivalent
# strain form carries a gamma. Left ungated, omega is driven up by the mean
# shear in a region that carries no turbulence, the streak energy is
# dissipated at the turbulent rate, and the pre-transitional reservoir the
# whole model rests on is emptied -- which is what the DNS comparison in
# scripts/analyze-streak-reservoir.py shows was happening.
#
# Each variant gets its own coefficient fit, so structures are compared after
# their inner fit rather than one structure with good constants against
# another with the wrong ones.
# The lift-up viscosity is the other structural axis. The DNS pre-transitional
# eddy viscosity is reproduced to within 1 % at every station by
# nu = C*sqrt(k)*ell with ell = 0.14*delta99 -- a strikingly constant fraction
# of the boundary-layer thickness. delta99 is not available to a local closure,
# so the two candidates are a wall-distance-limited mixing length (algebraic
# streak growth) and k/omega (exponential growth).
VARIANTS = [
    {"name": "ungated-omega",
     "extra": {"gate_omega": False, "liftup_form": "mixing",
               "liftup_gate": False},
     "bounds": {},
     "of": {"omegaGating": "none", "liftupForm": "mixing",
            "liftupGate": "false"}},
    {"name": "exact-gated-omega",
     "extra": {"gate_omega": "exact", "liftup_form": "mixing",
               "liftup_gate": False},
     "bounds": {},
     "of": {"omegaGating": "exact", "liftupForm": "mixing",
            "liftupGate": "false"}},
    {"name": "exact-gated-omega-gated-liftup",
     "extra": {"gate_omega": "exact", "liftup_form": "mixing",
               "liftup_gate": True},
     "bounds": {"CL": (0.005, 3.0, "log")},
     "of": {"omegaGating": "exact", "liftupForm": "mixing",
            "liftupGate": "true"}},
    {"name": "exact-gated-omega-komega-gated-liftup",
     "extra": {"gate_omega": "exact", "liftup_form": "komega",
               "liftup_gate": True},
     "bounds": {"CL": (0.001, 0.5, "log")},
     "of": {"omegaGating": "exact", "liftupForm": "komega",
            "liftupGate": "true"}},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-random", type=int, default=500)
    ap.add_argument("--n-refine", type=int, default=250)
    ap.add_argument("--x-stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="results/clip-k-gamma-coeffs.json")
    ap.add_argument("--variant", default=None,
                    help="fit only this structural variant (default: all)")
    args = ap.parse_args()

    case = Case(root=".", x_stride=args.x_stride)
    # freestream_decay makes the model generate its own free-stream boundary
    # values, so the fit feels whether beta reproduces the measured DNS decay.
    # Without it the solver is handed the correct free stream for free, and
    # beta drifts to a value that floods the boundary layer far downstream.
    extra = {"param": "Rev", "p": 1.0, "local_liftup": True,
             "log_layer_consistent": True, "freestream_decay": True,
             "x_virtual": -201.1, "x0": 30.2,
             "liftup_mode": "total", "gate_dissipation": False}

    # Reference: the defaults currently baked into run.py
    from pypkg.search import evaluate
    ref, ref_sc = evaluate(ClipKOmegaGamma, {}, ".", args.x_stride, extra)
    print(f"defaults: total={ref:.4f} cf_rel_rms={ref_sc.get('cf_rel_rms', float('nan')):.4f}",
          flush=True)

    # Outer loop over structure, inner loop over coefficients.
    trials = []
    for v in VARIANTS:
        if args.variant and v["name"] != args.variant:
            continue
        ex = dict(extra)
        ex.update(v["extra"])
        B = dict(BOUNDS)
        B.update(v["bounds"])
        # Cs only enters through the mixing-form length cap, so fitting it
        # under the komega form would search a dimension the model cannot see.
        if v["extra"].get("liftup_form") == "komega":
            B.pop("Cs_cap", None)
        print(f"\n--- {v['name']} ---", flush=True)
        res = random_search(ClipKOmegaGamma, B, n=args.n_random,
                            x_stride=args.x_stride, seed=args.seed, extra=ex,
                            log_every=200)
        t, c, sc = res[0]
        for i, shrink in enumerate([0.25, 0.12, 0.06, 0.03]):
            r2 = refine(ClipKOmegaGamma, B, c, n=args.n_refine, shrink=shrink,
                        x_stride=args.x_stride, seed=200 + i, extra=ex)
            if r2[0][0] < t:
                t, c, sc = r2[0]
            print(f"  refine {i} -> {t:.4f}", flush=True)
        trials.append({"name": v["name"], "total": float(t),
                       "score": {k: float(x) for k, x in sc.items()},
                       "coeffs": {k: float(x) for k, x in c.items()},
                       "extra": v["extra"], "of": v["of"]})

    if not trials:
        raise SystemExit("no structural variant selected")
    trials.sort(key=lambda d: d["total"])
    winner = trials[0]
    best_t, best_c, best_sc = winner["total"], winner["coeffs"], winner["score"]
    print(f"\nstructure ranking:", flush=True)
    for tr in trials:
        print(f"  {tr['name']:<20} total={tr['total']:8.3f} "
              f"cf={tr['score'].get('cf_rel_rms', float('nan')):.4f} "
              f"k_pre={tr['score'].get('k_log_rms_pre', float('nan')):.3f} "
              f"Lam_c={tr['coeffs'].get('Lam_c', float('nan')):.0f}", flush=True)

    # Recover the derived alpha for the OpenFOAM dictionary
    import numpy as np
    kappa, sigmaw, betaStar = 0.41, 2.0, 0.09
    best_c = dict(best_c)
    best_c["alpha"] = (best_c["beta"] / betaStar
                       - kappa ** 2 / (sigmaw * np.sqrt(betaStar)))
    best_c["betaStar"] = betaStar
    of = {OF_NAMES[k]: float(v) for k, v in best_c.items() if k in OF_NAMES}

    # Guard: every fitted coefficient must reach OpenFOAM. A fitted parameter
    # that silently fails to cross the boundary is not a small bug -- it means
    # the elliptic solver runs a different model from the one that was
    # calibrated. This has already happened twice (omega_fs_scale, Cd).
    unmapped = sorted(k for k in best_c
                      if k not in OF_NAMES and k not in ("alpha", "betaStar"))
    if unmapped:
        raise SystemExit(
            "fitted coefficients with no OpenFOAM mapping: "
            + ", ".join(unmapped)
            + "\nAdd them to OF_NAMES and to the model, or the elliptic run "
              "will silently use defaults."
        )

    # Free-stream inlet values from the SAME decay law and the same beta,
    # evaluated at the OpenFOAM inlet patch. These are boundary conditions,
    # not model coefficients; failing to carry them across is what made the
    # elliptic solution disagree with the screening solver.
    x_inlet, x_virtual, x0 = -118.0, -201.1, 30.2
    Ue = float(case.Ue.mean())
    k0 = float(case.kinf_fn()(x0))
    bfit = best_c["beta"]
    w0 = Ue / (bfit * (x0 - x_virtual))
    scale = 1.0 + bfit * w0 * (x_inlet - x0) / Ue
    inlet = {"x_inlet": x_inlet,
             "k_inlet": float(k0 * scale ** (-betaStar / bfit)),
             "omega_inlet": float(w0 / scale)}
    of.update({"pExp": 1.0, "a1": 0.0, "c1": 10.0, "gammaFs": 0.02,
               "sigmak_ko": 2.0, "sigmaOmega": 2.0, "sigmaGamma": 1.0})
    of.update(winner["of"])

    payload = {
        "model": "clipKGamma (k-omega-gamma, transported length scale)",
        "structure_name": winner["name"],
        "structure": winner["extra"],
        "structure_ranking": [
            {"name": t["name"], "total": t["total"], "score": t["score"],
             "coeffs": t["coeffs"]} for t in trials
        ],
        "freestream_inlet": inlet,
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
