#!/usr/bin/env python
"""Can a closure that carries the fluctuations' exergy close its
destruction with what it knows?

A model that transports X = k - 3/2 (det R)^(1/3) alongside k and omega
gets its source exactly, P_k (results/channel-exergy-budget.json), but has
to model its destruction. In the channels nearly all of it is inviscid,
by the velocity-pressure-gradient term, so the target here is that
destruction, D = -(pressure-strain + pressure transport contributions),
per unit dissipation. A two-equation model with X does not know the
anisotropy tensor, only the exergetic fraction f = X / k and the ratio
p = P_k / epsilon, so the candidate law is D / epsilon = F(f, p), from the
library

    1, f, f^2, f^3, p, p f, p f^2, p^2, p^2 f,

fitted by STLSQ from PySINDy on normalized columns, bagged as in
scripts/fit-transport-equations.py (each fit draws equally from every
channel; a term is kept if at least SELECTED of the fits keep it, at its
median coefficient). Points with y+ below Y_PLUS_MIN are left out, where
the weighting of the exergy budget is singular.

Reference: the destruction implied by Launder, Reece and Rodi's
isotropization-of-production model with its published coefficients,
Pi = -C1 epsilon b - C2 (P - 2/3 P_k I), which in the exergy budget is

    D = C1 epsilon (q / 6 - 3 lambda_g / (4 k)) + C2 P_k q / 3,

with lambda_g = (det R)^(1/3) and q = lambda_g tr(R^-1). It needs the
anisotropy, so it is an a-priori reference, not a closure.

Test, fixed before any fit: leaving each channel out in turn, the law
fitted on the other four predicts the held-out channel's integrated
destruction (y+ from Y_PLUS_MIN to the centerline) within INTEGRAL_TOL
and its profile with R^2 of at least R2_MIN, for every channel. Also
reported: the pooled fit, and the same held-out measures for the
reference with its published coefficients.

Outputs
-------
results/exergy-destruction-fit.json
"""

from __future__ import annotations

import json

import numpy as np
import pysindy as ps

OUT = "results/exergy-destruction-fit.json"
BUDGET = "results/channel-exergy-budget.json"
TERMS = ("1", "f", "f2", "f3", "p", "pf", "pf2", "p2", "p2f")
Y_PLUS_MIN = 5.0
N_BOOT = 100
THRESHOLD = 0.05
RIDGE = 1e-3
SELECTED = 0.8
INTEGRAL_TOL = 0.10
R2_MIN = 0.8
C1, C2 = 1.8, 0.6
SEED = 0


def channel(prof):
    y = np.array(prof["y_plus"])
    m = y >= Y_PLUS_MIN
    k = np.array(prof["k"])[m]
    X = np.array(prof["exergy"])[m]
    eps = np.array(prof["dissipation_k"])[m]
    pk = np.array(prof["production_k"])[m]
    q = np.array(prof["lam_tr_rinv"])[m]
    D = -(
        np.array(prof["pressure_strain"]) + np.array(prof["pressure_transport"])
    )[m]
    f, p = X / k, pk / eps
    theta = np.c_[
        np.ones_like(f), f, f**2, f**3, p, p * f, p * f**2, p**2, p**2 * f
    ]
    lam = (k - X) / 1.5
    ref = C1 * eps * (q / 6.0 - 3.0 * lam / (4.0 * k)) + C2 * pk * q / 3.0
    return {
        "y": y[m],
        "theta": theta,
        "t": D / eps,
        "D": D,
        "eps": eps,
        "ref": ref,
    }


def bagged_fit(chans, rng):
    n = min(len(c["t"]) for c in chans)
    coefs = []
    for _ in range(N_BOOT):
        picks = [rng.choice(len(c["t"]), n) for c in chans]
        A = np.vstack([c["theta"][i] for c, i in zip(chans, picks)])
        b = np.concatenate([c["t"][i] for c, i in zip(chans, picks)])
        opt = ps.STLSQ(threshold=THRESHOLD, alpha=RIDGE, normalize_columns=True)
        opt.fit(A, b)
        coefs.append(np.ravel(opt.coef_))
    C = np.array(coefs)
    keep = (C != 0).mean(axis=0) >= SELECTED
    coef = np.array(
        [
            np.median(C[C[:, i] != 0, i]) if keep[i] else 0.0
            for i in range(len(TERMS))
        ]
    )
    return {
        "inclusion": dict(zip(TERMS, (C != 0).mean(axis=0).round(3).tolist())),
        "coefficients": dict(zip(TERMS, coef.tolist())),
    }, coef


def scores(c, pred):
    """Integrated-destruction error and profile R^2 of a predicted D."""
    integ = float(
        np.trapezoid(pred, c["y"]) / np.trapezoid(c["D"], c["y"]) - 1.0
    )
    r2 = float(
        1.0
        - np.sum((pred - c["D"]) ** 2) / np.sum((c["D"] - c["D"].mean()) ** 2)
    )
    return {"integral_error": integ, "r2": r2}


def main():
    rng = np.random.default_rng(SEED)
    with open(BUDGET) as f:
        bud = json.load(f)["channels"]
    names = sorted(bud, key=lambda s: int(s.split("_")[-1]))
    chans = {n: channel(bud[n]["profiles"]) for n in names}
    pooled, _ = bagged_fit(list(chans.values()), rng)
    held = {}
    for n in names:
        fit, coef = bagged_fit([chans[m] for m in names if m != n], rng)
        c = chans[n]
        held[n] = {
            **fit,
            "law": scores(c, c["eps"] * (c["theta"] @ coef)),
            "reference": scores(c, c["ref"]),
        }
    passes = all(
        abs(h["law"]["integral_error"]) <= INTEGRAL_TOL
        and h["law"]["r2"] >= R2_MIN
        for h in held.values()
    )
    result = {
        "integral_tol": INTEGRAL_TOL,
        "r2_min": R2_MIN,
        "y_plus_min": Y_PLUS_MIN,
        "pooled": pooled,
        "held_out": held,
        "n_terms": len(TERMS),
        "n_kept_pooled": int(
            sum(v != 0 for v in pooled["coefficients"].values())
        ),
        "worst_integral_error": max(
            abs(h["law"]["integral_error"]) for h in held.values()
        ),
        "worst_r2": min(h["law"]["r2"] for h in held.values()),
        "passes": bool(passes),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        "pooled",
        {k: round(v, 3) for k, v in pooled["coefficients"].items() if v},
    )
    for n, h in held.items():
        print(
            n,
            "law",
            h["law"],
            "ref",
            h["reference"],
            {k: round(v, 3) for k, v in h["coefficients"].items() if v},
        )
    print("passes", passes)


if __name__ == "__main__":
    main()
