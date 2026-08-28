#!/usr/bin/env python
"""A-priori least-squares regression of closure terms against the DNS.

The original idea behind this project: rather than proposing a closure and
scoring it after a forward solve, compute a library of candidate terms
directly from the data and solve for their coefficients in a least-squares
sense across the whole flow. If one coefficient set works everywhere --
laminar, transitional and turbulent, and on a second independent DNS -- that
is a constitutive law. If the coefficients must be refitted per region or per
case, it is a curve fit, and we should say so.

Two targets are regressed, because they are the two things RANS has to close:

  stress   -<u'v'>/k, what the mean-momentum equation needs
  epsilon  the residual of the exact k budget, (P - advection)/(k S), which
           is dissipation plus turbulent transport -- the genuinely unclosed
           part of the k equation

THE TRIVIAL SOLUTION. The failure mode of this whole approach is a library
that carries information from the target itself: least squares then recovers a
near-identity and reports a superb R^2 that means nothing. Two guards are
applied here.

  1. The admissible library contains ONLY quantities a RANS solver has: mean
     velocity derivatives, wall distance, molecular viscosity, and its own
     transported k. Nothing from the fluctuation field.
  2. Every library column's correlation with the target is reported, and any
     column above TRIVIAL_CORR is flagged. A flagged column is not evidence.

The inadmissible anisotropy term v'/u' is kept only as a labelled diagnostic
of how much information a mean-field closure lacks. It is itself close to
tautological -- since -<u'v'> = R_uv u'v' and u'^2/2k is roughly steady,
-<u'v'>/k is nearly proportional to v'/u' -- so its R^2 is quoted as an upper
bound on what any anisotropy-aware closure could reach, never as a result.

Outputs
-------
results/pde-term-regression.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypkg.dns_case import load_dns

# Above this |correlation| with the target, a single column essentially IS the
# answer and the regression is not evidence of anything.
TRIVIAL_CORR = 0.95


def stress_library(y, S, dSdy, k, nu, vrms, urms, diagnostic=False):
    """Dimensionless groups for -<u'v'>/k, from RANS-available quantities."""
    sk = np.maximum(np.sqrt(np.maximum(k, 0.0)), 1e-12)
    g = y * S / sk
    cols = {
        "1": np.ones_like(g),
        "yS/sqrt(k)": g,
        "(yS/sqrt(k))^2": g ** 2,
        "1/(1+yS/sqrt(k))": 1.0 / (1.0 + g),
        "nu*S/k": nu * S / np.maximum(k, 1e-16),
        # Curvature of the mean profile, which distinguishes a full turbulent
        # profile from a laminar one at the same local shear
        "y*d2U/dy2/S": y * dSdy / np.maximum(S, 1e-12),
        # Bounded forms of the two Reynolds numbers the closure search kept
        # selecting, so their magnitude does not wreck the conditioning
        "Re_v/(1e3+Re_v)": (y ** 2 * S / nu) / (1e3 + y ** 2 * S / nu),
        "Re_k/(1e3+Re_k)": (sk * y / nu) / (1e3 + sk * y / nu),
    }
    if diagnostic:
        cols["v'/u' [INADMISSIBLE]"] = vrms / np.maximum(urms, 1e-12)
    return cols


def eps_library(y, S, dSdy, k, nu, vrms, urms, diagnostic=False):
    """Dimensionless groups for the unclosed part of the k budget, /(k S)."""
    sk = np.maximum(np.sqrt(np.maximum(k, 0.0)), 1e-12)
    kS = np.maximum(k * S, 1e-16)
    cols = {
        "1": np.ones_like(y),
        "sqrt(k)/(y S)": sk / np.maximum(y * S, 1e-12),
        "nu/(y^2 S)": nu / np.maximum(y ** 2 * S, 1e-16),
        "k/(nu*Re)": nu * k / np.maximum(y ** 2 * kS, 1e-16),
        "y*d2U/dy2/S": y * dSdy / np.maximum(S, 1e-12),
        "Re_v/(1e3+Re_v)": (y ** 2 * S / nu) / (1e3 + y ** 2 * S / nu),
    }
    if diagnostic:
        cols["v'/u' [INADMISSIBLE]"] = vrms / np.maximum(urms, 1e-12)
    return cols


def jhtdb_samples(target, y_lo=0.03, y_hi=0.9, diagnostic=False, stride=8):
    d = load_dns()
    x, y, U, V = d["x"], d["y"], d["U"], d["V"]
    nu = d["nu"]
    uu, vv, ww, uv = d["uu"], d["vv"], d["ww"], d["uv"]
    k = 0.5 * (uu + vv + ww)
    dUdy = np.gradient(U, y, axis=0)
    d2Udy2 = np.gradient(dUdy, y, axis=0)
    dUdx = np.gradient(U, x, axis=1)
    P = -uv * dUdy - (uu - vv) * dUdx
    adv = U * np.gradient(k, x, axis=1) + V * np.gradient(k, y, axis=0)

    rows, tgt, xs = [], [], []
    for j in range(0, len(x), stride):
        Ue = float(np.max(U[:, j]))
        i = int(np.argmax(U[:, j]))
        d99 = float(np.interp(0.99 * Ue, U[:i + 1, j], y[:i + 1]))
        m = (y > y_lo * d99) & (y <= y_hi * d99) & (k[:, j] > 1e-12)
        if m.sum() < 5:
            continue
        S = np.abs(dUdy[m, j])
        args = (y[m], S, d2Udy2[m, j], k[m, j], nu,
                np.sqrt(np.maximum(vv[m, j], 0)),
                np.sqrt(np.maximum(uu[m, j], 0)))
        if target == "stress":
            cols = stress_library(*args, diagnostic=diagnostic)
            t = -uv[m, j] / np.maximum(k[m, j], 1e-16)
        else:
            cols = eps_library(*args, diagnostic=diagnostic)
            t = (P[m, j] - adv[m, j]) / np.maximum(k[m, j] * S, 1e-16)
        good = np.isfinite(t) & np.all(
            np.isfinite(np.column_stack(list(cols.values()))), axis=1)
        if good.sum() < 5:
            continue
        rows.append(np.column_stack(list(cols.values()))[good])
        tgt.append(t[good])
        xs.append(np.full(int(good.sum()), x[j]))
    return (list(cols.keys()), np.vstack(rows), np.concatenate(tgt),
            np.concatenate(xs))


def jimenez_samples(target, y_lo=0.03, y_hi=0.9, diagnostic=False):
    rows, tgt = [], []
    for f in sorted(glob.glob("data/jiminez/Re_theta.*.prof")):
        a = np.loadtxt(f, comments="%")
        yd, yplus = a[:, 0], a[:, 1]
        urms, vrms, wrms, uv, dumdy = (a[:, 2], a[:, 3], a[:, 4], a[:, 5],
                                       a[:, 17])
        k = 0.5 * (urms ** 2 + vrms ** 2 + wrms ** 2)
        d2 = np.gradient(dumdy, yplus)
        m = (yd > y_lo) & (yd <= y_hi) & (k > 1e-12)
        if m.sum() < 5:
            continue
        S = np.abs(dumdy[m])
        args = (yplus[m], S, d2[m], k[m], 1.0, vrms[m], urms[m])
        if target == "stress":
            cols = stress_library(*args, diagnostic=diagnostic)
            t = -uv[m] / np.maximum(k[m], 1e-16)
        else:
            # No streamwise derivatives in this one-point data, so the budget
            # residual cannot be formed; skip.
            return None, None, None
        rows.append(np.column_stack(list(cols.values())))
        tgt.append(t)
    return list(cols.keys()), np.vstack(rows), np.concatenate(tgt)


def fit(X, t, ridge=1e-8):
    A = X.T @ X + ridge * np.eye(X.shape[1]) * np.trace(X.T @ X) / X.shape[1]
    return np.linalg.solve(A, X.T @ t)


def score(X, t, c):
    pred = X @ c
    ss_res = float(np.sum((t - pred) ** 2))
    ss_tot = float(np.sum((t - np.mean(t)) ** 2))
    return {"r2": 1.0 - ss_res / max(ss_tot, 1e-30),
            "rms": float(np.sqrt(np.mean((t - pred) ** 2))),
            "n": int(len(t))}


def trivial_check(names, X, t):
    flags = {}
    for i, n in enumerate(names):
        col = X[:, i]
        if np.std(col) < 1e-14:
            flags[n] = 0.0
            continue
        flags[n] = float(abs(np.corrcoef(col, t)[0, 1]))
    return flags


def run_target(target, x_split, out):
    names, Xj, tj, xj = jhtdb_samples(target)
    res = {"terms": names, "n_samples": int(len(tj))}

    corr = trivial_check(names, Xj, tj)
    res["term_correlation_with_target"] = corr
    res["trivially_correlated_terms"] = [n for n, v in corr.items()
                                         if v > TRIVIAL_CORR]

    c_all = fit(Xj, tj)
    res["in_sample"] = {"coeffs": c_all.tolist(), **score(Xj, tj, c_all)}

    up = xj <= x_split
    c_up, c_dn = fit(Xj[up], tj[up]), fit(Xj[~up], tj[~up])
    res["fit_upstream"] = {"coeffs": c_up.tolist(),
                           "fit": score(Xj[up], tj[up], c_up),
                           "predict_downstream": score(Xj[~up], tj[~up], c_up)}
    res["fit_downstream"] = {"coeffs": c_dn.tolist(),
                             "fit": score(Xj[~up], tj[~up], c_dn),
                             "predict_upstream": score(Xj[up], tj[up], c_dn)}
    # How far apart are the two regimes' coefficients? A law would have one set.
    denom = np.maximum(np.abs(c_up) + np.abs(c_dn), 1e-12)
    res["coeff_disagreement"] = float(np.max(np.abs(c_up - c_dn) / denom))

    zn, Xz, tz = jimenez_samples(target)
    if Xz is not None:
        res["cross_dataset"] = {
            "predict_jimenez": score(Xz, tz, c_all),
            "jimenez_own_fit": score(Xz, tz, fit(Xz, tz)),
            "jimenez_coeffs": fit(Xz, tz).tolist(),
        }

    dn, Xd, td, _ = jhtdb_samples(target, diagnostic=True)
    cd = fit(Xd, td)
    dcorr = trivial_check(dn, Xd, td)
    res["diagnostic_with_anisotropy"] = {
        "terms": dn, "coeffs": cd.tolist(), **score(Xd, td, cd),
        "term_correlation_with_target": dcorr,
        "warning": ("v'/u' is not available to a RANS solver and is close to "
                    "tautological for the stress target. Treat this R^2 as an "
                    "upper bound on an anisotropy-aware closure, not a result."),
    }
    out[target] = res
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-split", type=float, default=450.0)
    ap.add_argument("--out", default="results/pde-term-regression.json")
    args = ap.parse_args()

    out = {"trivial_correlation_threshold": TRIVIAL_CORR}
    for target in ("stress", "epsilon"):
        r = run_target(target, args.x_split, out)
        print(f"=== target: {target} ({r['n_samples']} samples) ===")
        flagged = r["trivially_correlated_terms"]
        print(f"  trivially correlated terms: {flagged or 'none'}")
        print(f"  in-sample                R2={r['in_sample']['r2']:8.3f}")
        print(f"  fit upstream  -> downstream "
              f"R2={r['fit_upstream']['predict_downstream']['r2']:8.3f}")
        print(f"  fit downstream -> upstream "
              f"R2={r['fit_downstream']['predict_upstream']['r2']:8.3f}")
        print(f"  max coeff disagreement between regimes: "
              f"{r['coeff_disagreement']:.2f}  (0 = one law, 1 = unrelated)")
        if "cross_dataset" in r:
            print(f"  JHTDB -> Jimenez         "
                  f"R2={r['cross_dataset']['predict_jimenez']['r2']:8.3f}"
                  f"   (Jimenez self-fit "
                  f"{r['cross_dataset']['jimenez_own_fit']['r2']:.3f})")
        print(f"  diagnostic +anisotropy   "
              f"R2={r['diagnostic_with_anisotropy']['r2']:8.3f}  [upper bound]")
        print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
