#!/usr/bin/env python
"""Validate the parabolic boundary-layer solver against the Blasius solution.

Every closure comparison in this project is screened with
pypkg/bl_solver.py, so its discretisation error needs to be known and
smaller than the differences between models it is used to judge.

Outputs
-------
results/blasius-validation.json
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.integrate import solve_ivp

from pypkg.bl_solver import BLSolver
from pypkg.closures import Laminar


def blasius():
    """Shoot for f''(0); returns it plus a dense solution."""

    def rhs(_, f):
        return [f[1], f[2], -0.5 * f[0] * f[2]]

    lo, hi = 0.1, 1.0
    mid = 0.5 * (lo + hi)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        s = solve_ivp(rhs, [0, 12], [0, 0, mid], rtol=1e-10, atol=1e-12)
        if s.y[1, -1] < 1.0:
            lo = mid
        else:
            hi = mid
    sol = solve_ivp(rhs, [0, 12], [0, 0, mid], dense_output=True,
                    rtol=1e-10, atol=1e-12)
    return mid, sol


def main():
    fpp0, sol = blasius()
    nu, Ue = 1.25e-3, 1.0
    y = np.concatenate(([0.0], np.geomspace(1e-3, 30.0, 300)))
    x = np.linspace(20.0, 1000.0, 2000)

    eta = y * np.sqrt(Ue / (nu * x[0]))
    U0 = Ue * sol.sol(np.clip(eta, 0, 12))[1]
    solver = BLSolver(y, x, nu, np.full_like(x, Ue), np.zeros_like(x),
                      Laminar(), U0)
    res = solver.run()
    U = res["U"]

    rows, errs = [], []
    for i in range(0, len(x), 100):
        xi = float(x[i])
        cf_exact = 2 * fpp0 / np.sqrt(Ue * xi / nu)
        cf = float(2 * nu * (U[1, i] - U[0, i]) / (y[1] - y[0]) / Ue ** 2)
        d99 = float(np.interp(0.99 * Ue, U[:, i], y))
        d99_exact = float(4.910 * np.sqrt(nu * xi / Ue))
        e = abs(cf - cf_exact) / cf_exact
        errs.append(e)
        rows.append({"x": xi, "cf": cf, "cf_blasius": cf_exact,
                     "cf_rel_err": e, "d99": d99, "d99_blasius": d99_exact})

    payload = {
        "blasius_fpp0": fpp0,
        "blasius_fpp0_reference": 0.332057,
        "cf_rel_err_max": float(np.max(errs)),
        "cf_rel_err_mean": float(np.mean(errs)),
        "stations": rows,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/blasius-validation.json", "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Blasius f''(0) = {fpp0:.6f} (reference 0.332057)")
    print(f"solver c_f error: mean {np.mean(errs):.4f}, max {np.max(errs):.4f}")
    if np.max(errs) > 0.05:
        raise SystemExit("solver error exceeds 5%; screening results are unreliable")


if __name__ == "__main__":
    main()
