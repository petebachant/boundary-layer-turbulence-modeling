#!/usr/bin/env python
"""Fit inlet k and omega so the free-stream turbulence decays like the DNS.

Bypass transition is driven by free-stream turbulence, so getting its level and
decay rate right matters as much as the closure itself. A k-omega model decays
homogeneous turbulence analytically,

    omega(x) = omega_0 / (1 + beta*omega_0*(x - x_in)/U)
    k(x)     = k_0 * (1 + beta*omega_0*(x - x_in)/U)^(-betaStar/beta)

so the inlet pair (k_0, omega_0) is fully determined by matching the measured
DNS free-stream decay. Setting omega_0 too low - the usual failure - leaves the
free stream barely decaying at all, which floods the boundary layer with
spurious turbulence far downstream.

Outputs
-------
results/freestream-bcs.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.optimize import least_squares

from pypkg.dns_case import load_dns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-inlet", type=float, default=-118.0,
                    help="streamwise position of the OpenFOAM inlet patch")
    ap.add_argument("--coeffs", default="results/clip-k-gamma-coeffs.json")
    ap.add_argument("--out", default="results/freestream-bcs.json")
    args = ap.parse_args()

    d = load_dns()
    x, y, U, k = d["x"], d["y"], d["U"], d["k"]
    k_fs = k[-1, :]
    Ue = float(U[-1, :].mean())

    with open(args.coeffs) as f:
        c = json.load(f)["openfoam_coeffs"]
    beta, betaStar = c["beta"], c["betaStar"]

    # Parameterise on the VIRTUAL ORIGIN of the decay rather than on the inlet
    # pair directly. Fitting (k_0, omega_0) at the inlet is ill-conditioned:
    # once beta*omega_0*(x-x_in)/U >> 1 only the combination
    # k_0*(beta*omega_0)^(-betaStar/beta) is identifiable, and the solver runs
    # away to absurd values. In virtual-origin form the decay is a clean power
    # law and both parameters are well determined.
    #
    #   omega(x) = U / (beta * (x - x_v))
    #   k(x)     = A * (x - x_v)^(-betaStar/beta)
    n = betaStar / beta

    def model(p, xs):
        logA, xv = p[0], p[1]
        return np.exp(logA) * np.maximum(xs - xv, 1e-6) ** (-n)

    sel = x > 40.0
    # x_v must sit upstream of the inlet for the inlet values to be physical
    res = least_squares(
        lambda p: np.log(model(p, x[sel])) - np.log(k_fs[sel]),
        x0=[np.log(1e-2), -200.0],
        bounds=([-30.0, -1e5], [30.0, args.x_inlet - 1.0]),
        max_nfev=20000,
    )
    logA, xv = res.x
    w0 = Ue / (beta * (args.x_inlet - xv))
    k0 = float(model(res.x, np.array([args.x_inlet]))[0])

    pred = model(res.x, x[sel])
    rel = np.abs(pred - k_fs[sel]) / k_fs[sel]

    payload = {
        "x_inlet": args.x_inlet,
        "virtual_origin_x": float(xv),
        "decay_exponent": float(n),
        "k_inlet": float(k0),
        "omega_inlet": float(w0),
        "Ue": Ue,
        "beta": beta, "betaStar": betaStar,
        "Tu_inlet_percent": float(np.sqrt(2 / 3 * k0) / Ue * 100),
        "fit_rel_err_mean": float(rel.mean()),
        "fit_rel_err_max": float(rel.max()),
        "check": [
            {"x": float(x[i]), "k_dns": float(k_fs[i]),
             "k_fit": float(model(res.x, np.array([x[i]]))[0])}
            for i in range(0, len(x), 400)
        ],
    }
    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"inlet at x={args.x_inlet}:  k={k0:.4e}  omega={w0:.4f}  "
          f"Tu={payload['Tu_inlet_percent']:.2f}%")
    print(f"free-stream decay fit: mean {rel.mean()*100:.1f}%, "
          f"max {rel.max()*100:.1f}% error over x>40")
    for r in payload["check"]:
        print(f"   x={r['x']:7.1f}  DNS {r['k_dns']:.3e}  fit {r['k_fit']:.3e}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
