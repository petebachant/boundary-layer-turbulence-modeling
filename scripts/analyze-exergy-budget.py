#!/usr/bin/env python
"""Exergy injection, entropy rejection, and turbulent storage in the DNS.

Frames the transitional boundary layer thermodynamically. A steady, nearly
uniform inflow is a low-entropy, high-exergy stream. The wall degrades it, and
that degradation is rejected as entropy by two routes:

  direct     nu*(dU/dy)^2   -- viscous dissipation of the MEAN field, immediate
  turbulent  production -> k -> epsilon -- BUFFERED through a reservoir

The turbulent route is not instantaneous: energy entering k is dissipated later
and downstream. So when the rejection rate cannot keep up with the injection
rate, the excess is *stored* as turbulent kinetic energy. This script measures
that storage, its timing relative to transition, and its Reynolds-number
dependence.

Outputs
-------
results/exergy-budget.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from py_package.dns_case import load_dns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y-max", type=float, default=12.0)
    ap.add_argument("--out", default="results/exergy-budget.json")
    args = ap.parse_args()

    d = load_dns()
    x, y, U, V = d["x"], d["y"], d["U"], d["V"]
    uv, k, nu = d["uv"], d["k"], d["nu"]
    dUdy = np.gradient(U, y, axis=0)
    m = y < args.y_max

    def I(a):
        return np.trapz(a[m, :], y[m], axis=0)

    Ue = U[-1, :]
    tau_w = nu * U[0, :] / y[0]
    Wdot = tau_w * Ue                 # exergy input rate (wall work scale)
    Phi = I(nu * dUdy ** 2)           # direct (immediate) rejection
    P = I(-uv * dUdy)                 # exergy routed into the reservoir
    Fk = I(U * k)                     # flux of stored turbulent energy
    dFk = np.gradient(Fk, x)          # storage rate
    eps = P - dFk                     # rejection via the turbulent route
    Ik = I(k)

    d99 = np.array([np.interp(0.99 * Ue[i], U[:, i], y) for i in range(len(x))])
    tau_store = Ik / np.maximum(eps, 1e-12)
    tau_flow = d99 / Ue
    Re_th = np.array([
        Ue[i] * np.trapz(U[m, i] / Ue[i] * (1 - U[m, i] / Ue[i]), y[m]) / nu
        for i in range(len(x))
    ])

    storage_frac = dFk / np.maximum(P, 1e-30)
    turb_frac = P / np.maximum(P + Phi, 1e-30)

    w = (x > 100) & (x < 600)
    i_pk = int(np.argmax(storage_frac[w]))
    x_pk = float(x[w][i_pk])

    re_bins = []
    for lo, hi in [(500, 650), (650, 800), (800, 900), (900, 1000)]:
        s = (x >= lo) & (x < hi)
        re_bins.append({"x_lo": lo, "x_hi": hi,
                        "Re_theta": float(Re_th[s].mean()),
                        "turbulent_rejection_fraction": float(turb_frac[s].mean())})

    stations = [
        {"x": float(x[i]), "Re_theta": float(Re_th[i]),
         "exergy_input": float(Wdot[i]),
         "direct_rejection": float(Phi[i]),
         "into_reservoir": float(P[i]),
         "storage_rate": float(dFk[i]),
         "storage_fraction": float(storage_frac[i]),
         "turbulent_rejection_fraction": float(turb_frac[i]),
         "tau_store_over_tau_flow": float(tau_store[i] / tau_flow[i])}
        for i in range(30, len(x), 100)
    ]

    payload = {
        "peak_storage_fraction": float(storage_frac[w][i_pk]),
        "peak_storage_x": x_pk,
        "reynolds_dependence": re_bins,
        "inflow_disorder_fraction": float(
            k[-1, 0] / (0.5 * U[-1, 0] ** 2)),
        "stations": stations,
    }
    os.makedirs("results", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"peak storage fraction {payload['peak_storage_fraction']:.3f} "
          f"at x={x_pk:.0f}")
    print("turbulent rejection fraction vs Re_theta:")
    for b in re_bins:
        print(f"  Re_theta {b['Re_theta']:6.0f} -> "
              f"{b['turbulent_rejection_fraction']:.4f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
