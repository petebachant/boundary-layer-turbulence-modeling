#!/usr/bin/env python
"""Integral energy-flux budget and the dissipation coefficient through
transition.

Closure by flux rather than by fitted coefficient. For a statistically steady
boundary layer the integral turbulent-energy budget is exact:

    d/dx integral(U k dy) = integral(P dy) - integral(eps dy)

because the turbulent, pressure and viscous transport terms are internal and
integrate to boundary fluxes that vanish at the wall and in the free stream.
So the integrated dissipation is *measurable* without modelling it:

    eps_int = P_int - d/dx(F_k),     F_k = integral(U k dy)

That is worth having on its own: it gives the one genuinely unclosed term in
the k equation directly from mean and second-moment data.

The physics question is then whether the cascade is in equilibrium. Standard
closures assume it is, which is the statement that the dissipation coefficient

    C_eps = eps_bar * L / u'^3,    u' = sqrt(2 k_bar / 3),  L = delta99

is a constant. If C_eps varies systematically through transition, the cascade
is out of equilibrium there and no equilibrium closure -- k-epsilon, k-omega,
or ours -- can be right for the reason it assumes. That is a statement about
the flow, not about a model, and it needs no fitted coefficient to make.

Outputs
-------
results/energy-flux.json
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

from py_package.dns_case import load_dns


def bl_edge(y, U):
    i = int(np.argmax(U))
    ue = float(U[i])
    return float(np.interp(0.99 * ue, U[:i + 1], y[:i + 1])), ue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", default="results/energy-flux.json")
    args = ap.parse_args()

    d = load_dns()
    x, y, U, V = d["x"], d["y"], d["U"], d["V"]
    nu = d["nu"]
    uu, vv, ww, uv = d["uu"], d["vv"], d["ww"], d["uv"]
    k = 0.5 * (uu + vv + ww)
    dUdy = np.gradient(U, y, axis=0)
    dUdx = np.gradient(U, x, axis=1)

    # Exact production, both shear and normal-stress parts
    P = -uv * dUdy - (uu - vv) * dUdx

    # Integrate across the layer. The upper limit is generous so the free
    # stream contributes its (small) share rather than being clipped.
    nx = len(x)
    Fk = np.zeros(nx)
    Pint = np.zeros(nx)
    d99 = np.zeros(nx)
    ue = np.zeros(nx)
    kbar = np.zeros(nx)
    visc_direct = np.zeros(nx)
    for j in range(nx):
        d99[j], ue[j] = bl_edge(y, U[:, j])
        m = y <= 3.0 * d99[j]
        Fk[j] = np.trapz(U[m, j] * k[m, j], y[m])
        Pint[j] = np.trapz(P[m, j], y[m])
        kbar[j] = np.trapz(k[m, j], y[m]) / max(3.0 * d99[j], 1e-12)
        visc_direct[j] = np.trapz(nu * dUdy[m, j] ** 2, y[m])

    # eps_int from the exact integral balance
    dFkdx = np.gradient(Fk, x)
    eps_int = Pint - dFkdx

    rows = []
    for j in range(0, nx, args.stride):
        up = float(np.sqrt(max(2.0 * kbar[j] / 3.0, 0.0)))
        eb = eps_int[j] / max(3.0 * d99[j], 1e-12)
        c_eps = eb * d99[j] / max(up ** 3, 1e-30)
        rows.append({
            "x": float(x[j]),
            "delta99": float(d99[j]),
            "Ue": float(ue[j]),
            "Re_theta": float(np.trapz(
                np.clip(U[y <= 3 * d99[j], j] / ue[j], 0, 1)
                * (1 - np.clip(U[y <= 3 * d99[j], j] / ue[j], 0, 1)),
                y[y <= 3 * d99[j]]) * ue[j] / nu),
            "F_k": float(Fk[j]),
            "dF_k/dx": float(dFkdx[j]),
            "P_int": float(Pint[j]),
            "eps_int": float(eps_int[j]),
            "visc_direct_int": float(visc_direct[j]),
            "P_over_eps": float(Pint[j] / eps_int[j])
            if abs(eps_int[j]) > 1e-30 else float("nan"),
            "turbulent_dissipation_fraction": float(
                eps_int[j] / max(eps_int[j] + visc_direct[j], 1e-30)),
            "C_eps": float(c_eps),
            "u_prime": up,
        })

    # Equilibrium reference from the independent turbulent DNS. Only local
    # production is available there (no streamwise derivatives), but in an
    # equilibrium layer P = eps, so P_int gives the equilibrium C_eps.
    jim = []
    for f in sorted(glob.glob("data/jiminez/Re_theta.*.prof")):
        ret = float(re.search(r"Re_theta\.(\d+)", f).group(1))
        hdr = open(f).read()
        dj = float(re.search(r"delta_99=\s*([\d.]+)", hdr).group(1))
        a = np.loadtxt(f, comments="%")
        yd, yp = a[:, 0], a[:, 1]
        urms, vrms, wrms, uvj, dumdy = (a[:, 2], a[:, 3], a[:, 4], a[:, 5],
                                        a[:, 17])
        kk = 0.5 * (urms ** 2 + vrms ** 2 + wrms ** 2)
        m = yd <= 1.0
        Pj = np.trapz(-uvj[m] * dumdy[m], yp[m])
        kb = np.trapz(kk[m], yp[m]) / max(yp[m][-1], 1e-12)
        upj = np.sqrt(max(2.0 * kb / 3.0, 0.0))
        dj_plus = yp[m][-1]
        jim.append({
            "Re_theta": ret,
            "C_eps_equilibrium": float((Pj / dj_plus) * dj_plus
                                       / max(upj ** 3, 1e-30)),
            "P_int_plus": float(Pj),
        })

    pre = [r for r in rows if r["x"] <= 205]
    post = [r for r in rows if r["x"] >= 600]
    out = {
        "note": ("eps_int is measured from the exact integral k budget, not "
                 "modelled. C_eps constant would mean the cascade is in "
                 "equilibrium, which every standard closure assumes."),
        "stations": rows,
        "jimenez_equilibrium": jim,
        "summary": {
            "C_eps_pre_transition_mean": float(np.mean([r["C_eps"]
                                                        for r in pre])),
            "C_eps_turbulent_mean": float(np.mean([r["C_eps"]
                                                   for r in post])),
            "C_eps_turbulent_spread": float(np.std([r["C_eps"]
                                                    for r in post])),
            "C_eps_jimenez_mean": float(np.mean([j["C_eps_equilibrium"]
                                                 for j in jim])),
            "P_over_eps_pre": float(np.mean([r["P_over_eps"] for r in pre])),
            "P_over_eps_turbulent": float(np.mean([r["P_over_eps"]
                                                   for r in post])),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print(f"{'x':>6} {'Re_th':>7} {'P_int':>10} {'eps_int':>10} "
          f"{'P/eps':>7} {'C_eps':>8} {'turb frac':>9}")
    for r in rows:
        if r["x"] > 1000:
            continue
        print(f"{r['x']:6.0f} {r['Re_theta']:7.0f} {r['P_int']:10.3e} "
              f"{r['eps_int']:10.3e} {r['P_over_eps']:7.3f} "
              f"{r['C_eps']:8.3f} {r['turbulent_dissipation_fraction']:9.3f}")
    s = out["summary"]
    print(f"\nC_eps pre-transition {s['C_eps_pre_transition_mean']:.3f}, "
          f"turbulent {s['C_eps_turbulent_mean']:.3f} "
          f"+- {s['C_eps_turbulent_spread']:.3f}")
    print(f"P/eps pre-transition {s['P_over_eps_pre']:.3f}, "
          f"turbulent {s['P_over_eps_turbulent']:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
