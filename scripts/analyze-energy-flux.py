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

from pypkg.dns_case import load_dns


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
        # Integrate to delta99, matching the independent dataset exactly.
        # An earlier version used 3*delta99 here and delta99 there, which made
        # the two equilibrium values incomparable -- a normalisation
        # difference masquerading as a physical discrepancy.
        m = y <= d99[j]
        Fk[j] = np.trapz(U[m, j] * k[m, j], y[m])
        Pint[j] = np.trapz(P[m, j], y[m])
        kbar[j] = np.trapz(k[m, j], y[m]) / max(d99[j], 1e-12)
        visc_direct[j] = np.trapz(nu * dUdy[m, j] ** 2, y[m])

    # eps_int from the exact integral balance
    dFkdx = np.gradient(Fk, x)
    eps_int = Pint - dFkdx

    # Eddy-turnover count: how many turnover times the turbulence has had to
    # equilibrate since the inlet. A cascade needs O(1) turnovers to reach
    # equilibrium, so if C_eps is set by history rather than by local state
    # this is the variable it should collapse against.
    #
    #   N(x) = integral( (eps/k) / U ) dx
    eps_bar = eps_int / np.maximum(d99, 1e-12)
    rate = eps_bar / np.maximum(kbar, 1e-16) / np.maximum(ue, 1e-12)
    turnovers = np.concatenate(([0.0], np.cumsum(
        0.5 * (rate[1:] + rate[:-1]) * np.diff(x))))

    rows = []
    for j in range(0, nx, args.stride):
        # C_eps = eps_bar * L / u'^3 with L = delta99 and
        # eps_bar = eps_int/delta99, which collapses to eps_int/u'^3.
        up = float(np.sqrt(max(2.0 * kbar[j] / 3.0, 0.0)))
        c_eps = eps_int[j] / max(up ** 3, 1e-30)
        rows.append({
            "x": float(x[j]),
            "delta99": float(d99[j]),
            "Ue": float(ue[j]),
            "Re_theta": float(np.trapz(
                np.clip(U[y <= d99[j], j] / ue[j], 0, 1)
                * (1 - np.clip(U[y <= d99[j], j] / ue[j], 0, 1)),
                y[y <= d99[j]]) * ue[j] / nu),
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
            "k_bar": float(kbar[j]),
            "turnovers": float(turnovers[j]),
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
        # Same definition as above: C_eps = eps_int/u'^3, integrated to
        # delta99, with eps_int = P_int in an equilibrium layer.
        jim.append({
            "Re_theta": ret,
            "C_eps_equilibrium": float(Pj / max(upj ** 3, 1e-30)),
            "P_int_plus": float(Pj),
        })

    # Does C_eps collapse onto anything? Fit C_inf*v/(v0+v) for each candidate
    # and -- crucially -- include the streamwise coordinate itself as a trivial
    # baseline. C_eps is monotonic in x, so any variable that is also monotonic
    # in x will correlate with it. A candidate only means something if it
    # collapses the data BETTER than the bare coordinate does. Correlation
    # alone is worthless here: gamma reaches r = 0.99 while collapsing worse
    # than x.
    from scipy.optimize import least_squares
    sel = [r for r in rows if 40 <= r["x"] <= 1000 and np.isfinite(r["C_eps"])]
    ce = np.array([r["C_eps"] for r in sel])
    c_inf = float(np.mean([r["C_eps"] for r in sel if r["x"] >= 600]))
    # a1 = -<u'v'>/2k integrated across the layer, at each selected station
    a1_sel = []
    for r in sel:
        j = int(np.argmin(np.abs(x - r["x"])))
        mm = y <= d99[j]
        a1_sel.append(np.trapz(-uv[mm, j], y[mm])
                      / max(np.trapz(2 * k[mm, j], y[mm]), 1e-30))
    a1_sel = np.array(a1_sel)
    a1_inf = float(np.mean(a1_sel[np.array([r["x"] for r in sel]) > 800]))
    cands = {
        "turnovers_history": np.array([r["turnovers"] for r in sel]),
        "Re_theta_local": np.array([r["Re_theta"] for r in sel]),
        "gamma_local": a1_sel / max(a1_inf, 1e-30),
        "x_TRIVIAL_BASELINE": np.array([r["x"] for r in sel]),
    }
    collapse = {}
    for name, v in cands.items():
        av = np.abs(v)
        r = least_squares(
            lambda p: c_inf * av / (abs(p[0]) + av) - ce, [np.median(av)])
        v0 = float(abs(r.x[0]))
        pred = c_inf * av / (v0 + av)
        collapse[name] = {
            "scale": v0,
            "rel_rms_percent": float(100 * np.sqrt(np.mean(
                ((pred - ce) / np.maximum(ce, 1e-9)) ** 2))),
            "correlation": float(np.corrcoef(v, ce)[0, 1]),
        }
    best = min(collapse, key=lambda n: collapse[n]["rel_rms_percent"])
    collapse["verdict"] = (
        "no candidate beats the trivial coordinate baseline"
        if collapse[best]["rel_rms_percent"]
        >= 0.9 * collapse["x_TRIVIAL_BASELINE"]["rel_rms_percent"]
        else f"{best} collapses better than the coordinate baseline")

    # Structure parameter, cross-validated against the independent DNS. This
    # is Bradshaw's a1; it is recorded here because it is the one quantity in
    # this study that behaves like a constant.
    a1_jim = []
    for f in sorted(glob.glob("data/jiminez/Re_theta.*.prof")):
        a = np.loadtxt(f, comments="%")
        yd = a[:, 0]
        urms, vrms, wrms, uvj = a[:, 2], a[:, 3], a[:, 4], a[:, 5]
        kk = 0.5 * (urms ** 2 + vrms ** 2 + wrms ** 2)
        mm = (yd > 0.01) & (yd <= 1.0)
        a1_jim.append(float(np.trapz(-uvj[mm], yd[mm])
                            / max(np.trapz(2 * kk[mm], yd[mm]), 1e-30)))
    a1_stats = {
        "a1_transitional_downstream": a1_inf,
        "a1_jimenez_mean": float(np.mean(a1_jim)),
        "a1_jimenez_sd": float(np.std(a1_jim)),
        "a1_relative_difference": float(
            abs(a1_inf - np.mean(a1_jim)) / max(np.mean(a1_jim), 1e-30)),
    }

    pre = [r for r in rows if r["x"] <= 205]
    post = [r for r in rows if r["x"] >= 600]
    out = {
        "note": ("eps_int is measured from the exact integral k budget, not "
                 "modelled. C_eps constant would mean the cascade is in "
                 "equilibrium, which every standard closure assumes."),
        "stations": rows,
        "c_eps_collapse": collapse,
        "structure_parameter": a1_stats,
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
            "P_over_eps_peak": float(max(
                (r["P_over_eps"] for r in rows
                 if 100 <= r["x"] <= 500 and np.isfinite(r["P_over_eps"])),
                default=float("nan"))),
            "P_over_eps_peak_x": float(max(
                (r for r in rows
                 if 100 <= r["x"] <= 500 and np.isfinite(r["P_over_eps"])),
                key=lambda r: r["P_over_eps"], default={"x": float("nan")})["x"]),
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
    print()
    for n, v in collapse.items():
        if n == "verdict":
            continue
        print(f"  collapse onto {n:<22} rel RMS {v['rel_rms_percent']:5.1f}%  "
              f"corr {v['correlation']:.4f}")
    print(f"  VERDICT: {collapse['verdict']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
