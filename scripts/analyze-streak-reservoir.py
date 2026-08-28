#!/usr/bin/env python
"""Measure the pre-transitional streak reservoir in the DNS and in the model.

The clipping closure rests on a two-reservoir picture: fluctuation energy
accumulates in streamwise streaks that carry almost no Reynolds shear stress,
and transition is the moment that energy is activated. This script checks
whether the model actually reproduces that reservoir, rather than only the
mean field it was scored against.

It measures, from the DNS:

* the fluctuation-energy budget across the boundary layer -- shear production
  -<u'v'> dU/dy, normal production, and mean advection -- so the size of the
  production the model must supply is known rather than assumed;
* the eddy viscosity nu_t = -<u'v'>/(dU/dy) implied by the DNS, and the
  structure parameter a1 = -<u'v'>/2k, at the point of peak production;
* the peak fluctuation energy, boundary-layer thickness, momentum thickness
  and vorticity Reynolds number Re_v = y^2 |dU/dy| / nu at each station.

and the same quantities from the parabolic screening solver, for each closure
variant given. Re_v is the quantity the clip fires on, so a model whose
pre-transitional boundary layer is the wrong thickness crosses the threshold
in the wrong place no matter how the threshold is calibrated.

Outputs
-------
results/streak-reservoir.json
figures/streak-reservoir.pdf
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypkg.closures import ClipKOmegaGamma
from pypkg.dns_case import Case, load_dns

STATIONS = [60, 100, 150, 205, 260, 310, 380, 450, 520, 600, 700, 800, 907, 980]

# Where the pre-transitional region ends. Chosen from the DNS skin friction,
# whose minimum sits near x = 205; downstream of that the flow is transitioning.
X_PRE = 205.0


def bl_edge(y, U):
    """99 percent thickness and edge velocity, measured below the U maximum."""
    i = int(np.argmax(U))
    ue = float(U[i])
    d99 = float(np.interp(0.99 * ue, U[: i + 1], y[: i + 1]))
    return d99, ue


def profile_metrics(y, U, nu):
    d99, ue = bl_edge(y, U)
    f = np.clip(U / ue, 0.0, 1.0)
    theta = float(np.trapz(f * (1.0 - f), y))
    dstar = float(np.trapz(1.0 - f, y))
    dudy = np.gradient(U, y)
    cf = float(2.0 * nu * (U[1] - U[0]) / (y[1] - y[0]) / ue ** 2)
    # Re_v is searched only inside the shear layer. Above it y^2 keeps growing
    # while dU/dy does not fall fast enough, so an unrestricted maximum can be
    # found in the free stream rather than in the boundary layer.
    m = y <= 1.3 * d99
    rev = float(np.max(y[m] ** 2 * np.abs(dudy[m]) / nu))
    ret = theta * ue / nu
    # Re_v / Re_theta is 2.193 for a Blasius profile. It is the cleanest single
    # measure of how far the mean profile has been distorted away from a
    # similarity solution, and it matters here because the clip fires on Re_v:
    # a model whose mean profile is Blasius-like reaches any given Re_v at a
    # different Re_theta -- that is, at a different place on the plate -- from
    # a boundary layer whose profile the streaks have already reshaped.
    return {"d99": d99, "theta": theta, "H": dstar / max(theta, 1e-12),
            "cf": cf, "Re_v": rev, "Re_theta": ret,
            "Re_v_over_Re_theta": rev / max(ret, 1e-12)}


def dns_measurements():
    d = load_dns()
    nu = d["nu"]
    x, y = d["x"], d["y"]
    U, V, k = d["U"], d["V"], d["k"]
    uu, vv, uv = d["uu"], d["vv"], d["uv"]

    dUdy = np.gradient(U, y, axis=0)
    dUdx = np.gradient(U, x, axis=1)
    P_shear = -uv * dUdy
    P_norm = -(uu - vv) * dUdx
    adv = U * np.gradient(k, x, axis=1) + V * np.gradient(k, y, axis=0)

    out = []
    for xq in STATIONS:
        j = int(np.argmin(np.abs(x - xq)))
        Uj = U[:, j]
        met = profile_metrics(y, Uj, nu)
        m = y <= 1.3 * met["d99"]
        ip = int(np.argmax(P_shear[m, j]))
        s = float(abs(dUdy[m, j][ip]))
        row = {
            "x": float(x[j]),
            "k_peak": float(np.max(k[m, j])),
            "int_P_shear": float(np.trapz(P_shear[m, j], y[m])),
            "int_P_norm": float(np.trapz(P_norm[m, j], y[m])),
            "int_advection": float(np.trapz(adv[m, j], y[m])),
            # nu_t and a1 evaluated where the shear production peaks, which is
            # the location that matters for the mean momentum balance
            "nut_over_nu": float(-uv[m, j][ip] / max(s, 1e-12) / nu),
            "a1_at_Ppeak": float(-uv[m, j][ip]
                                 / max(2.0 * k[m, j][ip], 1e-16)),
            "y_Ppeak": float(y[m][ip]),
            "y_Ppeak_over_d99": float(y[m][ip] / max(met["d99"], 1e-12)),
            # Mixing-length coefficient implied by the DNS at the production
            # peak: nu_t = C_ell * sqrt(k) * delta99. Through the whole
            # pre-transitional region this comes out essentially constant,
            # which says the streak eddy viscosity follows an outer
            # mixing-length scaling on a length that is a fixed fraction of
            # the boundary-layer thickness. A local closure cannot use
            # delta99, so this is the target a local length scale has to hit.
            "nut_over_k_at_Ppeak": float(
                -uv[m, j][ip] / max(abs(dUdy[m, j][ip]), 1e-12)
                / max(k[m, j][ip], 1e-16)),
            "dissipation_over_production": float(
                1.0 - np.trapz(adv[m, j], y[m])
                / max(np.trapz(P_shear[m, j], y[m]), 1e-30)),
            "C_ell_at_Ppeak": float(
                -uv[m, j][ip] / max(abs(dUdy[m, j][ip]), 1e-12)
                / max(np.sqrt(max(k[m, j][ip], 0.0)), 1e-16)
                / max(met["d99"], 1e-12)),
        }
        row.update(met)
        # The share of production that goes into growing k rather than being
        # dissipated. Small here means dissipation nearly balances production,
        # which is what lets a small nu_t sustain a large streak energy.
        row["advection_over_production"] = (
            row["int_advection"] / max(row["int_P_shear"], 1e-30))
        out.append(row)
    return out


def model_measurements(case, coeffs, extra):
    kw = dict(coeffs)
    kw.update(extra)
    kw["k_inf"] = case.kinf_fn()
    res = case.solve(ClipKOmegaGamma(**kw))
    U, V = res["U"], res["V"]
    k, g, nut = res["k"], res["gamma"], res["nut"]
    if not np.all(np.isfinite(U)):
        return None
    y, nu = case.y, case.nu
    # Energy budget, computed exactly as for the DNS so the two are
    # comparable. Dissipation is not evaluated from the closure's sink terms
    # -- that would tie this diagnostic to the model internals -- but inferred
    # the same way it is for the DNS, as whatever production does not go into
    # advecting k downstream.
    dUdy = np.gradient(U, y, axis=0)
    P_shear = nut * dUdy ** 2
    adv = (U * np.gradient(k, case.x, axis=1)
           + V * np.gradient(k, y, axis=0))
    out = []
    for xq in STATIONS:
        i = int(np.argmin(np.abs(case.x - xq)))
        met = profile_metrics(y, U[:, i], nu)
        m = y <= 1.3 * met["d99"]
        ip = int(np.argmax(P_shear[m, i]))
        iP = float(np.trapz(P_shear[m, i], y[m]))
        iA = float(np.trapz(adv[m, i], y[m]))
        row = {"x": float(case.x[i]),
               "k_peak": float(np.max(k[m, i])),
               "nut_over_nu": float(np.max(nut[m, i]) / nu),
               "gamma_max": float(np.max(g[m, i])),
               "int_P_shear": iP,
               "int_advection": iA,
               "advection_over_production": iA / max(iP, 1e-30),
               # 1 - advection/production. The DNS runs near 0.66 through the
               # pre-transitional region; a model above 1 is dissipating its
               # streak energy faster than it makes it, and k decays where the
               # DNS grows it.
               "dissipation_over_production": 1.0 - iA / max(iP, 1e-30),
               # nu_t/k at the production peak. This is the ratio that decides
               # whether the closure can hold a large streak energy at a small
               # momentum transport, which is the whole two-reservoir claim.
               "nut_over_k_at_Ppeak": float(
                   nut[m, i][ip] / max(k[m, i][ip], 1e-16)),
               "y_Ppeak_over_d99": float(y[m][ip] / max(met["d99"], 1e-12))}
        row.update(met)
        out.append(row)
    # Transition onset: first station where the activation passes one half
    gmax = np.array([np.max(g[:, i]) for i in range(len(case.x))])
    idx = np.where(gmax > 0.5)[0]
    x_onset = float(case.x[idx[0]]) if len(idx) else float("nan")
    return {"stations": out, "x_gamma_half": x_onset}


def onset_dns(rows):
    """DNS transition onset, taken as the skin-friction minimum."""
    cf = np.array([r["cf"] for r in rows])
    x = np.array([r["x"] for r in rows])
    return float(x[int(np.argmin(cf))])


def summarize(dns_rows, mod_rows):
    """Errors split at the end of the pre-transitional region.

    Rows are paired by position, not by x: the DNS grid and the marching grid
    land on slightly different nearest nodes for the same requested station.
    """
    pre_lk, all_lk, pre_rev, all_th = [], [], [], []
    for ref, r in zip(dns_rows, mod_rows):
        lk = np.log(max(r["k_peak"], 1e-16) / max(ref["k_peak"], 1e-16))
        all_lk.append(lk)
        all_th.append((r["theta"] - ref["theta"]) / ref["theta"])
        if r["x"] <= X_PRE:
            pre_lk.append(lk)
            pre_rev.append((r["Re_v"] - ref["Re_v"]) / ref["Re_v"])
    pre_ratio = [r["Re_v_over_Re_theta"] for r in mod_rows if r["x"] <= X_PRE]
    dns_ratio = [r["Re_v_over_Re_theta"] for r in dns_rows if r["x"] <= X_PRE]
    return {
        "Re_v_over_Re_theta_pre_mean": float(np.mean(pre_ratio)),
        "Re_v_over_Re_theta_pre_mean_dns": float(np.mean(dns_ratio)),
        "k_log_rms_pre": float(np.sqrt(np.mean(np.square(pre_lk)))),
        "k_log_rms_all": float(np.sqrt(np.mean(np.square(all_lk)))),
        "k_ratio_pre_median": float(np.exp(np.median(pre_lk))),
        "Re_v_rel_err_pre_mean": float(np.mean(pre_rev)),
        "theta_rel_rms": float(np.sqrt(np.mean(np.square(all_th)))),
    }


def make_figure(path, dns_rows, variants):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [r["x"] for r in dns_rows]
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
    ax[0].semilogy(xs, [r["k_peak"] for r in dns_rows], "k-o", ms=3,
                   label="DNS")
    ax[1].plot(xs, [r["Re_v"] for r in dns_rows], "k-o", ms=3, label="DNS")
    ax[2].plot(xs, [r["nut_over_nu"] for r in dns_rows], "k-o", ms=3,
               label="DNS")
    for name, v in variants.items():
        if v is None:
            continue
        rows = v["stations"]
        xm = [r["x"] for r in rows]
        ax[0].semilogy(xm, [r["k_peak"] for r in rows], "--", label=name)
        ax[1].plot(xm, [r["Re_v"] for r in rows], "--", label=name)
        ax[2].plot(xm, [r["nut_over_nu"] for r in rows], "--", label=name)
    for a, lab in zip(ax, [r"peak $k$", r"$Re_v$", r"$\nu_t/\nu$"]):
        a.set_xlabel("$x$")
        a.set_ylabel(lab)
        a.axvline(X_PRE, color="0.7", lw=0.8)
    ax[2].set_yscale("log")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coeffs", default="results/clip-k-gamma-coeffs.json")
    ap.add_argument("--x-stride", type=int, default=8)
    ap.add_argument("--out", default="results/streak-reservoir.json")
    ap.add_argument("--figure", default="figures/streak-reservoir.pdf")
    args = ap.parse_args()

    dns_rows = dns_measurements()
    case = Case(root=".", x_stride=args.x_stride)

    payload = json.load(open(args.coeffs))
    coeffs = payload["internal_coeffs"]
    structure = payload.get("structure", {})
    base = {"param": "Rev", "p": 1.0, "local_liftup": True,
            "log_layer_consistent": True, "freestream_decay": True,
            "x_virtual": -201.1, "x0": 30.2, "liftup_mode": "total",
            "gate_dissipation": False}
    base.update(structure)

    # The fitted structure, and the ungated omega production it replaced, so
    # the effect of the gate on the streak reservoir is visible here rather
    # than only in the fit score.
    variants = {}
    for name, over in [("fitted", {}),
                       ("omega production ungated", {"gate_omega": False})]:
        e = dict(base)
        e.update(over)
        variants[name] = model_measurements(case, coeffs, e)

    out = {
        "note": ("Pre-transitional streak reservoir, measured a priori from "
                 "the DNS and a posteriori from the screening solver."),
        "blasius_Re_v_over_Re_theta": 2.193,
        "x_pre_transition_end": X_PRE,
        "dns_onset_cf_min_x": onset_dns(dns_rows),
        "dns": dns_rows,
        "models": {
            name: (None if v is None else
                   {**v, "summary": summarize(dns_rows, v["stations"])})
            for name, v in variants.items()
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    make_figure(args.figure, dns_rows, variants)

    pre = [r for r in dns_rows if r["x"] <= X_PRE]
    ce = [r["C_ell_at_Ppeak"] for r in pre]
    out["dns_C_ell_pre"] = {
        "mean": float(np.mean(ce)), "std": float(np.std(ce)),
        "per_station": {str(int(r["x"])): r["C_ell_at_Ppeak"] for r in pre},
    }
    print(f"DNS onset (c_f minimum) at x = {out['dns_onset_cf_min_x']:.0f}")
    print(f"DNS streak mixing length: nu_t = {np.mean(ce):.4f} sqrt(k) delta99 "
          f"(std {np.std(ce):.4f} over x <= {X_PRE:.0f}), "
          f"production peak at y/delta99 = "
          f"{np.mean([r['y_Ppeak_over_d99'] for r in pre]):.2f}")
    for name, v in out["models"].items():
        if v is None:
            print(f"{name}: diverged")
            continue
        s = v["summary"]
        pre_m = [r for r in v["stations"] if r["x"] <= X_PRE]
        pre_d = [r for r in dns_rows if r["x"] <= X_PRE]
        print(f"  budget: model D/P {np.mean([r['dissipation_over_production'] for r in pre_m]):.2f} "
              f"vs DNS {np.mean([r['dissipation_over_production'] for r in pre_d]):.2f}; "
              f"nu_t/k model {np.mean([r['nut_over_k_at_Ppeak'] for r in pre_m]):.4f} "
              f"vs DNS {np.mean([r['nut_over_k_at_Ppeak'] for r in pre_d]):.4f}")
        print(f"{name}: k ratio pre-transition "
              f"{s['k_ratio_pre_median']:.2f}x DNS, "
              f"Re_v error {100*s['Re_v_rel_err_pre_mean']:+.0f}%, "
              f"Re_v/Re_theta {s['Re_v_over_Re_theta_pre_mean']:.2f} "
              f"(DNS {s['Re_v_over_Re_theta_pre_mean_dns']:.2f}, "
              f"Blasius 2.19), onset x = {v['x_gamma_half']:.0f}")


if __name__ == "__main__":
    main()
