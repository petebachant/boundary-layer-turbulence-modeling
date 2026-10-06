#!/usr/bin/env python
"""Is transition onset a limit on how much exergy the fluctuations of a
laminar layer can hold?

Exergy here is the energy that could be taken out reversibly. The mean flow's
is all of its kinetic energy. For the fluctuations, take each point's
velocity fluctuations as a Gaussian ensemble with covariance R, the Reynolds
stress: its entropy is ln det R / 2 up to a constant, and the least energy any
state with that entropy can have is the isotropic one's, 3/2 (det R)^(1/3).
What is left,

    X = k - 3/2 (det R)^(1/3) = k (1 - F^(1/3)),

with F = 27 det(b + I/3) Lumley's flatness parameter, is the energy a
reversible, entropy-preserving deformation of the fluctuations could
remove. It is zero for isotropic turbulence, which is as useless as heat,
and k for one- and two-component fluctuations, at the wall and in streaks.
It follows that mean-strain production, which leaves det R unchanged
(tr(R^-1 P) = -2 div U = 0), feeds X with the whole of P_k, and that return
to isotropy by the pressure-strain term is what destroys it.

Tests, fixed before any value was computed:

1. Where the exergy sits: on the JHTDB plate, the layer's exergetic
   fraction, the integral of X over the integral of k up to delta_99, is at
   least EXERGY_RATIO times higher before onset (x from half the onset to
   the onset) than in the turbulent end of the plate (its last third), and
   below FS_ISOTROPY in the free stream (the plate runs' free-stream band,
   1.5 to 2.5 delta_99), where the turbulence should be nearly isotropic.

2. A storage limit: onset is where the layer's fluctuation exergy,
   integral of X dy over U_e^2 theta, first reaches a critical value. Set
   on the plate alone, at its onset, it predicts Wu et al.'s onsets with a
   lower log-rms error in Re_theta than the inlet rule
   Re_theta = C / Tu_in does on the same flows (results/onset-history.json,
   C set on the plate the same way), the bar that rule had to clear.

Also reported: the exergy and its fraction along each flow, the critical
value each Wu flow reaches at its own onset, and the predictions leaving
each Wu flow out instead of using the plate's value.

Wu et al.'s stresses are given in wall units on y / delta at sixteen
stations per flow; outside the stations the exergy is interpolated in
Re_theta.

Outputs
-------
results/fluctuation-exergy.json
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import re

import numpy as np

from pypkg.dns_case import bl_metrics, load_dns

OUT = "results/fluctuation-exergy.json"
WU = "data/wu-bypass-transition"
ONSET = "results/bypass-onset.json"
MECHANICS = "results/transition-mechanics.json"
HISTORY = "results/onset-history.json"
EXERGY_RATIO = 1.5
FS_ISOTROPY = 0.1
FREESTREAM_Y = (1.5, 2.5)


def exergy(uu, vv, ww, uv):
    """Fluctuation exergy and k, for stresses with uw = vw = 0."""
    k = 0.5 * (uu + vv + ww)
    det = np.maximum(ww * (uu * vv - uv**2), 0.0)
    return k - 1.5 * np.cbrt(det), k


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def plate():
    """Along the plate: x, Re_theta, the layer's exergy over U_e^2 theta,
    its exergetic fraction, and the free stream's."""
    d = load_dns()
    nu, x, y = d["nu"], d["x"], d["y"]
    X, k = exergy(d["uu"], d["vv"], d["ww"], d["uv"])
    yw = np.concatenate(([0.0], y))
    rows = []
    for i in range(0, len(x), 4):
        u = d["U"][:, i]
        _, th, _ = bl_metrics(yw, np.concatenate(([0.0], u)), None, nu)
        j = int(np.argmax(u))
        ue = float(u[j])
        d99 = float(np.interp(0.99 * ue, u[: j + 1], y[: j + 1]))
        m = y <= d99
        fs = (y > FREESTREAM_Y[0] * d99) & (y < FREESTREAM_Y[1] * d99)
        ix = np.trapezoid(X[m, i], y[m])
        ik = np.trapezoid(k[m, i], y[m])
        rows.append(
            {
                "x": float(x[i]),
                "re_theta": float(th / nu),
                "exergy": float(ix / (ue**2 * th)),
                "fraction": float(ix / ik),
                "fs_fraction": float(
                    np.trapezoid(X[fs, i], y[fs])
                    / np.trapezoid(k[fs, i], y[fs])
                ),
            }
        )
    return rows


def wu_flow(tag):
    """At each of Wu et al.'s stations: Re_theta, the layer's exergy over
    U_e^2 theta and its exergetic fraction."""
    d = os.path.join(WU, f"stats_{tag}")
    utau = np.loadtxt(f"{d}/Re_theta_versus_utau_{tag}.dat")
    delta = np.loadtxt(f"{d}/Re_theta_versus_delta_{tag}.dat")
    re_theta0 = float(delta[0, 0])
    rows = []
    for f in sorted(
        glob.glob(f"{d}/y_over_delta_versus_urms_plus_at_Re_theta_*_{tag}.dat")
    ):
        rth = float(re.search(r"Re_theta_(\d+)", f).group(1))
        u = np.loadtxt(f)
        eta = u[u[:, 0] <= 1.0, 0]

        def get(name, col=1):
            p = np.loadtxt(f.replace("urms_plus", name))
            return np.interp(eta, p[:, 0], p[:, col])

        ut = float(np.interp(rth, utau[:, 0], utau[:, 1]))
        uu = (u[u[:, 0] <= 1.0, 1] * ut) ** 2
        vv = (get("vrms_plus") * ut) ** 2
        ww = (get("wrms_plus") * ut) ** 2
        uv = -get("-uv_plus") * ut**2
        X, k = exergy(uu, vv, ww, uv)
        # y in inlet momentum thicknesses, as theta = Re_theta / Re_theta0
        y = eta * float(np.interp(rth, delta[:, 0], delta[:, 1]))
        th = rth / re_theta0
        rows.append(
            {
                "re_theta": rth,
                "exergy": float(np.trapezoid(X, y) / th),
                "fraction": float(np.trapezoid(X, y) / np.trapezoid(k, y)),
            }
        )
    return rows


def first_reach(rows, level):
    """Re_theta where the exergy first reaches level, interpolated between
    stations, or None if it never does inside them."""
    r = np.array([s["re_theta"] for s in rows])
    e = np.array([s["exergy"] for s in rows])
    above = np.flatnonzero(e >= level)
    if len(above) == 0 or above[0] == 0:
        return None
    i = int(above[0])
    w = (level - e[i - 1]) / (e[i] - e[i - 1])
    return float(r[i - 1] + w * (r[i] - r[i - 1]))


def log_rms(pairs):
    p, d = np.array(pairs).T
    return float(np.sqrt(np.mean(np.log(p / d) ** 2)))


def main():
    with open(ONSET) as f:
        onset = json.load(f)["cases"]
    with open(MECHANICS) as f:
        x_on = float(json.load(f)["x_transition_onset"])
    with open(HISTORY) as f:
        hist = json.load(f)
    p = plate()
    px = np.array([s["x"] for s in p])
    frac = np.array([s["fraction"] for s in p])
    pre = (px >= 0.5 * x_on) & (px <= x_on)
    late = px >= px[0] + 2.0 / 3.0 * (px[-1] - px[0])
    frac_pre = float(np.mean(frac[pre]))
    frac_late = float(np.mean(frac[late]))
    fs_frac = float(np.mean([s["fs_fraction"] for s in p]))
    critical = float(np.interp(x_on, px, [s["exergy"] for s in p]))
    flows = {tag: wu_flow(tag) for tag in onset}
    cases = {}
    for tag, rows in flows.items():
        r = np.array([s["re_theta"] for s in rows])
        e = np.array([s["exergy"] for s in rows])
        dns = onset[tag]["re_theta_onset"]
        cases[tag] = {
            "dns": dns,
            "exergy_at_dns_onset": (
                float(np.interp(dns, r, e)) if r[0] <= dns <= r[-1] else None
            ),
            "predicted": first_reach(rows, critical),
            "theta_inlet": hist["cases"][tag]["theta_inlet"],
        }
    for tag, c in cases.items():
        others = [
            cases[t]["exergy_at_dns_onset"]
            for t in cases
            if t != tag and cases[t]["exergy_at_dns_onset"]
        ]
        c["predicted_loo"] = first_reach(
            flows[tag], float(np.exp(np.mean(np.log(others))))
        )
    # Compared on the flows both the exergy limit and the inlet rule place
    ok = [
        t
        for t, c in cases.items()
        if c["predicted"] and c["theta_inlet"] and c["dns"]
    ]
    err = log_rms([(cases[t]["predicted"], cases[t]["dns"]) for t in ok])
    err_rule = log_rms([(cases[t]["theta_inlet"], cases[t]["dns"]) for t in ok])
    loo = [t for t, c in cases.items() if c["predicted_loo"] and c["dns"]]
    result = {
        "exergy_ratio_bar": EXERGY_RATIO,
        "fs_isotropy_bar": FS_ISOTROPY,
        "plate": {
            "x_onset": x_on,
            "fraction_pre_onset": frac_pre,
            "fraction_turbulent": frac_late,
            "fraction_ratio": frac_pre / frac_late,
            "fs_fraction": fs_frac,
            "critical_exergy": critical,
            "stations": p,
        },
        "where_passes": bool(
            frac_pre >= EXERGY_RATIO * frac_late and fs_frac < FS_ISOTROPY
        ),
        "cases": cases,
        "wu_stations": flows,
        "n_compared": len(ok),
        "log_rms": err,
        "log_rms_inlet_rule": err_rule,
        "log_rms_loo": log_rms(
            [(cases[t]["predicted_loo"], cases[t]["dns"]) for t in loo]
        )
        if loo
        else None,
        "n_loo": len(loo),
        "limit_passes": bool(len(ok) > 0 and err < err_rule),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        {
            k: v
            for k, v in result.items()
            if k not in ("cases", "wu_stations", "plate")
        }
    )
    print({k: v for k, v in result["plate"].items() if k != "stations"})
    for tag, c in cases.items():
        print(tag, c)


if __name__ == "__main__":
    main()
