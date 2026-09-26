#!/usr/bin/env python
"""The free-stream turbulence length scale of every bypass-transition flow.

Forced lift-up fitted on one flow does not carry to the others in either
direction (results/streak-growth.json), so the forcing depends on more
than the free-stream intensity. The other thing known to move bypass
onset is the free-stream integral length scale. Neither dataset gives it
directly, but the free stream above the layer is close to homogeneous
decaying turbulence, so its decay gives the dissipation,

    eps = -U_inf dk/dx,

and with it the dissipation length scale L = k^(3/2) / eps. Each flow's
free-stream k is fitted as k = A (x - x_v)^(-n) with the virtual origin
free, as the Wu et al. bench cases do, and differentiated analytically:
eps = n k U_inf / (x - x_v), so L = sqrt(k) (x - x_v) / (n U_inf).

  Wu et al.  k = 1.5 u_rms^2 from the u_rms 500-800 inlet momentum
             thicknesses above the plate, assuming isotropy there; lengths
             in inlet momentum thicknesses theta_0, U_inf = 1
  JHTDB      the full k at the top of the domain, in the dataset's units,
             with theta_0 the momentum thickness at the first station

L is reported at the start of each domain and at onset (intermittency 0.1
for Wu et al., the C_f minimum for the plate), over theta_0 and over the
local delta_99.

Outputs
-------
results/freestream-length-scale.json
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import least_squares

from pypkg.dns_case import load_dns

ROOT = "data/wu-bypass-transition"
ONSET = "results/bypass-onset.json"
MECHANICS = "results/transition-mechanics.json"
OUT = "results/freestream-length-scale.json"
TAGS = ("WM075", "WM150", "WM225", "WM300", "WM600")


def fit_decay(x, k):
    """(A, x_v, n) of k = A (x - x_v)^(-n), least squares in log k."""
    m = k > 0
    x, k = x[m], k[m]

    def resid(p):
        logA, xv, n = p
        return logA - n * np.log(np.maximum(x - xv, 1e-9)) - np.log(k)

    r = least_squares(resid, [np.log(k[0]) + 1.3 * np.log(x[0] + 1.0),
                              min(0.0, x[0] - 1.0), 1.3],
                      bounds=([-50, -1e7, 0.1], [50, x[0] - 1e-6, 5.0]))
    logA, xv, n = r.x
    pred = logA - n * np.log(x - xv)
    return {"A": float(np.exp(logA)), "x_v": float(xv), "n": float(n),
            "log_rms": float(np.sqrt(np.mean((pred - np.log(k)) ** 2)))}


def length_scale(fit, x, ue=1.0):
    k = fit["A"] * (x - fit["x_v"]) ** (-fit["n"])
    return float(np.sqrt(k) * (x - fit["x_v"]) / (fit["n"] * ue)), float(k)


def wu(tag, onset):
    d = os.path.join(ROOT, f"stats_{tag}")
    load = lambda n: np.loadtxt(f"{d}/{n}_{tag}.dat")  # noqa: E731
    rth_delta = load("Re_theta_versus_delta")
    rex_rth = load("Re_x_versus_Re_theta")
    urms = load("Re_x_versus_urms_at_y_500_to_800_theta_0")
    re_theta0 = float(rth_delta[0, 0])
    x = urms[:, 0] / re_theta0
    fit = fit_decay(x, 1.5 * urms[:, 1] ** 2)

    def at(x_):
        L, k = length_scale(fit, x_)
        rth = float(np.interp(x_ * re_theta0, rex_rth[:, 0], rex_rth[:, 1]))
        delta = float(np.interp(rth, rth_delta[:, 0], rth_delta[:, 1]))
        return {"x": float(x_), "re_theta": rth, "L_over_theta0": L,
                "L_over_delta99": L / delta,
                "tu_percent": 100.0 * float(np.sqrt(2.0 * k / 3.0))}

    x0 = float(max(x[0], rex_rth[0, 0] / re_theta0))
    rex_on = onset[tag]["re_x_onset"]
    return {"tu_inlet_percent": onset[tag]["tu_inlet_percent"],
            "decay_fit": fit, "start": at(x0),
            "onset": at(rex_on / re_theta0) if rex_on is not None else None}


def jhtdb(x_on):
    d = load_dns()
    x, y, U = d["x"], d["y"], d["U"]
    ue = U[-1, :]
    k = d["k"][-1, :]
    fit = fit_decay(x, k)

    def at(i):
        u = U[:, i]
        j = int(np.argmax(u))
        d99 = float(np.interp(0.99 * u[j], u[: j + 1], y[: j + 1]))
        f = np.clip(u[: j + 1] / u[j], 0, 1)
        th = float(np.trapezoid(f * (1 - f), y[: j + 1]))
        L, kk = length_scale(fit, x[i], float(ue[i]))
        return {"x": float(x[i]), "L": L, "delta99": d99, "theta": th,
                "L_over_delta99": L / d99,
                "tu_percent": 100.0 * float(np.sqrt(2.0 * kk / 3.0)) / ue[i]}

    s = at(0)
    o = at(int(np.argmin(np.abs(x - x_on))))
    for r in (s, o):
        r["L_over_theta0"] = r["L"] / s["theta"]
    return {"decay_fit": fit, "start": s, "onset": o}


def main():
    with open(ONSET) as f:
        onset = json.load(f)["cases"]
    with open(MECHANICS) as f:
        x_on = json.load(f)["x_transition_onset"]
    cases = {t: wu(t, onset) for t in TAGS}
    j = jhtdb(x_on)
    wu_on = [c["onset"]["L_over_delta99"] for c in cases.values()
             if c["onset"] is not None]
    result = {
        "wu": cases, "jhtdb": j,
        "wu_L_over_delta99_onset_min": float(min(wu_on)),
        "wu_L_over_delta99_onset_max": float(max(wu_on)),
        "jhtdb_L_over_delta99_onset": j["onset"]["L_over_delta99"],
        "wu_L_over_theta0_start_min": float(min(
            c["start"]["L_over_theta0"] for c in cases.values())),
        "wu_L_over_theta0_start_max": float(max(
            c["start"]["L_over_theta0"] for c in cases.values())),
        "jhtdb_L_over_theta0_start": j["start"]["L_over_theta0"],
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for t, c in cases.items():
        print(t, "fit", {k: round(v, 3) for k, v in c["decay_fit"].items()},
              "start", {k: round(v, 3) for k, v in c["start"].items()},
              "onset", c["onset"] and {k: round(v, 3)
                                       for k, v in c["onset"].items()})
    print("JHTDB fit", {k: round(v, 3) for k, v in j["decay_fit"].items()})
    print("  start", {k: round(v, 3) for k, v in j["start"].items()})
    print("  onset", {k: round(v, 3) for k, v in j["onset"].items()})


if __name__ == "__main__":
    main()
