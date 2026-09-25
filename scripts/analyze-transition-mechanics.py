#!/usr/bin/env python
"""Integral balances and saturation through bypass transition on the plate.

Three questions from the overdriven-amplifier picture (ideas-log section
7.7), all answerable from the JHTDB time-averaged profiles:

1. Does the layer clip at a fixed local Reynolds number? The peak vorticity
   Reynolds number, Re_v,max = max over the layer of y^2 S / nu, against x.
   Test: over the transition region, from the minimum of C_f to its maximum
   (the usual markers of onset and completion), Re_v,max varies by less
   than CLIP_TOL of its value, while the laminar trend fitted upstream
   would have grown by more. Also reported, because a first look at the
   profiles suggested it before this test was written: the same measure
   over the first half of the transition region only.

2. Does the mean-flow distortion act as gain control? The production
   efficiency, layer-integrated production per unit layer-integrated k per
   outer time scale, P_int / (K_int U_e / delta_99). Test: it is lower at
   the end of the transition region than at its start while K_int is
   higher, i.e., the fluctuations grow and the production per unit of them
   falls.

3. Where does the integral dissipation coefficient leave its laminar value?
   C_D = int(tau dU/dy) dy / U_e^3, split into the viscous part
   int(nu (dU/dy)^2) dy / U_e^3 and the turbulent part int(-u'v' dU/dy) dy /
   U_e^3. Its laminar value is a fit of C_D proportional to Re_x^(-1/2) over
   the laminar stretch LAMINAR_X, and it departs where it first exceeds that
   by DEPART_TOL. Compared with where Re_v,max first reaches the classical
   threshold of 440, and with where the displacement-thickness Reynolds
   number reaches 520, the linear (Orr-Sommerfeld) critical value for the
   Blasius profile. Energy stability implies linear stability, so the
   energy-stability bound lies at or upstream of that point.

Outputs
-------
results/transition-mechanics.json
"""

from __future__ import annotations

import json

import numpy as np

from pypkg.dns_case import load_dns

OUT = "results/transition-mechanics.json"
X_MIN = 60.0
CLIP_TOL = 0.10
LAMINAR_X = (60.0, 120.0)
DEPART_TOL = 0.10
RE_V_CLASSICAL = 440.0
RE_DSTAR_LINEAR_CRITICAL = 520.0


def first_crossing(x, f, level):
    above = np.flatnonzero(f >= level)
    if len(above) == 0:
        return None
    i = above[0]
    if i == 0:
        return float(x[0])
    return float(np.interp(level, [f[i - 1], f[i]], [x[i - 1], x[i]]))


def main():
    d = load_dns()
    x, y, U, uu, uv, k, nu = (d["x"], d["y"], d["U"], d["uu"], d["uv"],
                              d["k"], d["nu"])
    dUdy = np.gradient(U, y, axis=0)
    cols = []
    for i in range(len(x)):
        if x[i] < X_MIN:
            continue
        u = U[:, i]
        j = int(np.argmax(u))
        ue = float(u[j])
        d99 = float(np.interp(0.99 * ue, u[: j + 1], y[: j + 1]))
        m = y <= d99
        r = np.clip(u[m] / ue, 0, 1)
        ym = y[m]
        dstar = float(np.trapezoid(1 - r, ym))
        theta = float(np.trapezoid(r * (1 - r), ym))
        delta3 = float(np.trapezoid(r * (1 - r ** 2), ym))
        s = dUdy[m, i]
        rev = ym ** 2 * np.abs(s) / nu
        prod = float(np.trapezoid(-uv[m, i] * s, ym))
        kint = float(np.trapezoid(k[m, i], ym))
        cols.append({
            "x": float(x[i]), "Ue": ue, "delta99": d99,
            "Re_x": float(x[i] * ue / nu),
            "Re_dstar": dstar * ue / nu, "Re_theta": theta * ue / nu,
            "theta": theta, "delta3": delta3,
            "cf": float(2 * nu * dUdy[0, i] / ue ** 2),
            "re_v_max": float(rev.max()),
            "u_rms_max": float(np.sqrt(max(uu[m, i].max(), 0.0)) / ue),
            "P_int": prod, "K_int": kint,
            "production_efficiency": prod / (kint * ue / d99)
            if kint > 0 else None,
            "cd_viscous": float(np.trapezoid(nu * s ** 2, ym)) / ue ** 3,
            "cd_turbulent": prod / ue ** 3,
        })
    xs = np.array([c["x"] for c in cols])
    cf = np.array([c["cf"] for c in cols])
    rev = np.array([c["re_v_max"] for c in cols])
    urms = np.array([c["u_rms_max"] for c in cols])
    eff = np.array([c["production_efficiency"] for c in cols], float)
    kint = np.array([c["K_int"] for c in cols])
    rex = np.array([c["Re_x"] for c in cols])
    cd = np.array([c["cd_viscous"] + c["cd_turbulent"] for c in cols])
    for c, v in zip(cols, cd):
        c["cd"] = float(v)

    # Transition region from the skin friction
    i_min = int(np.argmin(cf))
    i_max = i_min + int(np.argmax(cf[i_min:]))
    x_on, x_end = float(xs[i_min]), float(xs[i_max])
    region = slice(i_min, i_max + 1)
    half = slice(i_min, (i_min + i_max) // 2 + 1)

    # 1. Clipping of the local Reynolds number
    lam = (xs >= LAMINAR_X[0]) & (xs <= LAMINAR_X[1])
    # Laminar trend: Re_v,max grows like Re_x^(1/2) in a similarity layer
    c_lam = float(np.mean(rev[lam] / np.sqrt(rex[lam])))
    grew_lam = float(c_lam * np.sqrt(rex[i_max]) / (c_lam * np.sqrt(rex[i_min]))
                     - 1)

    def variation(sl):
        v = rev[sl]
        return float(v.max() / v.min() - 1)

    var_full = variation(region)
    var_half = variation(half)
    clip = bool(var_full < CLIP_TOL and grew_lam > CLIP_TOL)

    # 2. Gain control
    gain_control = bool(eff[i_max] < eff[i_min] and kint[i_max] > kint[i_min])

    # 3. Departure of the dissipation coefficient from laminar
    a_lam = float(np.mean(cd[lam] * np.sqrt(rex[lam])))
    cd_lam = a_lam / np.sqrt(rex)
    x_depart = first_crossing(xs, cd / cd_lam, 1 + DEPART_TOL)
    x_rev = first_crossing(xs, rev, RE_V_CLASSICAL)
    re_dstar = np.array([c["Re_dstar"] for c in cols])
    x_linear = first_crossing(xs, re_dstar, RE_DSTAR_LINEAR_CRITICAL)

    result = {
        "settings": {"x_min": X_MIN, "clip_tol": CLIP_TOL,
                     "laminar_x": list(LAMINAR_X), "depart_tol": DEPART_TOL,
                     "re_v_classical": RE_V_CLASSICAL,
                     "re_dstar_linear_critical": RE_DSTAR_LINEAR_CRITICAL},
        "x_transition_onset": x_on,
        "x_transition_end": x_end,
        "re_v_max_at_onset": float(rev[i_min]),
        "re_v_max_at_end": float(rev[i_max]),
        "re_v_max_variation_transition": var_full,
        "re_v_max_variation_first_half": var_half,
        "re_v_max_mean_first_half": float(rev[half].mean()),
        "laminar_trend_growth_over_transition": grew_lam,
        "clips": clip,
        "x_streak_amplitude_peak": float(xs[int(np.argmax(urms))]),
        "streak_amplitude_peak": float(urms.max()),
        "streak_amplitude_at_end": float(urms[i_max]),
        "production_efficiency_at_onset": float(eff[i_min]),
        "production_efficiency_at_end": float(eff[i_max]),
        "k_int_at_onset": float(kint[i_min]),
        "k_int_at_end": float(kint[i_max]),
        "gain_control": gain_control,
        "x_cd_departs_laminar": x_depart,
        "x_re_v_reaches_classical": x_rev,
        "x_re_dstar_linear_critical": x_linear,
        "cd_departs_near_re_v_threshold": bool(
            x_depart is not None and x_rev is not None
            and abs(x_depart - x_rev) < abs(x_depart - (x_linear or 0.0))),
        "stations": cols[::8],
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for key in ("x_transition_onset", "x_transition_end",
                "re_v_max_at_onset", "re_v_max_at_end",
                "re_v_max_variation_transition",
                "re_v_max_variation_first_half",
                "laminar_trend_growth_over_transition", "clips",
                "x_streak_amplitude_peak", "production_efficiency_at_onset",
                "production_efficiency_at_end", "gain_control",
                "x_cd_departs_laminar", "x_re_v_reaches_classical",
                "x_re_dstar_linear_critical"):
        print(f"{key:42s} {result[key]}")


if __name__ == "__main__":
    main()
