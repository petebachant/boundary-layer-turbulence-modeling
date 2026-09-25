#!/usr/bin/env python
"""A transition threshold that depends on free-stream turbulence, fitted on
Wu et al.'s five DNS and tested on the JHTDB plate, which it never saw.

The peak vorticity Reynolds number at onset falls as the free-stream
intensity rises (results/bypass-onset.json), so the clipping closure's
threshold should too. Fitted here as a power law in the local free-stream
intensity at onset,

    Lambda_c(Tu) = A Tu^(-m),   Tu in percent,

because a closure knows the free stream where it is, not at an inlet that
may be anywhere upstream. Both datasets are read the same way:

  onset  the minimum of C_f
  Tu     the free-stream u_rms at onset over the free-stream speed, in
         percent; Wu et al. give it measured 500-800 inlet momentum
         thicknesses above the plate, and the JHTDB plate's is taken from
         its profiles well outside the layer
  Re_v   the peak over the layer of y^2 |dU/dy| / nu at onset

Test, fixed before the JHTDB value was computed: the law, evaluated at the
JHTDB plate's Tu at onset, predicts the plate's Re_v,max at onset within
TRANSFER_TOL.

Exploratory, defined after that test failed: the same law in the inlet
intensity rather than the local one, fitted to the onset Re_v of
results/bypass-onset.json (intermittency 0.1, which is defined for every
case, where the C_f minimum is not when a case transitions at its inlet),
and evaluated at the JHTDB plate's inlet intensity. A match there would say
onset depends on how long the streaks have been forced, not on the
intensity where they break down.

Outputs
-------
results/threshold-law.json
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from pypkg.dns_case import load_dns

ROOT = "data/wu-bypass-transition"
MECHANICS = "results/transition-mechanics.json"
ONSET = "results/bypass-onset.json"
INLET = "results/inlet-profiles.json"
OUT = "results/threshold-law.json"
TRANSFER_TOL = 0.15
#: The JHTDB plate's free stream, in multiples of delta_99 above the wall
FREESTREAM_Y = (1.5, 2.5)


def wu_case(tag):
    d = os.path.join(ROOT, f"stats_{tag}")
    cf = np.loadtxt(f"{d}/Re_x_versus_cf_{tag}.dat")
    rex_rth = np.loadtxt(f"{d}/Re_x_versus_Re_theta_{tag}.dat")
    rth_delta = np.loadtxt(f"{d}/Re_theta_versus_delta_{tag}.dat")
    urms = np.loadtxt(f"{d}/Re_x_versus_urms_at_y_500_to_800_theta_0_{tag}.dat")
    re_theta0 = float(rth_delta[0, 0])
    i = int(np.argmin(cf[:, 1]))
    rex_on = float(cf[i, 0])
    rth_on = float(np.interp(rex_on, rex_rth[:, 0], rex_rth[:, 1]))
    tu_on = 100.0 * float(np.interp(rex_on, urms[:, 0], urms[:, 1]))
    rth_s, rev_s = [], []
    for f in sorted(glob.glob(f"{d}/y_over_delta_versus_U_at_Re_theta_*"
                              f"_{tag}.dat")):
        rth = float(re.search(r"Re_theta_(\d+)", f).group(1))
        prof = np.loadtxt(f)
        eta, u = prof[:, 0], prof[:, 1]
        m = eta <= 1.0
        delta = float(np.interp(rth, rth_delta[:, 0], rth_delta[:, 1]))
        rth_s.append(rth)
        rev_s.append(float(np.max(eta[m] ** 2 * np.gradient(u[m], eta[m]))
                           * delta * re_theta0))
    rth_s, rev_s = np.array(rth_s), np.array(rev_s)
    inside = rth_s.min() <= rth_on <= rth_s.max()
    return {
        "tu_inlet_percent": int(tag[2:]) / 100.0,
        "re_x_onset": rex_on, "re_theta_onset": rth_on,
        "tu_onset_percent": tu_on,
        "re_v_max_onset": float(np.interp(rth_on, rth_s, rev_s))
        if inside else None,
    }


def jhtdb_onset():
    with open(MECHANICS) as f:
        mech = json.load(f)
    x_on = mech["x_transition_onset"]
    d = load_dns()
    x, y, U, uu = d["x"], d["y"], d["U"], d["uu"]
    i = int(np.argmin(np.abs(x - x_on)))
    u = U[:, i]
    j = int(np.argmax(u))
    ue = float(u[j])
    d99 = float(np.interp(0.99 * ue, u[: j + 1], y[: j + 1]))
    fs = (y > FREESTREAM_Y[0] * d99) & (y < FREESTREAM_Y[1] * d99)
    return {
        "x_onset": x_on,
        "tu_onset_percent": 100.0 * float(np.sqrt(uu[fs, i].mean())) / ue,
        "re_v_max_onset": mech["re_v_max_at_onset"],
    }


def main():
    tags = sorted(os.path.basename(p)[len("stats_"):]
                  for p in glob.glob(os.path.join(ROOT, "stats_WM*")))
    cases = {t: wu_case(t) for t in tags}
    fit = [c for c in cases.values() if c["re_v_max_onset"] is not None]
    tu = np.array([c["tu_onset_percent"] for c in fit])
    rev = np.array([c["re_v_max_onset"] for c in fit])
    slope, icpt = np.polyfit(np.log(tu), np.log(rev), 1)
    pred = icpt + slope * np.log(tu)
    r2 = float(1 - np.sum((np.log(rev) - pred) ** 2)
               / np.sum((np.log(rev) - np.log(rev).mean()) ** 2))
    A, m = float(np.exp(icpt)), float(-slope)
    j = jhtdb_onset()
    law_at_jhtdb = A * j["tu_onset_percent"] ** (-m)
    err = law_at_jhtdb / j["re_v_max_onset"] - 1
    # Exploratory: in the inlet intensity
    with open(ONSET) as f:
        onset = json.load(f)["cases"]
    pts = [(c["tu_inlet_percent"], c["re_v_max_onset"]) for c in onset.values()
           if c["re_v_max_onset"] is not None]
    ti = np.log([p[0] for p in pts])
    ri = np.log([p[1] for p in pts])
    s_in, c_in = np.polyfit(ti, ri, 1)
    r2_in = float(1 - np.sum((ri - (c_in + s_in * ti)) ** 2)
                  / np.sum((ri - ri.mean()) ** 2))
    with open(INLET) as f:
        tu_in_jhtdb = json.load(f)["Tu_inlet_percent"]
    law_in = float(np.exp(c_in) * tu_in_jhtdb ** s_in)
    err_in = law_in / j["re_v_max_onset"] - 1
    exploratory = {
        "A_inlet": float(np.exp(c_in)), "m_inlet": float(-s_in),
        "fit_r2_log_inlet": r2_in, "n_fit_inlet": len(pts),
        "jhtdb_tu_inlet_percent": tu_in_jhtdb,
        "law_inlet_at_jhtdb": law_in,
        "law_inlet_error_at_jhtdb": err_in,
        "inlet_law_transfers": bool(abs(err_in) <= TRANSFER_TOL),
    }
    result = {
        "exploratory": exploratory,
        "settings": {"transfer_tol": TRANSFER_TOL,
                     "freestream_y_over_d99": list(FREESTREAM_Y)},
        "wu_cases": cases,
        "n_fit": len(fit),
        "A": A, "m": m, "fit_r2_log": r2,
        "jhtdb": j,
        "law_at_jhtdb": law_at_jhtdb,
        "law_error_at_jhtdb": err,
        "transfers": bool(abs(err) <= TRANSFER_TOL),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for t, c in cases.items():
        print(t, {k: round(v, 3) if isinstance(v, float) else v
                  for k, v in c.items()})
    print(f"Lambda_c(Tu) = {A:.1f} Tu^-{m:.3f}  (R^2 {r2:.3f}, n {len(fit)})")
    print(f"JHTDB: Tu at onset {j['tu_onset_percent']:.2f}%, Re_v,max "
          f"{j['re_v_max_onset']:.0f}, law {law_at_jhtdb:.0f} "
          f"({err:+.1%}), transfers {result['transfers']}")
    print("exploratory:", exploratory)


if __name__ == "__main__":
    main()
