#!/usr/bin/env python
"""Transition onset against free-stream intensity, from Wu et al.'s DNS.

The question (calkit.yaml): is onset set by the disturbance reaching a fixed
headroom, free-stream intensity times transient-growth gain, rather than by
a Reynolds number alone? Lift-up's gain grows like Re_x^(1/2), so streak
amplitude grows like Tu Re_x^(1/2), and a fixed headroom puts onset where
Tu^2 Re_x is constant: Re_x,t proportional to Tu^(-n) with n near 2. A
threshold on the Reynolds number alone would instead hold the peak
vorticity Reynolds number at onset near one value whatever Tu is.

Tests, fixed before the data were looked at:

  onset      the first Re_x at which the intermittency reaches ONSET_GAMMA
  headroom   the fitted exponent n, over the five inlet intensities, lies in
             N_RANGE, and the peak Re_v at onset varies between intensities
             by more than RE_V_SPREAD of its smallest value

Re_v,max at a station is max over the layer of eta^2 dU/deta times Re_delta,
with eta = y/delta, U in units of the free stream, and Re_delta = (delta /
theta_0) Re_theta_0, from the profiles the dataset gives at fixed Re_theta;
at onset it is interpolated between the profile stations either side.

Outputs
-------
results/bypass-onset.json
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

ROOT = "data/wu-bypass-transition"
OUT = "results/bypass-onset.json"
ONSET_GAMMA = 0.1
N_RANGE = (1.5, 2.5)
RE_V_SPREAD = 0.2


def load(path):
    return np.loadtxt(path)


def first_crossing(x, f, level):
    above = np.flatnonzero(f >= level)
    if len(above) == 0:
        return None
    i = above[0]
    if i == 0:
        return float(x[0])
    return float(np.interp(level, [f[i - 1], f[i]], [x[i - 1], x[i]]))


def case(tag):
    d = os.path.join(ROOT, f"stats_{tag}")
    # The case tag is the inlet intensity in percent, e.g., WM150 is 1.5
    tu = int(tag[2:]) / 100.0
    rex_gamma = load(f"{d}/Re_x_versus_intermittency_{tag}.dat")
    rex_rth = load(f"{d}/Re_x_versus_Re_theta_{tag}.dat")
    rth_delta = load(f"{d}/Re_theta_versus_delta_{tag}.dat")
    re_theta0 = float(rth_delta[0, 0])
    rex_t = first_crossing(rex_gamma[:, 0], rex_gamma[:, 1], ONSET_GAMMA)
    rth_t = (float(np.interp(rex_t, rex_rth[:, 0], rex_rth[:, 1]))
             if rex_t is not None else None)
    stations = []
    for f in sorted(glob.glob(f"{d}/y_over_delta_versus_U_at_Re_theta_*"
                              f"_{tag}.dat")):
        rth = float(re.search(r"Re_theta_(\d+)", f).group(1))
        prof = load(f)
        eta, u = prof[:, 0], prof[:, 1]
        m = eta <= 1.0
        du = np.gradient(u[m], eta[m])
        delta = float(np.interp(rth, rth_delta[:, 0], rth_delta[:, 1]))
        re_delta = delta * re_theta0
        stations.append({"re_theta": rth,
                         "re_v_max": float(np.max(eta[m] ** 2 * du)
                                           * re_delta)})
    rth_s = np.array([s["re_theta"] for s in stations])
    rev_s = np.array([s["re_v_max"] for s in stations])
    rev_t = (float(np.interp(rth_t, rth_s, rev_s))
             if rth_t is not None and rth_s.min() <= rth_t <= rth_s.max()
             else None)
    return {
        "tu_inlet_percent": tu, "re_theta_0": re_theta0,
        "re_x_onset": rex_t, "re_theta_onset": rth_t,
        "re_v_max_onset": rev_t,
        # Tu in percent, so this is 10^4 times the fraction's
        "tu2_re_x_onset": tu ** 2 * rex_t if rex_t is not None else None,
        "profile_stations": stations,
    }


def main():
    tags = sorted(os.path.basename(p)[len("stats_"):]
                  for p in glob.glob(os.path.join(ROOT, "stats_WM*")))
    cases = {t: case(t) for t in tags}
    ok = [c for c in cases.values() if c["re_x_onset"] is not None]
    tu = np.array([c["tu_inlet_percent"] for c in ok])
    rex = np.array([c["re_x_onset"] for c in ok])
    slope, _ = np.polyfit(np.log(tu), np.log(rex), 1)
    n = float(-slope)
    rev = [c["re_v_max_onset"] for c in ok if c["re_v_max_onset"] is not None]
    spread = float(max(rev) / min(rev) - 1) if len(rev) > 1 else None
    tu2 = np.array([c["tu2_re_x_onset"] for c in ok])
    result = {
        "settings": {"onset_gamma": ONSET_GAMMA, "n_range": list(N_RANGE),
                     "re_v_spread": RE_V_SPREAD},
        "cases": cases,
        "n_cases": len(ok),
        "onset_exponent_n": n,
        "tu2_re_x_cv": float(tu2.std() / tu2.mean()),
        "re_v_max_onset_min": float(min(rev)) if rev else None,
        "re_v_max_onset_max": float(max(rev)) if rev else None,
        "re_v_max_onset_spread": spread,
        "n_in_range": bool(N_RANGE[0] <= n <= N_RANGE[1]),
        "re_v_varies": bool(spread is not None and spread > RE_V_SPREAD),
        "headroom_supported": bool(N_RANGE[0] <= n <= N_RANGE[1]
                                   and spread is not None
                                   and spread > RE_V_SPREAD),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for t, c in cases.items():
        print(f"{t}: Tu {c['tu_inlet_percent']:.2f}% Re_x,t {c['re_x_onset']} "
              f"Re_theta,t {c['re_theta_onset']} Re_v,max {c['re_v_max_onset']}"
              f" Tu^2 Re_x {c['tu2_re_x_onset']}")
    print({k: v for k, v in result.items() if k not in ("cases",)})


if __name__ == "__main__":
    main()
