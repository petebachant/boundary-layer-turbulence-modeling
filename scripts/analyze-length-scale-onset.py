#!/usr/bin/env python
"""At fixed free-stream intensity, how much does the free-stream length scale
move bypass onset, and is it the length scale or the decay?

Three calibrations pull the JHTDB plate and Wu et al.'s flows apart, and the
plate's streaky inlet is not why (results/plate-quiet-inlet-test.json),
which leaves the free stream: its length scale and its decay. Bienner et
al.'s two Mach 0.1 air LES (data/bienner-bypass-les) share an inlet
intensity of about 4 percent and differ in integral length scale by a
factor of seven; the smaller-scale free stream also decays faster.

For each: onset as the C_f minimum, with Re_x and Re_theta there; the peak
u_rms in the layer at onset; and the free-stream dissipation length scale
k^(3/2)/eps from the decay of k = 1.5 (Tu U)^2, as in
results/freestream-length-scale.json, in inlet momentum thicknesses.

Tests, fixed before any of this was computed:

  direction  the larger length scale brings onset forward (smaller Re_theta
             at onset), as Brandt et al. found in DNS
  decay      Langtry-Menter's correlation, applied at each run's local
             free-stream intensity, sees the two runs only through their
             different decay; if the ratio of the onset Re_theta it predicts
             (large scale over small) is within DECAY_TOL of the LES's
             ratio, the decay explains the shift

Outputs
-------
results/length-scale-onset.json
"""

from __future__ import annotations

import json

import numpy as np

ROOT = "data/bienner-bypass-les/evol_streamwise"
RUNS = {"small": "low_Lf_Tu4_Air_M0p1", "large": "high_Lf_Tu4_Air_M0p1"}
OUT = "results/length-scale-onset.json"
LENGTHS = "results/freestream-length-scale.json"
DECAY_TOL = 0.2
COLS = ("re_x", "re_theta", "re_tau", "dstar_d99", "theta_d99", "H",
        "utau", "rho_w", "cf", "urms", "y_urms", "tu", "trms")


def load(run):
    rows = [l.split() for l in open(f"{ROOT}/stats_vs_x_{run}.dat")
            if l.strip() and l.strip()[0].isdigit()]
    a = np.array(rows, dtype=float)
    return {c: a[:, i] for i, c in enumerate(COLS)}


def lm_re_theta_t(tu):
    tu = max(tu, 0.027)
    if tu <= 1.3:
        return 1173.51 - 589.428 * tu + 0.2196 / tu ** 2
    return 331.50 * (tu - 0.5658) ** -0.671


def first_reach(x, f, target):
    d = f - target
    above = np.flatnonzero(d >= 0)
    if len(above) == 0:
        return None
    i = int(above[0])
    if i == 0:
        return float(x[0])
    w = d[i - 1] / (d[i - 1] - d[i])
    return float(x[i - 1] + w * (x[i] - x[i - 1]))


def dissipation_length(re_x, k):
    """L = k^(3/2)/eps with eps = -U dk/dx, from a power-law fit about a
    free virtual origin, in units of nu/U, at the first station."""
    from scipy.optimize import least_squares

    def resid(p):
        logA, xv, n = p
        return logA - n * np.log(np.maximum(re_x - xv, 1e-9)) - np.log(k)

    r = least_squares(resid, [np.log(k[0]) + np.log(re_x[0]), 0.0, 1.3],
                      bounds=([-80, -1e8, 0.1], [80, re_x[0] - 1.0, 6.0]))
    _, xv, n = r.x
    return float(np.sqrt(k[0]) * (re_x[0] - xv) / n)


def main():
    out = {}
    for name, run in RUNS.items():
        d = load(run)
        # Past the first tenth, so an inlet transient cannot be taken for
        # the C_f minimum
        m = d["re_x"] >= d["re_x"][0] + 0.1 * (d["re_x"][-1] - d["re_x"][0])
        i = int(np.flatnonzero(m)[0] + np.argmin(d["cf"][m]))
        tu = 100.0 * d["tu"]
        k = 1.5 * d["tu"] ** 2
        L = dissipation_length(d["re_x"], k)
        lm_x = first_reach(d["re_x"], d["re_theta"],
                           np.array([lm_re_theta_t(t) for t in tu]))
        out[name] = {
            "run": run, "tu_inlet_percent": float(tu[0]),
            "re_theta_inlet": float(d["re_theta"][0]),
            "re_x_onset": float(d["re_x"][i]),
            "re_theta_onset": float(d["re_theta"][i]),
            "tu_onset_percent": float(tu[i]),
            "tu_end_percent": float(tu[-1]),
            "urms_peak_onset": float(d["urms"][i]),
            "L_over_theta_inlet": L / float(d["re_theta"][0]),
            "lm_re_theta_onset": (float(np.interp(lm_x, d["re_x"],
                                                  d["re_theta"]))
                                  if lm_x is not None else None),
        }
    with open(LENGTHS) as f:
        lengths = json.load(f)
    s, l = out["small"], out["large"]
    les_ratio = l["re_theta_onset"] / s["re_theta_onset"]
    lm_ratio = (l["lm_re_theta_onset"] / s["lm_re_theta_onset"]
                if l["lm_re_theta_onset"] and s["lm_re_theta_onset"]
                else None)
    result = {
        "runs": out, "decay_tol": DECAY_TOL,
        "length_ratio": l["L_over_theta_inlet"] / s["L_over_theta_inlet"],
        "les_onset_ratio": les_ratio, "lm_onset_ratio": lm_ratio,
        "larger_scale_earlier": bool(les_ratio < 1.0),
        "decay_explains": bool(lm_ratio is not None
                               and abs(lm_ratio / les_ratio - 1) <= DECAY_TOL),
        "plate_L_over_theta_inlet": lengths["jhtdb_L_over_theta0_start"],
        "wu_L_over_theta_inlet_min": lengths["wu_L_over_theta0_start_min"],
        "wu_L_over_theta_inlet_max": lengths["wu_L_over_theta0_start_max"],
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
