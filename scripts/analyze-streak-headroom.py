#!/usr/bin/env python
"""Is there a fixed streak amplitude at which bypass transition begins?

If onset is a disturbance reaching a fixed headroom, the history a
threshold needs is already in the flow: the peak streamwise fluctuation in
the layer integrates how long and how hard the free stream has forced it.
Then the peak u_rms at onset is about the same whatever the free-stream
intensity, and a closure can switch on at a fixed amplitude of its own
fluctuation energy, with nothing to know about an inlet.

Tests, fixed before the amplitudes were computed:

  universal   the peak u_rms / U_inf at onset varies by less than SPREAD
              across Wu et al.'s cases that have profiles at onset
  transfers   the JHTDB plate's peak u_rms / U_e at onset lies within
              SPREAD of their mean

Onset for Wu et al. is where the intermittency reaches 0.1
(results/bypass-onset.json), with the peak interpolated between the
profile stations either side; u_rms / U_inf = u_rms^+ u_tau / U_inf. The
JHTDB plate has no intermittency, so its onset is the C_f minimum
(results/transition-mechanics.json).

Outputs
-------
results/streak-headroom.json
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from pypkg.dns_case import load_dns

ROOT = "data/wu-bypass-transition"
ONSET = "results/bypass-onset.json"
MECHANICS = "results/transition-mechanics.json"
OUT = "results/streak-headroom.json"
SPREAD = 0.2


def wu_peak(tag, rth_on):
    d = os.path.join(ROOT, f"stats_{tag}")
    utau = np.loadtxt(f"{d}/Re_theta_versus_utau_{tag}.dat")
    rth_s, peak = [], []
    for f in sorted(glob.glob(f"{d}/y_over_delta_versus_urms_plus_at_"
                              f"Re_theta_*_{tag}.dat")):
        rth = float(re.search(r"Re_theta_(\d+)", f).group(1))
        p = np.loadtxt(f)
        inside = p[:, 0] <= 1.0
        ut = float(np.interp(rth, utau[:, 0], utau[:, 1]))
        rth_s.append(rth)
        peak.append(float(p[inside, 1].max()) * ut)
    rth_s, peak = np.array(rth_s), np.array(peak)
    stations = [{"re_theta": float(r), "u_rms_peak": float(v)}
                for r, v in zip(rth_s, peak)]
    if rth_on is None or not rth_s.min() <= rth_on <= rth_s.max():
        return None, stations
    return float(np.interp(rth_on, rth_s, peak)), stations


def main():
    with open(ONSET) as f:
        onset = json.load(f)["cases"]
    cases = {}
    for tag, c in onset.items():
        amp, stations = wu_peak(tag, c["re_theta_onset"])
        cases[tag] = {"tu_inlet_percent": c["tu_inlet_percent"],
                      "re_theta_onset": c["re_theta_onset"],
                      "u_rms_peak_onset": amp, "stations": stations}
    vals = [c["u_rms_peak_onset"] for c in cases.values()
            if c["u_rms_peak_onset"] is not None]
    mean = float(np.mean(vals))
    spread = float(max(vals) / min(vals) - 1)
    with open(MECHANICS) as f:
        x_on = json.load(f)["x_transition_onset"]
    d = load_dns()
    i = int(np.argmin(np.abs(d["x"] - x_on)))
    ue = float(d["U"][:, i].max())
    j_amp = float(np.sqrt(max(d["uu"][:, i].max(), 0.0))) / ue
    err = j_amp / mean - 1
    result = {
        "settings": {"spread": SPREAD},
        "cases": cases,
        "n_cases": len(vals),
        "u_rms_peak_onset_mean": mean,
        "u_rms_peak_onset_min": float(min(vals)),
        "u_rms_peak_onset_max": float(max(vals)),
        "u_rms_peak_onset_spread": spread,
        "universal": bool(spread < SPREAD),
        "jhtdb_x_onset": x_on,
        "jhtdb_u_rms_peak_onset": j_amp,
        "jhtdb_error_from_mean": err,
        "transfers": bool(abs(err) < SPREAD),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for tag, c in cases.items():
        print(tag, c["tu_inlet_percent"], c["re_theta_onset"],
              c["u_rms_peak_onset"])
    print({k: v for k, v in result.items() if k != "cases"})


if __name__ == "__main__":
    main()
