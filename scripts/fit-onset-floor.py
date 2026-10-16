#!/usr/bin/env python
"""Fit an onset floor to the DNS, for the leading-edge clipping closure.

The closure's threshold scales as 1/Tu_LE, so its onset falls where
S0 = Tu_LE^2 Re_x reaches a fixed value, about 0.95 million (percent
squared) on every run (results/clip-le-onset-bias.json), and its onset
error on a flow is that value over the flow's own S0. The DNS's S0 is not
constant: it rises at high intensity, to 1.5 million on Wu et al.'s 6
percent flow against 0.7 to 1.0 up to 3 percent
(results/onset-dose.json). A layer that breaks down at small Re_x is
thin and slow against the free stream's forcing, and there is a lowest
Reynolds number at which it can transition however strong the forcing.
Abu-Ghannam and Shaw's correlation has one, onset Re_theta tending to 163
at high intensity (AbuGhannamShaw1980). Adding that floor to the
streak rule gives

    Re_x,onset = S0c / Tu_LE^2 + Re_x0,

two constants where the rule had one. Both are fitted here, by least
squares in ln Re_x, to the onsets of the six zero-pressure-gradient DNS
(Wu et al.'s five flows and the JHTDB plate, onset and Tu_LE as in
scripts/analyze-onset-dose.py). The closure keeps its threshold at the
plate's leading-edge intensity, TuRef, where it was calibrated; with the
floor its threshold is

    LambdaC sqrt(((TuRef/Tu_LE)^2 + c) / (1 + c)),   c = Re_x0 TuRef^2 / S0c,

(cFloor in sim/clip-le-floor.patch), since Re_v in a Blasius layer grows
as sqrt(Re_x), so only c is carried to the closure, not S0c.

The form was chosen after the DNS's and the T3 plates' S0 had been looked
at (results/onset-inflow.json, results/clip-le-onset-bias.json); the fit
uses the DNS alone. Reported with it: the fit's log-rms against the
constant-S0 and power-law fits to the same onsets, c with each DNS flow
left out, and the onset ratios the closure would reach if its onsets
moved by the floor's factor alone, (1 + c (Tu/TuRef)^2) / (1 + c) in Re_x,
on the runs of results/clip-le-onset-bias.json, with Tu the closure's
wall intensity at its threshold crossing on the T3 and nosed plates and
the DNS's first-station intensity on Wu et al.'s flows.

Outputs
-------
results/onset-floor.json
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import least_squares

DOSE = "results/onset-dose.json"
BIAS = "results/clip-le-onset-bias.json"
PLATE_LE = "results/plate-leading-edge-intensity.json"
OUT = "results/onset-floor.json"


def fit(tu, rex):
    """S0c and Re_x0 for ln Re_x = ln(S0c / Tu^2 + Re_x0)."""
    r = least_squares(
        lambda p: np.log(np.exp(p[0]) / tu**2 + np.exp(p[1])) - np.log(rex),
        [np.log(np.median(tu**2 * rex)), np.log(1e4)],
    )
    s0c, rex0 = np.exp(r.x)
    return float(s0c), float(rex0), float(np.sqrt(np.mean(r.fun**2)))


def main():
    with open(DOSE) as f:
        flows = json.load(f)["flows"]
    with open(BIAS) as f:
        bias = json.load(f)
    with open(PLATE_LE) as f:
        tu_ref = float(json.load(f)["tu_le_percent"])
    names = [n for n, r in flows.items() if r["kind"] == "dns"]
    tu = np.array([flows[n]["tu_le_percent"] for n in names])
    rex = np.array([flows[n]["re_x_onset"] for n in names])

    s0c, rex0, rms = fit(tu, rex)
    c = rex0 * tu_ref**2 / s0c
    ln_s0 = np.log(tu**2 * rex)
    rms_const = float(np.sqrt(np.mean((ln_s0 - ln_s0.mean()) ** 2)))
    a = np.c_[np.ones_like(tu), np.log(tu)]
    coef, *_ = np.linalg.lstsq(a, np.log(rex), rcond=None)
    rms_power = float(np.sqrt(np.mean((a @ coef - np.log(rex)) ** 2)))
    loo = {}
    for i, n in enumerate(names):
        m = np.arange(len(names)) != i
        s, r0, _ = fit(tu[m], rex[m])
        loo[n] = {"s0c": s, "re_x0": r0, "c_floor": r0 * tu_ref**2 / s}

    def factor(t):
        return float((1 + c * (t / tu_ref) ** 2) / (1 + c))

    predicted = {}
    for name, r in bias["flows"].items():
        f = factor(r["tu_le_wall_at_crossing"])
        predicted[name] = {
            "tu_percent": r["tu_le_wall_at_crossing"],
            "factor": f,
            "ratio_cf_min": r["ratio_cf_min"],
            "predicted_ratio_cf_min": r["ratio_cf_min"] * f,
        }
    for tag, r in bias["wu"].items():
        f = factor(r["ref_tu_le"])
        predicted[tag] = {
            "tu_percent": r["ref_tu_le"],
            "factor": f,
            "ratio_cf_min": r["ratio_cf_min"],
            "predicted_ratio_cf_min": r["ratio_cf_min"] * f,
        }
    result = {
        "flows": names,
        "tu_le_percent": tu.tolist(),
        "re_x_onset": rex.tolist(),
        "tu_ref_percent": tu_ref,
        "s0c": s0c,
        "re_x0": rex0,
        "re_theta_0": float(0.664 * np.sqrt(rex0)),
        "c_floor": c,
        "log_rms": rms,
        "log_rms_constant_s0": rms_const,
        "log_rms_power_law": rms_power,
        "power_law_exponent": float(coef[1]),
        "leave_one_out": loo,
        "c_floor_loo_range": [
            min(v["c_floor"] for v in loo.values()),
            max(v["c_floor"] for v in loo.values()),
        ],
        "predicted": predicted,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "predicted"}))
    for k, v in predicted.items():
        print(k, v)


if __name__ == "__main__":
    main()
