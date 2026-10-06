#!/usr/bin/env python
"""Does a transitional layer fill with fluctuation exergy to a common
capacity, with onset before it is full?

results/fluctuation-exergy.json measured the layer's fluctuation exergy,
the integral over the layer of X = k - 3/2 (det R)^(1/3) over U_e^2 theta,
along the JHTDB plate and Wu et al.'s five flows. Its pre-registered test,
that onset is where the exergy first reaches the plate's value at onset,
failed. But in every flow the exergy rose, peaked and fell, with the peak
near one value across a fourfold range of intensity and onset shortly
before it. That reading came after the data, so it is tested here on data
it has not seen: Bienner et al.'s two incompressible LES runs (Tu = 4
percent in air at Mach 0.1, small and large free-stream length scales),
the pair the onset rules were checked on.

Tests, fixed before Bienner et al.'s exergy was computed:

1. Each run's peak layer exergy is within PEAK_TOL of the reference
   capacity, the geometric mean of the peaks of the plate and of every Wu
   flow whose peak lies inside its stations (not at the first or last).
2. Each run's LES onset (results/length-scale-onset.json) is at or before
   its exergy peak in Re_theta.

Also reported: the ratio of onset to peak Re_theta in every flow, and the
peak exergy and its location in each.

Bienner et al.'s profiles are in wall units against y / delta_99, with
U_tau, U_inf and theta / delta_99 in each file's header; the layer is
integrated to delta_99.

Outputs
-------
results/exergy-saturation.json
"""

from __future__ import annotations

import glob
import json
import re

import numpy as np

OUT = "results/exergy-saturation.json"
EXERGY = "results/fluctuation-exergy.json"
ONSET_LES = "results/length-scale-onset.json"
ONSET_WU = "results/bypass-onset.json"
LES = "data/bienner-bypass-les/profiles"
RUNS = {"small": "low_Lf_Tu4_Air_M0p1", "large": "high_Lf_Tu4_Air_M0p1"}
PEAK_TOL = 0.25


def header(text, key):
    return float(re.search(rf"{re.escape(key)}\s*=\s*(\S+)", text).group(1))


def les_station(path):
    """Re_theta and the layer's exergy over U_e^2 theta at one station."""
    with open(path) as f:
        text = f.read()
    rth = float(re.search(r"Rtheta_(\d+)", path).group(1))
    a = np.array(
        [
            line.split()
            for line in text.splitlines()
            if line.strip()
            and re.match(r"\s*-?\d", line)
            and len(line.split()) >= 8
        ],
        dtype=float,
    )
    eta, urms, vrms, wrms, uv = a[:, 0], a[:, 4], a[:, 5], a[:, 6], a[:, 7]
    m = eta <= 1.0
    uu, vv, ww = urms[m] ** 2, vrms[m] ** 2, wrms[m] ** 2
    k = 0.5 * (uu + vv + ww)
    det = np.maximum(ww * (uu * vv - uv[m] ** 2), 0.0)
    X = k - 1.5 * np.cbrt(det)
    ratio = (header(text, "U_tau [m/s]") / header(text, "U_inf [m/s]")) ** 2
    th = header(text, "theta     / d99")
    return {
        "re_theta": rth,
        "exergy": float(np.trapezoid(X, eta[m]) * ratio / th),
        "fraction": float(np.trapezoid(X, eta[m]) / np.trapezoid(k, eta[m])),
    }


def peak(rows):
    """Peak exergy, its Re_theta, and whether it lies inside the stations."""
    e = np.array([r["exergy"] for r in rows])
    i = int(np.argmax(e))
    return float(e[i]), float(rows[i]["re_theta"]), 0 < i < len(rows) - 1


def main():
    with open(EXERGY) as f:
        ex = json.load(f)
    with open(ONSET_LES) as f:
        les_onset = json.load(f)["runs"]
    with open(ONSET_WU) as f:
        wu_onset = json.load(f)["cases"]
    flows = {}
    plate = ex["plate"]["stations"]
    pk, at, inside = peak(plate)
    x = np.array([s["x"] for s in plate])
    r = np.array([s["re_theta"] for s in plate])
    flows["plate"] = {
        "peak": pk,
        "peak_re_theta": at,
        "peak_inside": inside,
        "onset_re_theta": float(np.interp(ex["plate"]["x_onset"], x, r)),
    }
    for tag, rows in ex["wu_stations"].items():
        pk, at, inside = peak(rows)
        flows[tag] = {
            "peak": pk,
            "peak_re_theta": at,
            "peak_inside": inside,
            "onset_re_theta": wu_onset[tag]["re_theta_onset"],
        }
    used = [t for t, f in flows.items() if f["peak_inside"]]
    capacity = float(np.exp(np.mean([np.log(flows[t]["peak"]) for t in used])))
    les = {}
    for name, run in RUNS.items():
        rows = sorted(
            (les_station(p) for p in glob.glob(f"{LES}/{run}/profile_*.dat")),
            key=lambda s: s["re_theta"],
        )
        pk, at, inside = peak(rows)
        on = les_onset[name]["re_theta_onset"]
        les[name] = {
            "run": run,
            "peak": pk,
            "peak_re_theta": at,
            "peak_inside": inside,
            "onset_re_theta": on,
            "peak_over_capacity": pk / capacity,
            "stations": rows,
        }
    for f in list(flows.values()) + list(les.values()):
        f["onset_over_peak"] = (
            f["onset_re_theta"] / f["peak_re_theta"]
            if f["onset_re_theta"]
            else None
        )
    capacity_ok = all(
        abs(f["peak"] / capacity - 1.0) <= PEAK_TOL for f in les.values()
    )
    before_ok = all(
        f["onset_re_theta"] <= f["peak_re_theta"] for f in les.values()
    )
    result = {
        "peak_tol": PEAK_TOL,
        "capacity": capacity,
        "capacity_from": used,
        "flows": flows,
        "les": les,
        "capacity_passes": bool(capacity_ok),
        "onset_before_peak_passes": bool(before_ok),
        "passes": bool(capacity_ok and before_ok),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items() if k not in ("flows", "les")})
    for name, f in {**flows, **les}.items():
        print(name, {k: v for k, v in f.items() if k != "stations"})


if __name__ == "__main__":
    main()
