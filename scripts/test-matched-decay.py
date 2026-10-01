#!/usr/bin/env python
"""Is the free stream's decay what separates Langtry-Menter's onset on the
JHTDB plate from its onset on Wu et al.'s flows?

Neither the plate's streaky inlet (results/plate-quiet-inlet-test.json)
nor the free-stream length scale (results/length-scale-onset.json)
explains why the plate and Wu et al.'s flows pull calibrations apart. The
remaining difference is how well each run's free stream follows the
measured one up to onset: the baseline inlets fit the decay over a stretch
running well past onset. Here standard Langtry-Menter is rerun on every
flow with its inlet omega refitted to the decay from inlet to onset
(results/matched-decay.json), constants as published, and its onset ratio
measured exactly as in results/lm-onset-transfer.json.

Test, fixed before the runs: with every free stream matched up to onset,
the plate's onset ratio lies within the band of the ratios on Wu et al.'s
1.5 to 3 percent flows, widened by MARGIN on either side. Also reported:
each run's free-stream intensity at the DNS onset against the DNS's, to
check the match held in the solution, and the Wu scores against the
baseline runs'.

Outputs
-------
results/matched-decay-test.json
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import re

import numpy as np
import pandas as pd

from pypkg import registry

OUT = "results/matched-decay-test.json"
BAND = ("wu-bypass-tu150", "wu-bypass-tu225", "wu-bypass-tu300")
MARGIN = 0.05
PLATE_Y_FS = 20.0
WU_Y_FS = 550.0


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def latest(case_dir):
    root = os.path.join(case_dir, "postProcessing", "sample")
    return sorted(glob.glob(os.path.join(root, "*")),
                  key=lambda p: float(os.path.basename(p)))[-1]


def tu_at(case_dir, x, y_fs):
    """Model free-stream intensity at the sampled station nearest x."""
    files = glob.glob(os.path.join(latest(case_dir), "x*.csv"))
    f = min(files, key=lambda p: abs(
        float(re.search(r"x(\d+)", os.path.basename(p)).group(1)) - x))
    df = pd.read_csv(f)
    i = int(np.argmin(np.abs(df["y"] - y_fs)))
    k = df["k"].iloc[i]
    return float(100.0 * np.sqrt(2.0 * k / 3.0) / df["U_0"].iloc[i])


def main():
    gate = load("gate", "scripts/test-lm-onset-transfer.py")
    score = load("score", "scripts/score-wu-openfoam.py")
    from pypkg.dns_case import load_dns
    d = load_dns()
    gate.PLATE_CASE = "sim/cases/decay-plate"
    plate = gate.plate()
    with open("results/transition-mechanics.json") as f:
        x_on = json.load(f)["x_transition_onset"]
    j = int(np.argmin(np.abs(d["y"] - PLATE_Y_FS)))
    i = int(np.argmin(np.abs(d["x"] - x_on)))
    tu_dns_plate = float(100.0 * np.sqrt(2.0 * d["k"][j, i] / 3.0)
                         / d["U"][j, i])
    plate["tu_onset_model"] = tu_at("sim/cases/decay-plate", x_on,
                                    PLATE_Y_FS)
    plate["tu_onset_baseline"] = tu_at("sim/cases/k-omega-sst-lm-dns-domain",
                                       x_on, PLATE_Y_FS)
    plate["tu_onset_dns"] = tu_dns_plate
    with open("results/wu-openfoam.json") as f:
        base = json.load(f)
    with open("results/bypass-onset.json") as f:
        onset = json.load(f)["cases"]
    cases = registry.cases()
    wu = {}
    for tag in score.TAGS:
        name = "wu-bypass-tu" + tag[2:]
        case = cases[name].build()
        cdir = f"sim/cases/decay-{tag}"
        st = score.read_stations(cdir)
        U, inside = score.to_case_grid(case, st)
        first = int(np.argmax(inside))
        last = len(inside) - 1 - int(np.argmax(inside[::-1]))
        U[:, :first] = U[:, [first]]
        U[:, last + 1:] = U[:, [last]]
        sc = case.score({"U": U})
        cf, th, _ = case._metrics(U)
        _, th_m = gate.onset(case.x, cf, th)
        _, th_d = gate.onset(case.x, np.interp(case.x, case.x_cf, case.cf_ref),
                             np.interp(case.x, case.x_th, case.theta_ref))
        x_on_w = min(onset[tag]["re_x_onset"] / case.re_theta0, case.x[-1])
        urms = np.loadtxt(f"data/wu-bypass-transition/stats_{tag}/"
                          f"Re_x_versus_urms_at_y_500_to_800_theta_0_{tag}.dat")
        wu[name] = {
            "ratio": (th_m / th_d if th_m and th_d else None),
            "normalized": sc["normalized"],
            "baseline_normalized": base["cases"][name]["k-omega-sst-lm"]["normalized"],
            "tu_onset_model": tu_at(cdir, x_on_w, WU_Y_FS),
            "tu_onset_baseline": tu_at(f"sim/cases/wu-{tag}-k-omega-sst-lm",
                                       x_on_w, WU_Y_FS),
            "tu_onset_dns": float(100.0 * np.interp(x_on_w * case.re_theta0,
                                                    urms[:, 0], urms[:, 1]))}
        print(name, wu[name])
    ratios = [wu[n]["ratio"] for n in BAND if wu[n]["ratio"] is not None]
    lo, hi = min(ratios), max(ratios)
    s = np.array([w["normalized"] for w in wu.values()])
    b = np.array([w["baseline_normalized"] for w in wu.values()])
    result = {
        "margin": MARGIN, "plate": plate, "wu": wu,
        "band_min": lo, "band_max": hi, "plate_ratio": plate["ratio"],
        "wu_mean": float(s.mean()), "wu_mean_baseline": float(b.mean()),
        "in_band": bool(lo * (1 - MARGIN) <= plate["ratio"]
                        <= hi * (1 + MARGIN)),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items() if k not in ("wu",)})


if __name__ == "__main__":
    main()
