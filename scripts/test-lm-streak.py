#!/usr/bin/env python
"""Does Langtry-Menter place transition better with the streak onset
correlation in place of its own?

The streak rule in local form, Re_theta,onset = C / Tu with C from the
JHTDB plate, predicts Wu et al.'s onsets better than Langtry-Menter's
correlation when both are applied to the DNS (results/onset-laws.json).
kOmegaSSTLMStreak is Langtry-Menter with that one correlation replaced
(sim/lmStreak), with its constant calibrated by running the model on the
plate (results/lm-streak-calibration.json), then run on Wu et al.'s five
flows on the same meshes, inlets and free streams as the standard model.

A first test used the constant set by applying the rule to the DNS, and
required the plate's C_f error to stay within 10 percent of standard
Langtry-Menter's; it failed on the plate alone (the calibration scan
records it), so its Wu runs were not made.

Test, fixed before any of Wu et al.'s flows was run with this model: with
C calibrated on the plate, it beats standard Langtry-Menter's mean
normalized score on Wu et al.'s flows (results/wu-openfoam.json). The
plate's C_f error at that C is reported beside standard Langtry-Menter's.

Outputs
-------
results/lm-streak.json
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
from pypkg.dns_case import load_dns

OUT = "results/lm-streak.json"
WU = "results/wu-openfoam.json"
PLATE_LM = "sim/cases/k-omega-sst-lm-dns-domain"
CAL = "results/lm-streak-calibration.json"


def plate_stations(case_dir):
    root = os.path.join(case_dir, "postProcessing", "sample")
    t = sorted(glob.glob(os.path.join(root, "*")),
               key=lambda p: float(os.path.basename(p)))[-1]
    return {float(re.search(r"x(\d+)", os.path.basename(p)).group(1)):
            pd.read_csv(p) for p in glob.glob(os.path.join(t, "x*.csv"))}


def plate_cf_error(case_dir):
    """Mean |C_f / C_f,DNS - 1| over the sampled stations, with C_f from the
    first cell above the wall in each, as scripts/compare-openfoam-dns.py
    computes it."""
    d = load_dns()
    nu = d["nu"]
    errs, rows = [], []
    for x, df in sorted(plate_stations(case_dir).items()):
        y, u = df["y"].to_numpy(), df["U_0"].to_numpy()
        cf = 2.0 * nu * (u[1] - u[0]) / (y[1] - y[0])
        i = int(np.argmin(np.abs(d["x"] - x)))
        cf_dns = 2.0 * nu * d["U"][0, i] / d["y"][0]
        errs.append(abs(cf / cf_dns - 1.0))
        rows.append({"x": x, "cf": float(cf), "cf_dns": float(cf_dns)})
    return float(np.mean(errs)), rows


def main():
    spec = importlib.util.spec_from_file_location(
        "score_wu", "scripts/score-wu-openfoam.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with open(WU) as f:
        wu = json.load(f)
    cases = registry.cases()
    rows = {}
    for tag in mod.TAGS:
        name = "wu-bypass-tu" + tag[2:]
        case = cases[name].build()
        st = mod.read_stations(os.path.join("sim", "cases",
                                            f"lm-streak-{tag}"))
        U, inside = mod.to_case_grid(case, st)
        first = int(np.argmax(inside))
        last = len(inside) - 1 - int(np.argmax(inside[::-1]))
        U[:, :first] = U[:, [first]]
        U[:, last + 1:] = U[:, [last]]
        sc = case.score({"U": U})
        _, th, H = case._metrics(U)
        xo, i = mod.onset(case.x, H)
        rows[name] = {
            "streak": sc, "streak_normalized": sc["normalized"],
            "lm_normalized": wu["cases"][name]["k-omega-sst-lm"]["normalized"],
            "streak_re_theta_onset": (float(th[i] * case.re_theta0)
                                      if i is not None else None),
            "lm_re_theta_onset":
                wu["cases"][name]["k-omega-sst-lm"].get("re_theta_onset"),
            "dns_re_theta_onset": wu["dns_onset"][name]["re_theta_onset"]}
        print(name, round(sc["normalized"], 3),
              round(rows[name]["lm_normalized"], 3))
    s = np.array([r["streak_normalized"] for r in rows.values()])
    lm = np.array([r["lm_normalized"] for r in rows.values()])
    with open(CAL) as f:
        c = json.load(f)["c_streak"]
    cf_lm, st_lm = plate_cf_error(PLATE_LM)
    cf_s, st_s = plate_cf_error(f"sim/cases/lm-streak-plate-C{c}")
    result = {
        "c_streak": c, "cases": rows,
        "mean_streak": float(s.mean()), "mean_lm": float(lm.mean()),
        "n_wins": int(np.sum(s < lm)), "n_cases": len(rows),
        "plate_cf_err_streak": cf_s, "plate_cf_err_lm": cf_lm,
        "plate_stations_streak": st_s, "plate_stations_lm": st_lm,
        "passes": bool(s.mean() < lm.mean()),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items()
           if k not in ("cases", "plate_stations_streak",
                        "plate_stations_lm")})


if __name__ == "__main__":
    main()
