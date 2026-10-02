#!/usr/bin/env python
"""Calibrate the leading-edge variant's streak constant on the JHTDB plate
by running the model itself.

The variant is Langtry-Menter with the streak onset correlation and no
diffusion of ReThetat (sim/run_lm_leading_edge.py). As for
results/lm-streak-calibration.json, C is the grid value with the lowest
mean C_f error over the plate's sampled stations. The grid ends at the
value set by applying the rule to the DNS at the inlet intensity
(results/onset-history.json). A first scan, from that value upward with
ReThetat at the inlet left at Langtry-Menter's correlation, moved the
plate's C_f error by under half a percent from end to end: without
diffusion the layer kept the inlet's value. With the runs started from
C / Tu_in, a trial at the DNS-level value put the plate's onset late, so
the grid runs down from there.

Outputs
-------
results/lm-le-calibration.json
"""

from __future__ import annotations

import importlib.util
import json

GRID = (450, 530, 610, 690, 769)
OUT = "results/lm-le-calibration.json"


def main():
    spec = importlib.util.spec_from_file_location(
        "t", "scripts/test-lm-streak.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scan = []
    for c in GRID:
        err, _ = mod.plate_cf_error(f"sim/cases/lm-le-plate-C{c}")
        scan.append({"c_streak": c, "plate_cf_err": err})
        print(c, round(err, 4))
    lm, _ = mod.plate_cf_error(mod.PLATE_LM)
    best = min(scan, key=lambda r: r["plate_cf_err"])
    with open("results/onset-history.json") as f:
        c_dns = json.load(f)["c_inlet"]
    out = {
        "grid": list(GRID),
        "scan": scan,
        "c_streak": best["c_streak"],
        "plate_cf_err": best["plate_cf_err"],
        "plate_cf_err_lm": lm,
        "c_dns_level": c_dns,
        "at_grid_edge": best["c_streak"] in (GRID[0], GRID[-1]),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(out)


if __name__ == "__main__":
    main()
