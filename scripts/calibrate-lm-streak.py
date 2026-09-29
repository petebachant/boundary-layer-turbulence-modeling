#!/usr/bin/env python
"""Calibrate the streak onset correlation's constant on the JHTDB plate by
running the model itself.

The constant set by applying the rule to the DNS (results/onset-laws.json)
does not carry into the model: Langtry-Menter's own dynamics already put its
onset late on the plate, so the correlation's constant makes it later
still. So choose C by running kOmegaSSTLMStreak on the plate over a grid,
which includes the DNS-level value, and keeping the one with the lowest mean
C_f error over the sampled stations (scripts/test-lm-streak.py's measure).

Outputs
-------
results/lm-streak-calibration.json
"""

from __future__ import annotations

import importlib.util
import json

GRID = (420, 470, 530, 628)
OUT = "results/lm-streak-calibration.json"


def main():
    spec = importlib.util.spec_from_file_location(
        "t", "scripts/test-lm-streak.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scan = []
    for c in GRID:
        err, _ = mod.plate_cf_error(f"sim/cases/lm-streak-plate-C{c}")
        scan.append({"c_streak": c, "plate_cf_err": err})
        print(c, round(err, 4))
    lm, _ = mod.plate_cf_error(mod.PLATE_LM)
    best = min(scan, key=lambda r: r["plate_cf_err"])
    with open("results/onset-laws.json") as f:
        c_dns = json.load(f)["c_local"]
    out = {"grid": list(GRID), "scan": scan, "c_streak": best["c_streak"],
           "plate_cf_err": best["plate_cf_err"], "plate_cf_err_lm": lm,
           "c_dns_level": c_dns,
           "plate_cf_err_at_628": next(r["plate_cf_err"] for r in scan
                                       if r["c_streak"] == 628)}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(out)


if __name__ == "__main__":
    main()
