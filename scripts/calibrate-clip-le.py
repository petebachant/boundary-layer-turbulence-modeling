#!/usr/bin/env python
"""Calibrate the leading-edge clipping closure's reference intensity on the
JHTDB plate by running the model itself.

clipKGammaLE (sim/clipLE, sim/run_clip_le.py) is the clipping closure with
its transition threshold scaled as TuRef/TuLE, TuLE the free-stream
intensity carried into the layer from where its fluid entered it. A trial
run at TuRef equal to the plate's inlet intensity transitioned late: the
plate's free stream decays as its layer grows, so the layer's outer part,
where the threshold acts, holds a TuLE well below the inlet's. TuRef is
therefore the grid value with the lowest mean plate C_f error, as for the
Langtry-Menter variants, every other coefficient as calibrated. Each run's
largest C_f over the DNS's, and the smallest after it, are recorded to show
the shape of its transition.

Outputs
-------
results/clip-le-calibration.json
"""

from __future__ import annotations

import importlib.util
import json

GRID = (1.4, 1.6, 1.8, 2.0, 2.2)
OUT = "results/clip-le-calibration.json"
PLATE_CLIP = "sim/cases/clip-k-gamma-dns-domain"


def main():
    spec = importlib.util.spec_from_file_location(
        "t", "scripts/test-lm-streak.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scan = []
    for tu in GRID:
        err, rows = mod.plate_cf_error(f"sim/cases/clip-le-plate-T{tu}")
        r = [row["cf"] / row["cf_dns"] for row in rows]
        i = max(range(len(r)), key=r.__getitem__)
        scan.append(
            {
                "tu_ref": tu,
                "plate_cf_err": err,
                "cf_ratio_max": r[i],
                "x_cf_ratio_max": rows[i]["x"],
                "cf_ratio_min_after": min(r[i:]),
            }
        )
        print(tu, round(err, 4))
    clip, _ = mod.plate_cf_error(PLATE_CLIP)
    with open("results/inlet-profiles.json") as f:
        tu_in = json.load(f)["Tu_inlet_percent"]
    best = min(scan, key=lambda r: r["plate_cf_err"])
    out = {
        "grid": list(GRID),
        "scan": scan,
        "tu_ref": best["tu_ref"],
        "plate_cf_err": best["plate_cf_err"],
        "plate_cf_err_clip": clip,
        "error_over_clip": best["plate_cf_err"] / clip,
        "best": best,
        "plate_tu_inlet": tu_in,
        "at_grid_edge": best["tu_ref"] in (GRID[0], GRID[-1]),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(out)


if __name__ == "__main__":
    main()
