#!/usr/bin/env python
"""Do the models transition the JHTDB plate as they did from x = 30 once
the plate's own leading edge is in the domain?

Every plate run so far starts at x = 30, from the DNS's profile there. The
nosed runs (scripts/run-nosed-plate.py) start upstream of the super-
elliptic nose, under a free stream fitted to each model's own decay of the
DNS's, so the layer starts at the stagnation point as in the DNS.

For the leading-edge clipping closure (clipKGammaLE with wallTu on) this is
the first real leading edge: its threshold reads the intensity carried in
from where the layer started, which here is the stagnation region, about
3 percent in the DNS. Its reference is the plate's leading-edge
intensity (results/plate-leading-edge-intensity.json); until 2026-10-06 it
was the inlet value at x = 30, 2.54 percent, and the test failed by 4.8
times with the closure carrying 3.25 percent from the nose against it.

Test, fixed before any full nosed run: the leading-edge closure's plate
C_f error on the nosed plate is within NOSE_TOL of its error from the
x = 30 inlet (results/clip-wall.json's plate run). Also reported: every
model's error and onset ratio, nosed against inlet-started; each nosed
run's inlet intensity and decay fit; and the intensity the leading-edge
closure carries next to the wall along the plate.

Outputs
-------
results/nosed-plate.json
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

OUT = "results/nosed-plate.json"
NOSE_TOL = 0.25
#: Each nosed run and the same model started at x = 30
PAIRS = {
    "sst": "k-omega-sst-dns-domain",
    "lm": "k-omega-sst-lm-dns-domain",
    "clip": "clip-k-gamma-dns-domain",
    "clip-le": "clip-wall-plate",
}


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def onset_ratio(gate, case_dir):
    gate.PLATE_CASE = case_dir
    try:
        return float(gate.plate()["ratio"])
    except TypeError:
        # No onset by the C_f measure inside the plate
        return None


def main():
    streak = load("streak", "scripts/test-lm-streak.py")
    gate = load("gate", "scripts/test-lm-onset-transfer.py")
    rows = {}
    for key, inlet in PAIRS.items():
        nosed = f"sim/cases/nosed-plate-{key}"
        e_n, st_n = streak.plate_cf_error(nosed)
        e_i, _ = streak.plate_cf_error(f"sim/cases/{inlet}")
        with open(os.path.join(nosed, "postProcessing", "run.json")) as f:
            meta = json.load(f)
        row = {
            "inlet_case": inlet,
            "cf_err_nosed": e_n,
            "cf_err_inlet": e_i,
            "onset_ratio_nosed": onset_ratio(gate, nosed),
            "onset_ratio_inlet": onset_ratio(gate, f"sim/cases/{inlet}"),
            "tu_inlet_percent": meta["tu_inlet_percent"],
            "decay_fit_log_rms": meta["decay_fit_log_rms"],
            "stations_nosed": st_n,
        }
        if "tule_wall" in meta:
            x, tu = np.array(meta["tule_wall_x"]), np.array(meta["tule_wall"])
            row["tule_wall_x"] = x.tolist()
            row["tule_wall"] = tu.tolist()
            row["tule_wall_at_30"] = float(np.interp(30.0, x, tu))
            row["tule_wall_at_300"] = float(np.interp(300.0, x, tu))
        rows[key] = row
        print(key, {k: v for k, v in row.items() if not isinstance(v, list)})
    le = rows["clip-le"]
    with open("results/plate-leading-edge-intensity.json") as f:
        tu_ref = json.load(f)["tu_le_percent"]
    result = {
        "nose_tol": NOSE_TOL,
        "tu_ref": tu_ref,
        "models": rows,
        "le_err_ratio": le["cf_err_nosed"] / le["cf_err_inlet"],
        "passes": bool(
            le["cf_err_nosed"] <= (1 + NOSE_TOL) * le["cf_err_inlet"]
        ),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items() if k != "models"})


if __name__ == "__main__":
    main()
