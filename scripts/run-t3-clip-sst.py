#!/usr/bin/env python
"""Run SST with the leading-edge clipping closure's onset mechanism
(clipKOmegaSSTLE, sim/clipSST) on one of the ERCOFTAC T3 plates, set up as
scripts/run-t3.py sets SST up, free stream fitted to SST's decay of the
LES's, with gamma and TuLE as the leading-edge closure carries them
there, except that gamma has zero gradient on the plate
(docs/ideas-log.md, section 7.15).

Kept apart from scripts/run-t3.py so that adding it reruns none of that
script's models.

Usage: python scripts/run-t3-clip-sst.py <case>

Outputs
-------
sim/cases/t3-<case>-clip-sst/postProcessing/t3.json, as scripts/run-t3.py
writes it
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys

import numpy as np

from pypkg import foam_body, sst_le_closures
from pypkg.cases import openfoam as of

KEY = "clip-sst"
MODEL = sst_le_closures.MODEL
#: SST's free-stream destruction coefficient, as run-t3.py gives SST
BETA = 0.0828


def load_t3():
    spec = importlib.util.spec_from_file_location("run_t3", "scripts/run-t3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(t3, case, setup):
    k0, w0, fit_err = t3.fit_freestream(setup["les"], BETA)
    tu_in = 100 * np.sqrt(2 * k0 / 3)
    case_dir = os.path.join("sim", "cases", f"t3-{case}-{KEY}")
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    shutil.copytree(os.path.join("sim", "t3", case, "template"), case_dir)
    of.write_turbulence_properties(case_dir, MODEL)
    of.write_model_coeffs(case_dir, MODEL, ".")
    of.ensure_libs(case_dir, MODEL)
    names = ["U", "p", "k", "omega", "nut", "gamma"]
    t3.write_fields(case_dir, names, {"k": k0, "omega": w0})
    path = os.path.join(case_dir, "0", "gamma")
    with open(path) as f:
        text = f.read()
    with open(path, "w") as f:
        f.write(t3.set_patch(text, "plate", "        type zeroGradient;\n"))
    t3.add_tule(case_dir, tu_in)
    print(
        f"{case} {KEY}: k0 {k0:.4g}, omega0 {w0:.4g}, Tu at inlet "
        f"{tu_in:.2f} percent, decay fit log-rms {fit_err:.3f}",
        flush=True,
    )
    n_plate = setup["mesh"]["n_plate"]
    sol = foam_body.solve(t3.Stations(setup, n_plate), case_dir, fast=False)
    lines = t3.plate_lines(case_dir, n_plate)
    t = of.latest_time_dir(case_dir)
    tu = of.read_field(os.path.join(case_dir, t, "TuLE"))
    n_up = t3.N_UP * t3.N_NORMAL
    out = {
        "model": MODEL,
        "iterations": sol["iterations"],
        "converged": sol["converged"],
        "settled": sol["settled"],
        "beta": BETA,
        "k_inlet": k0,
        "omega_inlet": w0,
        "tu_inlet_percent": tu_in,
        "decay_fit_log_rms": fit_err,
        "p0": lines["p0"],
        "plate": {n: t3.thin(lines[n]) for n in ("x", "cf", "ue", "tau")},
        "free_stream": {
            "x": t3.thin(lines["fs_x"], 300),
            "y": t3.thin(lines["fs_y"], 300),
            "k": t3.thin(lines["fs_k"], 300),
            "U": t3.thin(lines["fs_U"], 300),
        },
        "tule_wall": t3.thin(tu[n_up : n_up + n_plate]),
    }
    os.makedirs(os.path.join(case_dir, "postProcessing"), exist_ok=True)
    with open(os.path.join(case_dir, "postProcessing", "t3.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")


def main():
    case = sys.argv[1]
    t3 = load_t3()
    with open(t3.SETUP) as f:
        setup = json.load(f)["cases"][case]
    run(t3, case, setup)
    print(f"{case} {KEY} done", flush=True)


if __name__ == "__main__":
    main()
