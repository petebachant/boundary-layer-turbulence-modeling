#!/usr/bin/env python
"""Run one model on the JHTDB plate with its leading edge.

Copies the meshed template (sim/nosed-plate/template), sets the model up
with the OpenFOAM benchmark's own functions (pypkg/cases/openfoam.py, and
pypkg/le_closures.py for the leading-edge clipping closure), and runs it.

The free stream is set at the inlet so that each model's own decay matches
the DNS's: with no production, k and omega along a streamline follow

    omega = omega0 / (1 + beta omega0 s),
    k = k0 (1 + beta omega0 s)^(-betaStar / beta),

with s the distance from the inlet. k0 is the DNS's intensity at its
inflow, 3.5 percent, and omega0 is fitted to the DNS's free-stream k at
FREESTREAM_Y over x from 30 to 300, where onset is decided, with each
model's beta (SST's outer value, the clipping closure's calibrated one).
A first version fitted k0 too, over x from 30 to 600; since a k-omega free
stream cannot follow the DNS's decay exponent, that put SST's and
Langtry-Menter's inlet intensity at 11 percent, and their layers
transitioned far too early.
The activation fraction enters at the plate runs' free-stream value.

Usage: python scripts/run-nosed-plate.py <sst|lm|clip|clip-le>

Outputs
-------
sim/cases/nosed-plate-<model>/postProcessing (the plate stations' samples,
and for clip-le the leading-edge intensity next to the wall)
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import sys

import numpy as np
from scipy.optimize import least_squares

from pypkg import (
    foam_body,
    le_closures,  # noqa: F401  (wraps the case set-up)
)
from pypkg.cases import openfoam as of
from pypkg.dns_case import load_dns

TEMPLATE = "sim/nosed-plate/template"
FIELDS = "sim/fields-low-re"
COEFFS = "results/clip-k-gamma-coeffs.json"
FREESTREAM_Y = 20.0
FIT_X = (30.0, 300.0)
#: The DNS's free-stream intensity at its curved inflow boundary, percent
TU_INFLOW = 3.5
#: x where the streamline at FREESTREAM_Y crosses the inlet, from the
#: inlet's quarter ellipse about (20, -1) with semi-axes 60 and 41
X_INLET_FS = 20.0 - 60.0 * np.sqrt(1 - ((FREESTREAM_Y + 1) / 41) ** 2)
MODELS = {
    "sst": ("kOmegaSST", 0.0828),
    "lm": ("kOmegaSSTLM", 0.0828),
    "clip": ("clipKGamma", None),
    "clip-le": ("clipKGammaLE", None),
}
BETA_STAR = 0.09


class Stations:
    """The plate stations' C_f against the DNS's, the measure
    pypkg/foam_body.solve watches for a run to settle on, when the run does
    not first reach the residual targets the plate runs from x = 30 reach.

    The plate runs' 3,000 iterations are not enough here: the first nosed
    runs ended with the pressure residual near 1e-3, while the runs from
    x = 30 reach the targets in 1,600 to 2,200."""

    def __init__(self):
        spec = importlib.util.spec_from_file_location(
            "streak", "scripts/test-lm-streak.py"
        )
        self.streak = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.streak)
        with open(os.path.join(TEMPLATE, "system", "sample")) as f:
            xs = re.findall(r"start\s*\((\S+) 0 0\)", f.read())
        self.TARGETS = {f"cf_ratio_x{float(x):g}": 0.05 for x in xs}

    def read_solution(self, case_dir):
        with open(os.path.join(case_dir, "log.simpleFoam")) as f:
            converged = "SIMPLE solution converged" in f.read()
        return {
            "case_dir": case_dir,
            "time": of.latest_time_dir(case_dir),
            "converged": converged,
        }

    def errors(self, sol):
        _, rows = self.streak.plate_cf_error(sol["case_dir"])
        return {f"cf_ratio_x{r['x']:g}": r["cf"] / r["cf_dns"] for r in rows}


def fit_freestream(beta):
    d = load_dns()
    j = int(np.argmin(np.abs(d["y"] - FREESTREAM_Y)))
    m = (d["x"] >= FIT_X[0]) & (d["x"] <= FIT_X[1])
    x, k = d["x"][m], d["k"][j, m]
    s = x - X_INLET_FS

    k0 = 1.5 * (TU_INFLOW / 100) ** 2

    def resid(p):
        w0 = np.exp(p[0])
        model = k0 * (1 + beta * w0 * s) ** (-BETA_STAR / beta)
        return np.log(model) - np.log(k)

    r = least_squares(resid, [np.log(0.05)])
    w0 = float(np.exp(r.x[0]))
    err = float(np.sqrt(np.mean(r.fun**2)))
    return float(k0), float(w0), err


def set_patch(text, patch, body):
    """Replace one patch's block in a field file."""
    new, n = re.subn(
        rf"(\n\s*{patch}\s*\n\s*)\{{.*?\n\s*\}}",
        lambda m: m.group(1) + "{\n" + body + "    }",
        text,
        count=1,
        flags=re.S,
    )
    assert n == 1, f"no {patch} block"
    return new


def write_fields(case_dir, names, values):
    zero = os.path.join(case_dir, "0")
    os.makedirs(zero, exist_ok=True)
    for name in names:
        with open(os.path.join(FIELDS, name)) as f:
            text = f.read()
        text = set_patch(text, "lowerWall", "        type symmetryPlane;\n")
        if name in values:
            text, n = re.subn(
                r"internalField\s+uniform\s+[-\d.eE+]+;",
                f"internalField   uniform {values[name]:.8g};",
                text,
            )
            assert n == 1, f"{name} has no uniform internalField"
        with open(os.path.join(zero, name), "w") as f:
            f.write(text)


def lm_re_theta_t(tu):
    return (
        1173.51 - 589.428 * tu + 0.2196 / tu**2
        if tu <= 1.3
        else 331.50 * (tu - 0.5658) ** -0.671
    )


def main():
    key = sys.argv[1]
    model, beta = MODELS[key]
    if beta is None:
        with open(COEFFS) as f:
            beta = float(json.load(f)["openfoam_coeffs"]["beta"])
    k0, w0, fit_err = fit_freestream(beta)
    tu_in = 100 * np.sqrt(2 * k0 / 3)
    case_dir = os.path.join("sim", "cases", f"nosed-plate-{key}")
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    shutil.copytree(TEMPLATE, case_dir)
    of.write_turbulence_properties(case_dir, model)
    of.write_model_coeffs(case_dir, model, ".")
    of.ensure_libs(case_dir, model)
    names = ["U", "p", "k", "omega", "nut"]
    values = {"k": k0, "omega": w0}
    if model in ("clipKGamma", "clipKGammaLE"):
        names.append("gamma")
    if model == "kOmegaSSTLM":
        names += ["gammaInt", "ReThetat"]
        values["ReThetat"] = lm_re_theta_t(tu_in)
    write_fields(case_dir, names, values)
    if model == "clipKGammaLE":
        # The leading-edge intensity starts at the inlet's
        with open(os.path.join(case_dir, "0", "k")) as f:
            text = f.read()
        text = text.replace("object      k;", "object      TuLE;")
        text = re.sub(
            r"dimensions\s+\[[^\]]*\];",
            "dimensions      [0 0 0 0 0 0 0];",
            text,
        )
        text = re.sub(
            r"internalField\s+uniform\s+\S+;",
            f"internalField   uniform {tu_in:.8g};",
            text,
        )
        text = set_patch(text, "plate", "        type zeroGradient;\n")
        with open(os.path.join(case_dir, "0", "TuLE"), "w") as f:
            f.write(text)
        for name, old, new in (
            ("fvSolution", "|ReThetat|", "|ReThetat|TuLE|"),
            # Plain upwind, not the bounded scheme: TuLE has no diffusion,
            # and the bounded form's subtracted flux divergence made its
            # matrix singular under the nose's early continuity errors
            (
                "fvSchemes",
                "div(phi,gamma)  $turbulence;",
                "div(phi,gamma)  $turbulence;\n    div(phi,TuLE)   Gauss upwind;",
            ),
            (
                "fvSchemes",
                "gradSchemes\n{\n",
                "gradSchemes\n{\n    grad(TuLE)      Gauss linear;\n",
            ),
        ):
            path = os.path.join(case_dir, "system", name)
            with open(path) as f:
                text = f.read()
            assert old in text, f"{name} lacks {old!r}"
            with open(path, "w") as f:
                f.write(text.replace(old, new, 1))
    print(
        f"{key}: k0 {k0:.4g}, omega0 {w0:.4g}, Tu at inlet {tu_in:.2f} "
        f"percent, decay fit log-rms {fit_err:.3f}"
    )
    post = "postProcess -latestTime -func sample > log.sample 2>&1"
    if model == "clipKGammaLE":
        post += (
            " && postProcess -func writeCellCentres -latestTime > log.cc 2>&1"
        )
    # At the plate runs' own relaxation throughout: switched to SIMPLEC's
    # after the first stretch, SST's omega went to NaN within 70 iterations
    # at the nose
    sol = foam_body.solve(Stations(), case_dir, post=post, fast=False)
    meta = {
        "model": model,
        "iterations": sol["iterations"],
        "converged": sol["converged"],
        "settled": sol["settled"],
        "beta": beta,
        "k_inlet": k0,
        "omega_inlet": w0,
        "tu_inlet_percent": tu_in,
        "decay_fit_log_rms": fit_err,
    }
    if model == "clipKGammaLE":
        t = of.latest_time_dir(case_dir)
        C = of.read_field(os.path.join(case_dir, t, "C"))
        tu = of.read_field(os.path.join(case_dir, t, "TuLE"))
        # The first cells above the flat plate
        flat = (C[:, 0] > 20) & (C[:, 1] < 0.006)
        order = np.argsort(C[flat, 0])
        meta["tule_wall_x"] = C[flat, 0][order][::20].tolist()
        meta["tule_wall"] = tu[flat][order][::20].tolist()
    with open(os.path.join(case_dir, "postProcessing", "run.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
