#!/usr/bin/env python
"""Run SST, Langtry-Menter and the leading-edge clipping closure on one of
the ERCOFTAC T3 plates, on Zemmour et al.'s LES domain.

Copies the meshed template (sim/t3/<case>/template, from
scripts/make-t3-cases.py), sets each model up with the OpenFOAM benchmark's
own functions (pypkg/cases/openfoam.py, and pypkg/le_closures.py for the
leading-edge closure), and runs the models side by side.

The inlet is uniform at the LES inlet's bulk speed, 1 here. Its k and
omega are set so that each model's own decay matches the LES's free
stream, as on the nosed plate: with no production, along a streamline

    omega = omega0 / (1 + beta omega0 t),
    k = k0 (1 + beta omega0 t)^(-betaStar / beta),

with t the time of flight from the inlet at the LES's edge velocity, the
form a free stream that speeds up and slows down follows. k0 and omega0
are both fitted, to the LES's free-stream k from the leading edge to its
onset, where onset is decided, with k0 no more than the LES's k at its
inlet: the LES's inlet turbulence is synthetic, and its first stretch
decays faster than any k-omega free stream can (in T3C3 from 10 to 3
percent ahead of the plate), so fitting k0 to the inlet's would leave the
models' free stream far from the LES's at the plate. The
models' production in the free stream under the T3C plates' strain is not
fitted out; the runs report their own free stream.

The leading-edge closure runs as it stands, with its threshold's reference
the JHTDB plate's leading-edge intensity, and its intensity referred to the
total-pressure speed, whose datum is the outlet's pressure.

Usage: python scripts/run-t3.py <case> [model ...]

Outputs
-------
sim/cases/t3-<case>-<model>/postProcessing (t3.json: C_f, the edge
velocity and the free stream along the plate, and the inlet's fit)
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.optimize import least_squares

from pypkg import (
    foam_body,
    le_closures,  # noqa: F401  (wraps the case set-up)
)
from pypkg.cases import openfoam as of

SETUP = "results/t3-rans-setup.json"
FIELDS = "sim/fields-low-re"
COEFFS = "results/clip-k-gamma-coeffs.json"
MODELS = {
    "sst": ("kOmegaSST", 0.0828),
    "lm": ("kOmegaSSTLM", 0.0828),
    "clip-le": ("clipKGammaLE", None),
}
BETA_STAR = 0.09
#: Must match scripts/make-t3-cases.py
N_UP, N_NORMAL = 80, 160
#: Stations whose C_f the solver watches to settle, from X_FIRST to the
#: sponge
N_STATIONS, X_FIRST = 12, 10.0
#: Points kept along the plate in the output
N_OUT = 600


def time_of_flight(x, ue):
    """Time from the first point of the line at the edge velocity."""
    dt = np.diff(x) * 0.5 * (1 / ue[1:] + 1 / ue[:-1])
    return np.concatenate([[0.0], np.cumsum(dt)])


def fit_freestream(les, beta):
    x, ue, k = (np.array(les[n]) for n in ("x", "ue", "k"))
    t = time_of_flight(x, ue)
    m = (x >= 0) & (x <= les["x_onset"])

    def resid(p):
        k0, w0 = np.exp(p)
        model = k0 * (1 + beta * w0 * t[m]) ** (-BETA_STAR / beta)
        return np.log(model) - np.log(k[m])

    # k0 no more than the LES's own at its inlet, its first point past the
    # boundary: unbounded, the fit to T3C3's fast decay ran to k-omega's
    # power-law limit, k0 and omega0 without bound
    k_max = float(k[1])
    r = least_squares(
        resid,
        [np.log(min(k[m][0], 0.99 * k_max)), np.log(0.05)],
        bounds=([-np.inf, -np.inf], [np.log(k_max), np.inf]),
    )
    k0, w0 = np.exp(r.x)
    return float(k0), float(w0), float(np.sqrt(np.mean(r.fun**2)))


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
        if name == "U":
            # The LES's top is a slip wall
            text = set_patch(text, "upperWall", "        type slip;\n")
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


def add_tule(case_dir, tu_in):
    """The leading-edge closure's intensity field and its solver entries,
    as on the nosed plate."""
    with open(os.path.join(case_dir, "0", "k")) as f:
        text = f.read()
    text = text.replace("object      k;", "object      TuLE;")
    text = re.sub(
        r"dimensions\s+\[[^\]]*\];", "dimensions      [0 0 0 0 0 0 0];", text
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


def plate_lines(case_dir, n_plate):
    """Along the plate: x, C_f against the model's own edge velocity, the
    edge velocity from the wall pressure and the inlet's total pressure, and
    the free stream's k at mid-height."""
    t = of.latest_time_dir(case_dir)
    path = os.path.join(case_dir, t)
    C = of.read_field(os.path.join(path, "C"))
    p = of.read_field(os.path.join(path, "p"))
    U = of.read_field(os.path.join(path, "U"))
    k = of.read_field(os.path.join(path, "k"))
    xf = foam_body.patch_values(os.path.join(path, "C"), "plate")[:, 0]
    wss = foam_body.patch_values(os.path.join(path, "wallShearStress"), "plate")
    tau = -wss[:, 0]
    # blockMesh numbers the upstream block's cells first, then the plate's,
    # x fastest
    n_up = N_UP * N_NORMAL
    row = np.arange(n_up, n_up + n_plate)
    assert np.allclose(C[row, 0], xf, atol=1e-6 * (1 + np.abs(xf).max()))
    inlet = np.arange(N_NORMAL) * N_UP
    mid = inlet[N_NORMAL // 4 : 3 * N_NORMAL // 4]
    p0 = float(np.mean(p[mid] + 0.5 * np.sum(U[mid] ** 2, axis=1)))
    ue = np.sqrt(np.clip(2 * (p0 - p[row]), 0, None))
    cf = 2 * tau / ue**2
    # The free stream at mid-height, along the whole domain
    j = N_NORMAL // 2
    up_mid = np.arange(N_UP) + j * N_UP
    pl_mid = n_up + np.arange(n_plate) + j * n_plate
    cells = np.concatenate([up_mid, pl_mid])
    return {
        "x": xf,
        "cf": cf,
        "ue": ue,
        "tau": tau,
        "p0": p0,
        "fs_x": C[cells, 0],
        "fs_y": C[cells, 1],
        "fs_k": k[cells],
        "fs_U": np.linalg.norm(U[cells], axis=1),
    }


class Stations:
    """C_f at stations along the plate against the LES's, the measure
    pypkg/foam_body.solve watches for a run to settle on."""

    def __init__(self, setup, n_plate):
        les = setup["les"]
        self.les_x, self.les_cf = np.array(les["plate_x"]), np.array(les["cf"])
        self.xs = np.linspace(X_FIRST, setup["x_sponge"], N_STATIONS)
        self.n_plate = n_plate
        self.TARGETS = {f"cf_ratio_x{x:.0f}": 0.05 for x in self.xs}

    def read_solution(self, case_dir):
        with open(os.path.join(case_dir, "log.simpleFoam")) as f:
            converged = "SIMPLE solution converged" in f.read()
        return {
            "case_dir": case_dir,
            "time": of.latest_time_dir(case_dir),
            "converged": converged,
        }

    def errors(self, sol):
        lines = plate_lines(sol["case_dir"], self.n_plate)
        cf = np.interp(self.xs, lines["x"], lines["cf"])
        ref = np.interp(self.xs, self.les_x, self.les_cf)
        return {
            f"cf_ratio_x{x:.0f}": c / r for x, c, r in zip(self.xs, cf, ref)
        }


def thin(a, n=N_OUT):
    a = np.asarray(a)
    idx = np.unique(np.linspace(0, len(a) - 1, n).round().astype(int))
    return a[idx].tolist()


def run(case, key, setup):
    model, beta = MODELS[key]
    if beta is None:
        with open(COEFFS) as f:
            beta = float(json.load(f)["openfoam_coeffs"]["beta"])
    k0, w0, fit_err = fit_freestream(setup["les"], beta)
    tu_in = 100 * np.sqrt(2 * k0 / 3)
    case_dir = os.path.join("sim", "cases", f"t3-{case}-{key}")
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    shutil.copytree(os.path.join("sim", "t3", case, "template"), case_dir)
    of.write_turbulence_properties(case_dir, model)
    of.write_model_coeffs(case_dir, model, ".")
    of.ensure_libs(case_dir, model)
    names = ["U", "p", "k", "omega", "nut"]
    values = {"k": k0, "omega": w0}
    if model == "clipKGammaLE":
        names.append("gamma")
    if model == "kOmegaSSTLM":
        names += ["gammaInt", "ReThetat"]
        values["ReThetat"] = lm_re_theta_t(tu_in)
    write_fields(case_dir, names, values)
    if model == "clipKGammaLE":
        add_tule(case_dir, tu_in)
    print(
        f"{case} {key}: k0 {k0:.4g}, omega0 {w0:.4g}, Tu at inlet "
        f"{tu_in:.2f} percent, decay fit log-rms {fit_err:.3f}",
        flush=True,
    )
    n_plate = setup["mesh"]["n_plate"]
    sol = foam_body.solve(Stations(setup, n_plate), case_dir, fast=False)
    lines = plate_lines(case_dir, n_plate)
    out = {
        "model": model,
        "iterations": sol["iterations"],
        "converged": sol["converged"],
        "settled": sol["settled"],
        "beta": beta,
        "k_inlet": k0,
        "omega_inlet": w0,
        "tu_inlet_percent": tu_in,
        "decay_fit_log_rms": fit_err,
        "p0": lines["p0"],
        "plate": {n: thin(lines[n]) for n in ("x", "cf", "ue", "tau")},
        "free_stream": {
            "x": thin(lines["fs_x"], 300),
            "y": thin(lines["fs_y"], 300),
            "k": thin(lines["fs_k"], 300),
            "U": thin(lines["fs_U"], 300),
        },
    }
    if model == "clipKGammaLE":
        t = of.latest_time_dir(case_dir)
        tu = of.read_field(os.path.join(case_dir, t, "TuLE"))
        n_up = N_UP * N_NORMAL
        out["tule_wall"] = thin(tu[n_up : n_up + n_plate])
    os.makedirs(os.path.join(case_dir, "postProcessing"), exist_ok=True)
    with open(os.path.join(case_dir, "postProcessing", "t3.json"), "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    return key


def main():
    case = sys.argv[1]
    keys = sys.argv[2:] or list(MODELS)
    with open(SETUP) as f:
        setup = json.load(f)["cases"][case]
    with ThreadPoolExecutor(len(keys)) as pool:
        for key in pool.map(lambda k: run(case, k, setup), keys):
            print(f"{case} {key} done", flush=True)


if __name__ == "__main__":
    main()
