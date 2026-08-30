"""Tier-2 cases: OpenFOAM setups from the Closure Challenge, scored against
the DNS interpolated onto the same mesh.

The Closure Challenge (McConkey et al. 2026) ships each of its cases as a
complete OpenFOAM directory -- mesh, boundary conditions, schemes -- with the
reference data interpolated onto the RANS mesh and a list of evaluation
points. Only its DNS cases are used here: the parameterized periodic hills of
Xiao et al. (2020) and the square/rectangular ducts of Vinuesa et al. (2014).
Both are flows the parabolic tier cannot reach: the hills separate, and the
ducts' secondary flow is driven by the anisotropy of the normal stresses,
which an eddy-viscosity model sets to zero by construction. The ducts are
therefore the one case in the suite that a linear eddy-viscosity model
*cannot* pass, whatever its coefficients, and the sharpest test of anything
beyond one.

Running a closure
-----------------
The case directory is copied under ``sim/cases/``, ``turbulenceProperties``
is written for the closure's registered OpenFOAM model (with the custom
model library loaded when it is one of ours), and ``simpleFoam`` is run in
the project's ``blsim`` environment through ``calkit xenv``, so the solver
and its version are the pipeline's rather than whatever is on the path.
The challenge's own iteration count is kept.

Scoring
-------
Cell values at the final time are sampled at the challenge's evaluation
points by nearest cell -- the DNS is on the same mesh, so both sides are
sampled identically and no interpolation error enters the comparison.
Errors are relative RMS over the points, normalized by the RMS of the DNS
quantity, which is scale-free and the same for a hill at Re = 5600 and a
duct at Re_tau = 360:

* ``U_rel_rms``: the full velocity vector.
* ``k_log_rms``: turbulent kinetic energy where the DNS has it.
* ``Usec_rel_rms`` (ducts): the in-plane (secondary) velocity, normalized
  by the DNS secondary flow itself, so a model that produces none scores
  exactly 1.0 on it.

The challenge scores a scaled MAE at the same points; the two are close
enough that a ranking here should match a ranking there, and the raw
per-point errors are kept in the solution for anyone who wants to submit.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import numpy as np

from ..registry import TIER_OPENFOAM, register_case
from .base import BenchmarkCase, log_rms

DATA_DIR = "data/closure-challenge"
CASES_DIR = "sim/cases"
#: Models we build ourselves, which the solver has to be told to load
CUSTOM_LIB = "libransFromDns.so"
CUSTOM_MODELS = ("clipKGamma", "ransFromDns")

CASES = {
    # name: (family, reference, data subdirectory, evaluation points file)
    "phll-alpha-15-13929-4048": ("periodic-hill", "Xiao2020",
                                 "phll_alpha_15_13929_4048",
                                 "alpha_15_13929_4048_points.csv"),
    "phll-alpha-15-13929-2024": ("periodic-hill", "Xiao2020",
                                 "phll_alpha_15_13929_2024",
                                 "alpha_15_13929_2024_points.csv"),
    "phll-alpha-05-4071-4048": ("periodic-hill", "Xiao2020",
                                "phll_alpha_05_4071_4048",
                                "alpha_05_4071_4048_points.csv"),
    "phll-alpha-05-4071-2024": ("periodic-hill", "Xiao2020",
                                "phll_alpha_05_4071_2024",
                                "alpha_05_4071_2024_points.csv"),
    "duct-ar-1-retau-180": ("duct", "Vinuesa2014", "duct_AR_1_Ret_180",
                            None),
    "duct-ar-1-retau-360": ("duct", "Vinuesa2014", "duct_AR_1_Ret_360",
                            "AR_1_Ret_360_points.csv"),
    "duct-ar-3-retau-360": ("duct", "Vinuesa2014", "duct_AR_3_Ret_360",
                            "AR_3_Ret_360_points.csv"),
    "duct-ar-14-retau-180": ("duct", "Vinuesa2014", "duct_AR_14_Ret_180",
                             "AR_14_Ret_180_points.csv"),
}


# -- OpenFOAM ASCII I/O ------------------------------------------------------

_NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"


def read_field(path):
    """The values of an ASCII OpenFOAM field file as an array.

    Handles a full field file (``internalField nonuniform List<vector>``)
    and the bare ``name nonuniform List<...>`` form the challenge uses for
    its reference fields, which are meant to be ``#include``d. A
    ``uniform`` field comes back as a 0-d or length-3 array.
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    m = re.search(r"nonuniform\s+List<(scalar|vector)>\s*(\d+)\s*\(", text)
    if m is not None:
        n = int(m.group(2))
        vals = np.fromstring(
            re.sub(r"[()]", " ", text[m.end():]).split(";")[0], sep=" "
        )
        if m.group(1) == "vector":
            return vals[: 3 * n].reshape(n, 3)
        return vals[:n]
    m = re.search(r"uniform\s*\(?([^;()]*)\)?\s*;", text)
    if m is None:
        raise ValueError(f"no field values in {path}")
    vals = [float(v) for v in re.findall(_NUM, m.group(1))]
    return np.array(vals[0] if len(vals) == 1 else vals)


def latest_time_dir(case_dir):
    """The numerically largest time directory, as a string."""
    times = []
    for name in os.listdir(case_dir):
        try:
            times.append((float(name), name))
        except ValueError:
            continue
    if not times:
        raise FileNotFoundError(f"no time directories in {case_dir}")
    return max(times)[1]


def write_turbulence_properties(case_dir, model):
    text = (
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        "    class dictionary;\n    object turbulenceProperties;\n}\n\n"
    )
    if model == "laminar":
        text += "simulationType laminar;\n"
    else:
        text += (
            "simulationType RAS;\n\nRAS\n{\n"
            f"    model {model};\n    turbulence on;\n"
            "    printCoeffs on;\n}\n"
        )
    with open(os.path.join(case_dir, "constant",
                           "turbulenceProperties"), "w") as f:
        f.write(text)


def _strip_functions(text):
    """Remove the ``functions { ... }`` block from a controlDict.

    The challenge's function objects include OpenFOAM.org configuration
    files that the OpenFOAM.com build in ``blsim`` does not ship, and
    nothing here reads their output: fields are read straight from the
    final time directory.
    """
    m = re.search(r"^\s*functions\s*\{", text, re.M)
    if m is None:
        return text
    depth, i = 0, m.end() - 1
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[:m.start()] + text[i + 1:]


def prepare_control_dict(case_dir, n_iter=None):
    path = os.path.join(case_dir, "system", "controlDict")
    with open(path) as f:
        text = f.read()
    text = _strip_functions(text)
    if n_iter is not None:
        text = re.sub(r"endTime\s+\S+;", f"endTime {int(n_iter)};", text)
        text = re.sub(r"writeInterval\s+\S+;",
                      f"writeInterval {int(n_iter)};", text)
    with open(path, "w") as f:
        f.write(text)


def ensure_model_fields(case_dir, model):
    """Fields our own model transports that the case does not provide.

    The activation fraction gamma is given the boundary structure of k --
    zero where k is held at zero, i.e., at walls -- and a fully active
    interior, since every case here is fully turbulent.
    """
    if model not in CUSTOM_MODELS:
        return
    zero = os.path.join(case_dir, "0")
    if os.path.isfile(os.path.join(zero, "gamma")):
        return
    with open(os.path.join(zero, "k")) as f:
        text = f.read()
    text = re.sub(r"object\s+k;", "object gamma;", text)
    text = re.sub(r"dimensions\s*\[[^\]]*\];", "dimensions [0 0 0 0 0 0 0];",
                  text)
    text = re.sub(r"internalField\s+uniform\s+\S+;",
                  "internalField uniform 1;", text)
    text = re.sub(r"internalField\s+nonuniform.*?\)\s*;",
                  "internalField uniform 1;", text, flags=re.S)
    with open(os.path.join(zero, "gamma"), "w") as f:
        f.write(text)


def ensure_libs(case_dir, model):
    """Load our model library from controlDict when the model is ours."""
    if model not in CUSTOM_MODELS:
        return
    path = os.path.join(case_dir, "system", "controlDict")
    with open(path) as f:
        text = f.read()
    if CUSTOM_LIB in text:
        return
    text = text.rstrip() + f'\n\nlibs ("{CUSTOM_LIB}");\n'
    with open(path, "w") as f:
        f.write(text)


# -- the case -------------------------------------------------------------------


class ClosureChallengeCase(BenchmarkCase):
    fidelity = "dns"
    # A separated or secondary flow at 5 % on the mean velocity is about
    # where the published DNS-to-DNS and LES-to-DNS comparisons land; the
    # secondary-flow target is loose because the question it asks is
    # whether a model produces any at all.
    TARGETS = {"U_rel_rms": 0.05, "k_log_rms": 0.30, "Usec_rel_rms": 0.50}

    #: Iteration caps. The hills keep the challenge's 20,000; the ducts'
    #: controlDict says 500,000 with residual control on k and omega only,
    #: which a model without omega would never satisfy, so they are capped
    #: at a count the SST baseline converged well inside (405 iterations).
    N_ITER = {"periodic-hill": None, "duct": 20000}

    def __init__(self, name, root=".", n_iter=None):
        family, reference, subdir, points = CASES[name]
        self.name, self.family, self.reference = name, family, reference
        self.root = root
        self.template = os.path.join(root, DATA_DIR, subdir)
        if not os.path.isdir(self.template):
            raise FileNotFoundError(self.template)
        self.n_iter = self.N_ITER[family] if n_iter is None else n_iter
        # Cell centres sit in 0/ for the hills and constant/ for the ducts
        for sub in ("0", "constant"):
            c_path = os.path.join(self.template, sub, "C")
            if os.path.isfile(c_path):
                break
        self.centres = read_field(c_path)
        self.U_dns = read_field(os.path.join(self.template, "0", "U_LES"))
        k_path = os.path.join(self.template, "0", "k_LES")
        self.k_dns = read_field(k_path) if os.path.isfile(k_path) else None
        if points is not None:
            pts = np.loadtxt(os.path.join(root, DATA_DIR, "evaluation_points",
                                          points), delimiter=",")
        else:
            pts = self.centres
        from scipy.spatial import cKDTree

        _, self.idx = cKDTree(self.centres).query(pts)
        # Streamwise direction: the DNS component with the largest RMS;
        # the other two carry the secondary flow (ducts) or nothing (hills)
        rms = np.sqrt(np.mean(self.U_dns[self.idx] ** 2, axis=0))
        self.i_stream = int(np.argmax(rms))
        self.i_sec = [i for i in range(3) if i != self.i_stream]
        if family != "duct":
            self.TARGETS = {k: v for k, v in self.TARGETS.items()
                            if k != "Usec_rel_rms"}
        self.last_case_dir = None

    def case_dir(self, closure_name):
        return os.path.join(self.root, CASES_DIR,
                            f"cc-{self.name}-{closure_name}")

    def run(self, closure):
        model = getattr(closure, "openfoam_model", None)
        name = getattr(closure, "gym_name", None) or "closure"
        if not model:
            raise NotImplementedError(
                f"{name} has no OpenFOAM model registered")
        case_dir = self.case_dir(name)
        if os.path.isdir(case_dir):
            shutil.rmtree(case_dir)
        shutil.copytree(self.template, case_dir)
        write_turbulence_properties(case_dir, model)
        ensure_libs(case_dir, model)
        ensure_model_fields(case_dir, model)
        prepare_control_dict(case_dir, self.n_iter)
        rel = os.path.relpath(case_dir, self.root)
        cmd = ["calkit", "xenv", "-n", "blsim", "--no-check", "--",
               "bash", "-c", f"cd {rel} && simpleFoam > log.simpleFoam 2>&1"]
        subprocess.run(cmd, cwd=self.root, check=True)
        self.last_case_dir = case_dir
        return self.read_solution(case_dir)

    def read_solution(self, case_dir):
        t = latest_time_dir(case_dir)
        if float(t) == 0.0:
            raise RuntimeError(f"no solution written in {case_dir}")
        sol = {"U": read_field(os.path.join(case_dir, t, "U")), "time": t}
        k_path = os.path.join(case_dir, t, "k")
        if os.path.isfile(k_path):
            sol["k"] = read_field(k_path)
        log = os.path.join(case_dir, "log.simpleFoam")
        if os.path.isfile(log):
            with open(log, errors="replace") as f:
                tail = f.read()[-4000:]
            sol["converged"] = "End" in tail
        return sol

    def errors(self, solution):
        U = np.asarray(solution["U"])[self.idx]
        Ud = self.U_dns[self.idx]
        if not np.all(np.isfinite(U)):
            return {"U_rel_rms": np.inf}
        errs = {
            "U_rel_rms": float(np.sqrt(np.mean(np.sum((U - Ud) ** 2, axis=1)))
                               / np.sqrt(np.mean(np.sum(Ud ** 2, axis=1)))),
        }
        if self.family == "duct":
            sec = U[:, self.i_sec]
            secd = Ud[:, self.i_sec]
            errs["Usec_rel_rms"] = float(
                np.sqrt(np.mean(np.sum((sec - secd) ** 2, axis=1)))
                / np.sqrt(np.mean(np.sum(secd ** 2, axis=1))))
        if self.k_dns is not None and solution.get("k") is not None:
            k = np.asarray(solution["k"])[self.idx]
            kd = self.k_dns[self.idx]
            # Away from the wall and the corners, where k falls through
            # decades and a log ratio measures the last cell's position
            # rather than the model
            m = kd > 0.05 * kd.max()
            if m.any():
                errs["k_log_rms"] = log_rms(np.maximum(k[m], 1e-12), kd[m])
        return errs

    def reference_score(self):
        """The DNS against itself: zero by construction, kept as the check
        that the sampling is identical on both sides."""
        return self.score({"U": self.U_dns, "k": self.k_dns})

    def describe(self):
        d = super().describe()
        d.update({"n_cells": int(len(self.centres)),
                  "n_points": int(len(self.idx)),
                  "streamwise_axis": self.i_stream,
                  "template": os.path.relpath(self.template, self.root)})
        return d


def make_case(name, root=".", **kw):
    return ClosureChallengeCase(name, root=root, **kw)


for _name, (_family, _ref, _sub, _pts) in CASES.items():
    register_case(
        _name,
        (lambda n: (lambda root=".": make_case(n, root=root)))(_name),
        family=_family,
        tier=TIER_OPENFOAM,
        reference=_ref,
        description=(
            f"Closure Challenge DNS case {_sub}: OpenFOAM setup with the DNS "
            "on the RANS mesh, scored at the challenge's evaluation points."
        ),
    )
