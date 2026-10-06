"""Set-up shared by the OpenFOAM cases that solve the whole flow around a
body (pypkg/airfoil_case.py, pypkg/bump_case.py).

Kept apart from each case's module so that a fix to one case's set-up does
not rerun every other case, and so the cases are solved the same way.
"""

from __future__ import annotations

import os
import re
import subprocess

import numpy as np


def patch_values(path, patch):
    """The nonuniform values of one boundary patch of an ASCII field."""
    with open(path) as f:
        text = f.read()
    body = text[text.index("boundaryField") :]
    m = re.search(rf"\n\s*{patch}\s*\n\s*\{{", body)
    if m is None:
        raise ValueError(f"{path} has no patch {patch}")
    block = body[m.end() :]
    v = re.search(
        r"value\s+nonuniform\s+List<(scalar|vector)>\s*(\d+)\s*\(", block
    )
    n = int(v.group(2))
    data = block[v.end() :]
    if v.group(1) == "scalar":
        return np.array(data.split(")")[0].split(), dtype=float)[:n]
    rows = re.findall(r"\(([^()]*)\)", data[: data.index("\n)")])[:n]
    return np.array([r.split() for r in rows], dtype=float)


def prepare_case(case_dir):
    """Settings every model needs on a whole-body case, applied after the
    benchmark harness has set the model up.

    The plate runs' schemes, which these cases start from, name every
    gradient they need and default to none, which Launder-Sharma's damping
    functions fall through. And a uniform initial field over a body, or a
    potential-flow one, is a start some models do not survive at the plate
    runs' relaxation factors (0.7 on U, 0.6 on the rest): on the Gaussian
    bump the clipping closure's omega grew without bound within twenty
    iterations, and a potential-flow start broke several models on the
    NACA 4412. Every model therefore starts from the uniform field, with U
    relaxed at 0.5 and every other equation at 0.3.
    """
    path = os.path.join(case_dir, "system", "fvSchemes")
    with open(path) as f:
        text = f.read()
    text, n = re.subn(
        r"(gradSchemes\s*\{\s*\n\s*default\s+)none;",
        r"\1Gauss linear;",
        text,
        count=1,
    )
    assert n == 1, f"{path} has no gradSchemes default"
    with open(path, "w") as f:
        f.write(text)
    path = os.path.join(case_dir, "system", "fvSolution")
    with open(path) as f:
        text = f.read()
    text, n1 = re.subn(r"(\n\s*U\s+)0\.7;", r"\g<1>0.5;", text, count=1)
    text, n2 = re.subn(r'(\n\s*"\.\*"\s+)0\.6;', r"\g<1>0.3;", text, count=1)
    assert n1 == n2 == 1, f"{path} lacks the plate runs' relaxation factors"
    with open(path, "w") as f:
        f.write(text)


#: Post-processing after each stretch of iterations, run in the case
#: directory
POST = (
    "postProcess -func writeCellCentres -latestTime "
    "> log.cellCentres 2>&1 && "
    "simpleFoam -postProcess -func wallShearStress -latestTime "
    "> log.wallShearStress 2>&1"
)
#: Iterations per stretch, and the most stretches a run gets
CHUNK = 2000
MAX_CHUNKS = 20
#: A run has settled once every scored error moved by less than this
#: fraction of its target over the last stretch
SETTLE = 0.05


def _set(path, pattern, value):
    with open(path) as f:
        text = f.read()
    text, n = re.subn(pattern, lambda m: m.group(1) + value, text, count=1)
    assert n == 1, f"{path} has no match for {pattern}"
    with open(path, "w") as f:
        f.write(text)


def _fast_relaxation(case_dir):
    """The relaxation SIMPLEC is meant for: the pressure field unrelaxed, U
    at 0.9 and every other equation at 0.7."""
    path = os.path.join(case_dir, "system", "fvSolution")
    _set(path, r"(fields\s*\{\s*\n\s*p\s+)[\d.]+", "1")
    _set(path, r"(\n\s*U\s+)[\d.]+", "0.9")
    _set(path, r'(\n\s*"\.\*"\s+)[\d.]+', "0.7")


def solve(case, case_dir, root=".", post=POST, fast=True):
    """Run simpleFoam on a prepared case until its scored errors settle.

    A fixed iteration count is not enough on these cases: with the pressure
    field relaxed under SIMPLEC (0.3, as in the plate runs) the pressure
    residual stalled near 1e-2, and on the Gaussian bump SST's separation
    bubble was still moving after 20,000 iterations, so its reported bubble
    depended on how the run was started (0.098 to 0.380 from potential flow,
    0.148 to 0.279 from a uniform field). Runs from both starts close on the
    same bubble once the pressure is unrelaxed, but slowly, so the solver
    runs in stretches of CHUNK iterations and stops once every error the
    case scores moved by less than SETTLE times its target over a stretch.
    The first stretch is run at prepare_case's gentle relaxation, which
    every model survives from the uniform start, and the rest at SIMPLEC's
    own where the case survives it (``fast``): on the NACA 4412 and the
    nosed plate SST went to NaN within 70 iterations of the switch, so
    those stay at their gentle relaxation throughout and rely on the
    settling criterion alone.

    ``post`` is the post-processing run after each stretch, in the case
    directory; with ``fast`` false every stretch keeps the case's own
    relaxation; ``case`` needs ``read_solution``, ``errors`` and ``TARGETS``.

    Returns the case's solution, with ``iterations``, ``settled`` and the
    errors after each stretch as ``history``.
    """
    rel = os.path.relpath(case_dir, root)
    control = os.path.join(case_dir, "system", "controlDict")
    _set(control, r"(\nstartFrom\s+)\w+", "latestTime")
    prev, history, settled = None, [], False
    for i in range(MAX_CHUNKS):
        if i == 1 and fast:
            _fast_relaxation(case_dir)
        _set(control, r"(\nendTime\s+)\d+", str((i + 1) * CHUNK))
        cmd = [
            "calkit",
            "xenv",
            "-n",
            "blsim",
            "--no-check",
            "--",
            "bash",
            "-c",
            f"source sim/foam-env.sh && cd {rel} && "
            "FOAM_SIGFPE=false simpleFoam >> log.simpleFoam 2>&1 && " + post,
        ]
        subprocess.run(cmd, cwd=root, check=True)
        sol = case.read_solution(case_dir)
        err = case.errors(sol)
        history.append({k: float(v) for k, v in err.items()})
        if not all(np.isfinite(v) for v in err.values()):
            break
        if prev is not None:
            settled = all(
                abs(err[k] - prev[k]) < SETTLE * tol
                for k, tol in case.TARGETS.items()
                if k in err and k in prev
            )
        if settled or sol.get("converged"):
            break
        prev = err
    sol["iterations"] = int(float(sol["time"]))
    sol["settled"] = settled
    sol["history"] = history
    return sol


def run_info(sol):
    """How long a run took to settle, for its scores' record."""
    if "iterations" not in sol:
        return {}
    return {
        "iterations": float(sol["iterations"]),
        "settled": float(sol["settled"]),
    }
