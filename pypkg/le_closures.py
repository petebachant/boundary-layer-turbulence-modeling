"""The clipping closure with its threshold scaled by the leading-edge
intensity, registered as a benchmark plugin.

Kept out of pypkg/closures.py so that adding it does not invalidate the
pipeline stages that read that module. It is loaded by naming this module
in RANS_BENCH_PLUGINS, which the OpenFOAM benchmark stage does.
"""

from __future__ import annotations

from .closures import JHTDB, OpenFoamOnly
from .registry import register_closure


@register_closure(
    "clip-k-omega-gamma-le",
    description=(
        "clip-k-omega-gamma with its transition threshold scaled as "
        "TuRef/TuLE, the streak onset rule: TuLE is the free-stream "
        "intensity carried into the layer, read next to each cell's nearest "
        "wall (sim/clipLE). OpenFOAM only."
    ),
    calibrated_on=(JHTDB,),
    openfoam_model="clipKGammaLE",
    python_tier=False,
)
class ClipKOmegaGammaLE(OpenFoamOnly):
    """clip-k-omega-gamma with its threshold scaled by the leading-edge
    intensity, read next to the nearest wall."""


# -- case setup for the OpenFOAM tier --------------------------------------
#
# The OpenFOAM case runner (pypkg/cases/openfoam.py) sets a model up by
# name. Teaching it this model there would change a module many stages
# read, so the setup is added here instead, by wrapping the runner's setup
# functions for clipKGammaLE and passing every other model straight
# through.

import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

from .cases import openfoam as _of  # noqa: E402

MODEL = "clipKGammaLE"
#: Its own library, outside FOAM_USER_LIBBIN, found from the case
LIB = "$FOAM_CASE/../../clipLE/platforms/$WM_OPTIONS/lib/libclipKGammaLE.so"
#: The plate's inlet intensity in percent, at which the threshold is the
#: calibrated one
INLET_PROFILES = "results/inlet-profiles.json"


def _tu_ref(root="."):
    with open(os.path.join(root, INLET_PROFILES)) as f:
        return float(json.load(f)["Tu_inlet_percent"])


def _write_model_coeffs(case_dir, model, root="."):
    """The calibrated clipKGamma block, renamed, plus TuRef and wallTu."""
    if model != MODEL:
        return _orig["write_model_coeffs"](case_dir, model, root)
    _orig["write_model_coeffs"](case_dir, "clipKGamma", root)
    path = os.path.join(case_dir, "constant", "turbulenceProperties")
    with open(path) as f:
        text = f.read()
    text, n = re.subn(
        r"\n(\s*)clipKGammaCoeffs\s*\n\s*\{",
        lambda m: (
            f"\n{m.group(1)}{MODEL}Coeffs\n{m.group(1)}{{\n"
            f"{m.group(1)}    TuRef        {_tu_ref(root)};\n"
            f"{m.group(1)}    wallTu       on;"
        ),
        text,
    )
    assert n == 1, f"{path} has no clipKGammaCoeffs block"
    with open(path, "w") as f:
        f.write(text)


def _ensure_libs(case_dir, model):
    if model != MODEL:
        return _orig["ensure_libs"](case_dir, model)
    path = os.path.join(case_dir, "system", "controlDict")
    with open(path) as f:
        text = f.read()
    if LIB not in text:
        with open(path, "w") as f:
            f.write(text.rstrip() + f'\n\nlibs ("{LIB}");\n')


def _ensure_model_fields(case_dir, model):
    """gamma as for clipKGamma, and TuLE. These flows are periodic, so no
    fluid enters with an intensity: TuLE starts at the reference, where the
    threshold is the calibrated one, and moves only where cells are
    free-stream-like."""
    if model != MODEL:
        return _orig["ensure_model_fields"](case_dir, model)
    _orig["ensure_model_fields"](case_dir, "clipKGamma")
    if not os.path.isfile(os.path.join(case_dir, "0", "TuLE")):
        _of.derive_field(
            case_dir,
            "TuLE",
            "[0 0 0 0 0 0 0]",
            _tu_ref(),
            "zeroGradient",
            None,
        )


def _prepare_fv_solution(case_dir, family):
    """The runner's settings, then upwind advection, a smoothing solver and
    relaxation for TuLE when the case runs this model.

    TuLE is only advected, with no diffusion to damp the oscillations the
    challenge's default central scheme would give it, and with almost no
    source its matrix is too weakly diagonal for the challenge's PBiCG,
    which diverged on a hill within a few iterations. It is solved as in
    the plate and Wu et al. runs, by symmetric Gauss-Seidel, and without
    the bounded form, whose subtracted flux divergence turned the diagonal
    negative under the large continuity errors of the hills' first
    iterations.
    """
    _orig["prepare_fv_solution"](case_dir, family)
    tp = os.path.join(case_dir, "constant", "turbulenceProperties")
    with open(tp) as f:
        if MODEL not in f.read():
            return
    path = os.path.join(case_dir, "system", "fvSchemes")
    with open(path) as f:
        text = f.read()
    text, n = re.subn(
        r"(divSchemes\s*\{\n)",
        r"\1    div(phi,TuLE)   Gauss upwind;\n",
        text,
        count=1,
    )
    assert n == 1, f"{path} has no divSchemes block"
    with open(path, "w") as f:
        f.write(text)
    path = os.path.join(case_dir, "system", "fvSolution")
    with open(path) as f:
        text = f.read()
    block = (
        "    TuLE\n    {\n        solver          smoothSolver;\n"
        "        smoother        symGaussSeidel;\n"
        "        nSweeps         1;\n        tolerance       1e-9;\n"
        "        relTol          0.1;\n    }\n"
    )
    text, n = re.subn(
        r"(solvers\s*\{\n)", lambda m: m.group(1) + block, text, count=1
    )
    assert n == 1, f"{path} has no solvers block"
    # The same equation relaxation as k, in whichever form the case uses
    m = re.search(r"\n(\s*)(\"?\(?[\w|]*\bk\b[\w|]*\)?\"?)\s+([\d.]+);", text)
    if m is not None:
        text = (
            text[: m.end()]
            + f"\n{m.group(1)}TuLE {m.group(3)};"
            + text[m.end() :]
        )
    with open(path, "w") as f:
        f.write(text)


_orig = {
    name: getattr(_of, name)
    for name in (
        "write_model_coeffs",
        "ensure_libs",
        "ensure_model_fields",
        "prepare_fv_solution",
    )
}
_of.write_model_coeffs = _write_model_coeffs
_of.ensure_libs = _ensure_libs
_of.ensure_model_fields = _ensure_model_fields
_of.prepare_fv_solution = _prepare_fv_solution
