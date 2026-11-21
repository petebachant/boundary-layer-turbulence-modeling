"""SST with the leading-edge clipping closure's onset mechanism, registered
as a benchmark plugin.

clipKOmegaSSTLE (sim/clipSST) is OpenFOAM's kOmegaSST with the leading-edge
clipping closure's activation fraction gamma, and the intensity TuLE that
sets its threshold, gating SST's k production (docs/ideas-log.md, section
7.15). Loaded by naming this module in RANS_BENCH_PLUGINS, so adding it
leaves pypkg/closures.py and pypkg/le_closures.py, and every stage reading
them, untouched.
"""

from __future__ import annotations

import json
import os
import re

from .cases import openfoam as _of
from .closures import JHTDB, OpenFoamOnly
from .le_closures import WU, _tu_ref, c_floor
from .registry import register_closure

MODEL = "clipKOmegaSSTLE"
#: Its own library, outside FOAM_USER_LIBBIN, found from the case
LIB = "$FOAM_CASE/../../clipSST/platforms/$WM_OPTIONS/lib/libclipKOmegaSSTLE.so"
#: The onset coefficients carried from the clipping closure
ONSET = ("Cgam", "LambdaC", "pExp", "gseed", "sigmaGamma")
COEFFS = "results/clip-k-gamma-coeffs.json"
#: LambdaC, the one coefficient refitted, on the JHTDB plate
#: (scripts/calibrate-clip-sst.py)
CALIBRATION = "results/clip-sst-calibration.json"


@register_closure(
    "clip-sst-le",
    description=(
        "k-omega SST with the leading-edge clipping closure's onset "
        "mechanism: its activation fraction, with the threshold scaled by "
        "the leading-edge intensity and the onset floor, gates SST's k "
        "production (sim/clipSST). OpenFOAM only."
    ),
    calibrated_on=(JHTDB, *WU),
    openfoam_model=MODEL,
    python_tier=False,
)
class ClipSSTLE(OpenFoamOnly):
    """SST with the leading-edge clipping closure's onset mechanism."""


def onset_coeffs(root="."):
    """The onset coefficients as the leading-edge closure runs them, with
    LambdaC refitted on the plate."""
    with open(os.path.join(root, COEFFS)) as f:
        calibrated = json.load(f)["openfoam_coeffs"]
    out = {name: float(calibrated[name]) for name in ONSET}
    with open(os.path.join(root, CALIBRATION)) as f:
        out["LambdaC"] = float(json.load(f)["lambda_c"])
    out.update(TuRef=_tu_ref(root), cFloor=c_floor(root))
    return out


def _write_model_coeffs(case_dir, model, root="."):
    """SST's defaults, plus the onset block."""
    if model != MODEL:
        return _orig["write_model_coeffs"](case_dir, model, root)
    path = os.path.join(case_dir, "constant", "turbulenceProperties")
    with open(path) as f:
        text = f.read()
    block = "".join(
        f"        {name:<12} {value:.10g};\n"
        for name, value in onset_coeffs(root).items()
    )
    text, n = re.subn(
        r"(\n([ \t]*)(?:RASModel|model)\s+clipKOmegaSSTLE;[^\n]*\n)",
        lambda m: (
            m.group(1)
            + f"\n{m.group(2)}{MODEL}Coeffs\n{m.group(2)}{{\n"
            + block
            + f"{m.group(2)}    wallTu       on;\n{m.group(2)}}}\n"
        ),
        text,
    )
    assert n == 1, f"{path} has no single model {MODEL} entry"
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
    """gamma at one, the fully turbulent value, with zero gradient at the
    walls, and TuLE at the reference, where the threshold is the
    calibrated one. These flows are periodic, so no fluid enters with an
    intensity."""
    if model != MODEL:
        return _orig["ensure_model_fields"](case_dir, model)
    zero = os.path.join(case_dir, "0")
    for name, value in (("gamma", 1.0), ("TuLE", _tu_ref())):
        if not os.path.isfile(os.path.join(zero, name)):
            _of.derive_field(
                case_dir,
                name,
                "[0 0 0 0 0 0 0]",
                value,
                "zeroGradient",
                None,
            )


def _prepare_fv_solution(case_dir, family):
    """The runner's settings, then TuLE's, as for the leading-edge closure
    (pypkg/le_closures.py): upwind advection, a smoothing solver, and k's
    relaxation."""
    _orig["prepare_fv_solution"](case_dir, family)
    tp = os.path.join(case_dir, "constant", "turbulenceProperties")
    with open(tp) as f:
        if MODEL not in f.read():
            return
    path = os.path.join(case_dir, "system", "fvSchemes")
    with open(path) as f:
        text = f.read()
    if "div(phi,TuLE)" not in text:
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
    if re.search(r"\n\s*TuLE\s*\n\s*\{", text):
        return
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
