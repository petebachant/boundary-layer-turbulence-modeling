"""Run SST carrying the leading-edge clipping closure's onset mechanism.

Sets the case up exactly as run.py does for k-omega-sst, free stream and
all, then, just before the solver starts, switches it to clipKOmegaSSTLE
(sim/clipSST), which gates SST's k production by the clipping closure's
activation fraction. The onset coefficients are the clipping closure's as
calibrated on the JHTDB plate (results/clip-k-gamma-coeffs.json), with its
leading-edge intensity TuRef, read at the wall, and its onset floor, as
sim/run-clip-wall.sh runs the leading-edge closure.

gamma enters at the case's inlet value and, unlike in the clipping
closure, has zero gradient at the wall, as Langtry and Menter's
intermittency does: the wall does not hold it at zero, so where it reaches
one, SST runs unchanged to the wall. TuLE is set as sim/run_clip_le.py
sets it, except that it is advected by plain upwind, as the closure's T3
and benchmark runs advect it (scripts/run-t3-clip-sst.py,
pypkg/sst_le_closures.py), not by the bounded form of the turbulence
scheme. In SST's first iterations on Wu et al.'s flows the near-wall
cells carry almost no flux, the bounded form's subtracted flux divergence
then dominates their diagonal, and TuLE ran away there until the solver
failed (WM300, iteration 34).

Usage: python run_clip_sst.py --tu-ref TU --c-floor C [--tu-le TU]
    [--lambda-c L] [run.py arguments...]
"""

import json
import os
import re
import runpy
import sys

import foampy


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def _pop(name, cast=float):
    value = _arg(name)
    if value is not None:
        i = sys.argv.index(name)
        del sys.argv[i : i + 2]
        value = cast(value)
    return value


PLATE = os.path.abspath("../results/inlet-profiles.json")
COEFFS = os.path.abspath("../results/clip-k-gamma-coeffs.json")
INLET_JSON = os.path.abspath(_arg("--inlet-json", PLATE))
INLET_KEY = _arg("--inlet-key")
TU_REF = _pop("--tu-ref")
C_FLOOR = _pop("--c-floor")
TU_LE = _pop("--tu-le")
#: The threshold in place of the calibrated one, for the plate check
LAMBDA_C = _pop("--lambda-c")
assert TU_REF is not None and C_FLOOR is not None, __doc__
sys.argv = ["run.py", "--turbulence-model", "k-omega-sst"] + sys.argv[1:]

MODEL = "clipKOmegaSSTLE"
#: The onset coefficients carried from the clipping closure
ONSET = ("Cgam", "LambdaC", "pExp", "gseed", "sigmaGamma")

_run = foampy.run

HEADER = """FoamFile
{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      TuLE;
}

dimensions      [0 0 0 0 0 0 0];

"""


def write_tule(tu):
    """0/TuLE with the patches of 0/k: held at tu on the inlet, empty where
    k is, zero gradient everywhere else."""
    with open("0/k") as f:
        k = f.read()
    patches = re.findall(
        r"\n    (\w+)\n    \{[^}]*?\btype\s+(\w+);",
        k[k.index("boundaryField") :],
    )
    lines = [HEADER, f"internalField   uniform {tu:.8g};\n\nboundaryField\n{{"]
    for name, kind in patches:
        if name == "inlet":
            body = f"type fixedValue; value uniform {tu:.8g};"
        elif kind == "empty":
            body = "type empty;"
        else:
            body = "type zeroGradient;"
        lines.append(f"    {name}\n    {{\n        {body}\n    }}")
    lines.append("}\n")
    with open("0/TuLE", "w") as f:
        f.write("\n".join(lines))


def free_gamma_at_walls():
    """gamma's wall patches, held at zero for the clipping closure, become
    zero gradient."""
    with open("0/gamma") as f:
        text = f.read()
    text, n = re.subn(
        r"type\s+fixedValue;\s*\n(\s*)value\s+uniform\s+0;",
        r"type            zeroGradient;",
        text,
    )
    assert n >= 1, "0/gamma has no wall held at zero"
    with open("0/gamma", "w") as f:
        f.write(text)


def _switch_model():
    with open(INLET_JSON) as f:
        prof = json.load(f)
    if INLET_KEY is not None:
        prof = prof[INLET_KEY]
    tu_in = prof["Tu_inlet_percent"] if TU_LE is None else TU_LE
    with open(COEFFS) as f:
        calibrated = json.load(f)["openfoam_coeffs"]
    onset = {name: float(calibrated[name]) for name in ONSET}
    if LAMBDA_C is not None:
        onset["LambdaC"] = LAMBDA_C
    onset.update(TuRef=TU_REF, cFloor=C_FLOOR)
    with open("constant/turbulenceProperties") as f:
        text = f.read()
    text, n = re.subn(r"(RASModel\s+)kOmegaSST;", rf"\1{MODEL};", text)
    assert n == 1, "turbulenceProperties has no single kOmegaSST entry"
    block = "".join(
        f"        {name:<12} {value:.10g};\n" for name, value in onset.items()
    )
    text, n = re.subn(
        r"(\n([ \t]*)printCoeffs\s+on;\n)",
        lambda m: (
            m.group(1)
            + f"\n{m.group(2)}{MODEL}Coeffs\n{m.group(2)}{{\n"
            + block
            + f"{m.group(2)}    wallTu       on;\n{m.group(2)}}}\n"
        ),
        text,
    )
    assert n == 1, "turbulenceProperties has no single printCoeffs entry"
    with open("constant/turbulenceProperties", "w") as f:
        f.write(text)
    with open("system/controlDict") as f:
        text = f.read()
    text, n = re.subn(
        r'libs\s*\("libransFromDns.so"\);',
        f'libs            ("libransFromDns.so" "lib{MODEL}.so");',
        text,
    )
    assert n == 1, "controlDict has no single libs entry to extend"
    with open("system/controlDict", "w") as f:
        f.write(text)
    free_gamma_at_walls()
    write_tule(tu_in)
    with open("system/fvSchemes") as f:
        text = f.read()
    text, n = re.subn(
        r"(\n(\s*)div\(phi,gamma\)[^\n]*\n)",
        r"\1\2div(phi,TuLE)   Gauss upwind;\n",
        text,
    )
    assert n == 1, "fvSchemes has no single div(phi,gamma) entry"
    text, n = re.subn(
        r"(gradSchemes\s*\{\n)", r"\1    grad(TuLE)      Gauss linear;\n", text
    )
    assert n == 1, "fvSchemes has no single gradSchemes block"
    with open("system/fvSchemes", "w") as f:
        f.write(text)
    with open("system/fvSolution") as f:
        text = f.read()
    text, n = re.subn(r"\|ReThetat\|", "|ReThetat|TuLE|", text)
    assert n == 1, "fvSolution has no single solver entry naming ReThetat"
    with open("system/fvSolution", "w") as f:
        f.write(text)
    print(
        f"Switched to {MODEL}: "
        + ", ".join(f"{k} = {v:.4g}" for k, v in onset.items())
        + f", TuLE inlet = {tu_in:.4g}"
    )


def run(app, *a, **kw):
    if app == "simpleFoam":
        _switch_model()
    return _run(app, *a, **kw)


foampy.run = run
runpy.run_path("run.py", run_name="__main__")
