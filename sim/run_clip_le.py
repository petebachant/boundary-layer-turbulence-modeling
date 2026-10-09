"""Run the clipping closure with a transported leading-edge intensity.

Sets the case up exactly as run.py does for clip-k-gamma, then, just before
the solver starts, switches it to clipKGammaLE (sim/clipLE), whose
transition threshold scales as TuRef/TuLE, the streak onset rule, with
TuLE the free-stream intensity carried into the layer from where its fluid
entered it. Every other coefficient stays as calibrated on the JHTDB plate;
TuRef is given, and calibrated on the plate by running this model, since
the plate's free stream decays as its layer grows and the layer's outer
part, where the threshold acts, holds a TuLE below the inlet's.

TuLE starts uniform at the inlet's measured intensity and is held there on
the inlet; every other patch takes it with zero gradient.

Usage: python run_clip_le.py --tu-ref TU [--tu-le TU] [--wall-tu]
    [run.py arguments...]
"""

import json
import os
import re
import runpy
import sys

import foampy


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


PLATE = os.path.abspath("../results/inlet-profiles.json")
INLET_JSON = os.path.abspath(_arg("--inlet-json", PLATE))
INLET_KEY = _arg("--inlet-key")
TU_REF = float(_arg("--tu-ref"))
_i = sys.argv.index("--tu-ref")
del sys.argv[_i : _i + 2]
# --tu-le: TuLE on the inlet, in place of the inlet's measured intensity.
# On the plate from x = 30 the layer has already carried the leading
# edge's intensity that far, which is more than the free stream's there
TU_LE = _arg("--tu-le")
if TU_LE is not None:
    _i = sys.argv.index("--tu-le")
    del sys.argv[_i : _i + 2]
    TU_LE = float(TU_LE)
# --wall-tu: the threshold reads TuLE next to each cell's nearest wall
WALL_TU = "--wall-tu" in sys.argv
if WALL_TU:
    sys.argv.remove("--wall-tu")
sys.argv = ["run.py", "--turbulence-model", "clip-k-gamma"] + sys.argv[1:]

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


def _switch_model():
    tu_ref = TU_REF
    with open(INLET_JSON) as f:
        prof = json.load(f)
    if INLET_KEY is not None:
        prof = prof[INLET_KEY]
    tu_in = prof["Tu_inlet_percent"] if TU_LE is None else TU_LE
    with open("constant/turbulenceProperties") as f:
        text = f.read()
    text, n = re.subn(r"(RASModel\s+)clipKGamma;", r"\1clipKGammaLE;", text)
    assert n == 1, "turbulenceProperties has no single clipKGamma entry"
    text, n = re.subn(
        r"clipKGammaCoeffs(\s*\{\n)",
        rf"clipKGammaLECoeffs\1        TuRef       {tu_ref:.8g};\n"
        + ("        wallTu      on;\n" if WALL_TU else ""),
        text,
    )
    assert n == 1, "turbulenceProperties has no clipKGammaCoeffs block"
    with open("constant/turbulenceProperties", "w") as f:
        f.write(text)
    with open("system/controlDict") as f:
        text = f.read()
    text, n = re.subn(
        r'libs\s*\("libransFromDns.so"\);',
        'libs            ("libransFromDns.so" "libclipKGammaLE.so");',
        text,
    )
    assert n == 1, "controlDict has no single libs entry to extend"
    with open("system/controlDict", "w") as f:
        f.write(text)
    write_tule(tu_in)
    # The case's own scheme and solver entries for TuLE, as for gamma
    with open("system/fvSchemes") as f:
        text = f.read()
    text, n = re.subn(
        r"(\n(\s*)div\(phi,gamma\)[^\n]*\n)",
        r"\1\2div(phi,TuLE)   $turbulence;\n",
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
        f"Switched to clipKGammaLE, TuRef = {tu_ref:.4g}, TuLE inlet = {tu_in:.4g}"
    )


def run(app, *a, **kw):
    if app == "simpleFoam":
        _switch_model()
    return _run(app, *a, **kw)


foampy.run = run
runpy.run_path("run.py", run_name="__main__")
