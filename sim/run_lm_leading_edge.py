"""Run Langtry-Menter with the streak onset correlation and no diffusion of
its onset Reynolds number, so the layer keeps the free stream's intensity
from where its fluid entered it.

Langtry-Menter's ReThetat equation relaxes ReThetat to the correlation's
value at the local intensity in the free stream and is transported inside
the layer, where diffusion, sigmaThetat (nu + nut), mixes in the decayed
values above it. Without diffusion it is only advected there, so each
streamline carries the intensity at which it entered the layer, which near
the height where Re_v peaks is close to the leading edge's.

Wraps run_lm_streak.py, which in turn wraps run.py, so the case is set up
exactly as for the other Langtry-Menter runs, with two changes: sigmaThetat
is added to the model's coefficients, and ReThetat at the inlet, and as the
initial field, is the streak correlation's C / Tu_in rather than the value
of Langtry-Menter's correlation that run.py writes. Without diffusion the
layer keeps what enters it, so the inlet value is the one that matters; a
first calibration scan with run.py's value left in place showed C barely
moving the plate.

Usage: python run_lm_leading_edge.py --c-streak C [run.py arguments...]
"""

import json
import os
import re
import runpy
import sys

import foampy


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


C_STREAK = float(_arg("--c-streak"))
# Absolute, since the solver runs from inside the case directory
INLET_JSON = os.path.abspath(
    _arg("--inlet-json", "../results/inlet-profiles.json")
)
INLET_KEY = _arg("--inlet-key")

_run = foampy.run


def _no_diffusion():
    with open("constant/turbulenceProperties") as f:
        text = f.read()
    text, n = re.subn(
        r"(kOmegaSSTLMStreakCoeffs\s*\{\n)",
        r"\1        sigmaThetat     0;\n",
        text,
    )
    assert n == 1, "turbulenceProperties has no kOmegaSSTLMStreakCoeffs"
    with open("constant/turbulenceProperties", "w") as f:
        f.write(text)
    print("Set sigmaThetat = 0")
    with open(INLET_JSON) as f:
        prof = json.load(f)
    if INLET_KEY is not None:
        prof = prof[INLET_KEY]
    old = f"{prof['ReThetat_inlet']:.8g}"
    new = f"{C_STREAK / prof['Tu_inlet_percent']:.8g}"
    with open("0/ReThetat") as f:
        text = f.read()
    n = text.count(old)
    assert n >= 2, "0/ReThetat lacks run.py's inlet and initial values"
    with open("0/ReThetat", "w") as f:
        f.write(text.replace(old, new))
    print(f"Set ReThetat at the inlet and initially to {new}")


def run(app, *a, **kw):
    if app == "simpleFoam":
        _no_diffusion()
    return _run(app, *a, **kw)


foampy.run = run
runpy.run_path("run_lm_streak.py", run_name="__main__")
