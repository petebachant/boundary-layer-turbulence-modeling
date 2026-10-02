"""Run the leading-edge variant of Langtry-Menter with only its onset
correlation able to start transition.

Langtry-Menter starts producing intermittency either where Re_v passes the
correlation's critical value or where the eddy-viscosity ratio R_T reaches
2.5, and on Wu et al.'s flows the second starts transition before the
first could (results/lm-onset-gate.json), so the correlation never sets
onset. kOmegaSSTLMGate (sim/lmGate) is kOmegaSSTLMStreak with a switch,
rtOnset, that closes the second route.

Wraps run_lm_leading_edge.py, so the case is set up as for the leading-edge
variant (streak correlation, no diffusion of ReThetat, C / Tu_in at the
inlet); this only swaps in kOmegaSSTLMGate, its library, and rtOnset off.

Usage: python run_lm_gate.py --c-streak C [run.py arguments...]
"""

import re
import runpy

import foampy

_run = foampy.run


def _gate():
    with open("constant/turbulenceProperties") as f:
        text = f.read()
    text, n = re.subn(r"kOmegaSSTLMStreak", "kOmegaSSTLMGate", text)
    assert n == 2, "expected the model and its coefficients block"
    text = re.sub(
        r"(kOmegaSSTLMGateCoeffs\s*\{\n)",
        r"\1        rtOnset         off;\n",
        text,
    )
    with open("constant/turbulenceProperties", "w") as f:
        f.write(text)
    with open("system/controlDict") as f:
        text = f.read()
    text, n = re.subn(
        r'"libkOmegaSSTLMStreak.so"', '"libkOmegaSSTLMGate.so"', text
    )
    assert n == 1, "controlDict does not load the streak library"
    with open("system/controlDict", "w") as f:
        f.write(text)
    print("Switched to kOmegaSSTLMGate with rtOnset off")


def run(app, *a, **kw):
    if app == "simpleFoam":
        _gate()
    return _run(app, *a, **kw)


foampy.run = run
runpy.run_path("run_lm_leading_edge.py", run_name="__main__")
