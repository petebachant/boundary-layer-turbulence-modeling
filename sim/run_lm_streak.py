"""Run Langtry-Menter with the streak onset correlation.

Sets the case up exactly as run.py does for k-omega-sst-lm, then, just
before the solver starts, switches it to kOmegaSSTLMStreak with the streak
correlation and loads that model's library. Wrapping run.py rather than
editing it keeps every existing result, all of which depend on run.py,
valid.

Usage: python run_lm_streak.py --c-streak C [run.py arguments...]
"""

import re
import runpy
import sys

import foampy

args = sys.argv[1:]
i = args.index("--c-streak")
c_streak = float(args[i + 1])
del args[i:i + 2]
sys.argv = ["run.py", "--turbulence-model", "k-omega-sst-lm"] + args

_run = foampy.run


def _switch_model():
    with open("constant/turbulenceProperties") as f:
        text = f.read()
    text, n = re.subn(r"(RASModel\s+)kOmegaSSTLM;",
                      r"\1kOmegaSSTLMStreak;", text)
    assert n == 1, "turbulenceProperties has no single kOmegaSSTLM entry"
    coeffs = ("    kOmegaSSTLMStreakCoeffs\n    {\n"
              "        onsetCorrelation streak;\n"
              f"        CStreak         {c_streak:.8g};\n    }}\n")
    text = re.sub(r"(RAS\s*\{\n)", r"\1" + coeffs, text, count=1)
    with open("constant/turbulenceProperties", "w") as f:
        f.write(text)
    with open("system/controlDict") as f:
        text = f.read()
    text, n = re.subn(r'libs\s*\("libransFromDns.so"\);',
                      'libs            ("libransFromDns.so" '
                      '"libkOmegaSSTLMStreak.so");', text)
    assert n == 1, "controlDict has no single libs entry to extend"
    with open("system/controlDict", "w") as f:
        f.write(text)
    print(f"Switched to kOmegaSSTLMStreak, CStreak = {c_streak:.4g}")


def run(app, *a, **kw):
    if app == "simpleFoam":
        _switch_model()
    return _run(app, *a, **kw)


foampy.run = run
runpy.run_path("run.py", run_name="__main__")
