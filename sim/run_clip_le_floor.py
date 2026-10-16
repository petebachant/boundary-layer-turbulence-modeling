"""Run the leading-edge clipping closure with the onset floor.

sim/run_clip_le.py as it stands, with cFloor added to the closure's
coefficients just before the solver starts. The library is picked by
LD_LIBRARY_PATH (sim/run-clip-floor.sh).

Usage: python run_clip_le_floor.py --c-floor C [run_clip_le.py arguments...]
"""

import re
import runpy
import sys

import foampy

_i = sys.argv.index("--c-floor")
C_FLOOR = float(sys.argv[_i + 1])
del sys.argv[_i : _i + 2]
sys.argv[0] = "run_clip_le.py"

_run = foampy.run


def run(app, *a, **kw):
    if app == "simpleFoam":
        path = "constant/turbulenceProperties"
        with open(path) as f:
            text = f.read()
        text, n = re.subn(
            r"(clipKGammaLECoeffs\s*\{\n)",
            rf"\1        cFloor      {C_FLOOR:.8g};\n",
            text,
        )
        assert n == 1, "turbulenceProperties has no clipKGammaLECoeffs block"
        with open(path, "w") as f:
            f.write(text)
        print(f"cFloor = {C_FLOOR:.4g}")
    return _run(app, *a, **kw)


foampy.run = run
runpy.run_path("run_clip_le.py", run_name="__main__")
