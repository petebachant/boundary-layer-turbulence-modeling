#!/usr/bin/env python
"""Run the leading-edge clipping closure with the onset floor on a T3
plate.

The runs are scripts/run-t3.py's for the closure, unchanged but for the
library, built from sim/clip-le-floor.patch into sim/clipLEFloor, and
cFloor, fitted to the DNS (results/onset-floor.json). Each case is
sim/cases/t3-<case>-clip-le-floor, so the closure's own runs are kept.

Usage: python scripts/run-t3-floor.py <case>

Outputs
-------
sim/cases/t3-<case>-clip-le-floor/postProcessing
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys

from pypkg import le_closures
from pypkg.cases import openfoam as of

FLOOR = "results/onset-floor.json"
KEY = "clip-le-floor"
LIB = le_closures.LIB.replace("/clipLE/", "/clipLEFloor/")


def main():
    case = sys.argv[1]
    with open(FLOOR) as f:
        c_floor = float(json.load(f)["c_floor"])
    assert LIB != le_closures.LIB
    le_closures.LIB = LIB
    write = of.write_model_coeffs

    def write_model_coeffs(case_dir, model, root="."):
        write(case_dir, model, root)
        if model != le_closures.MODEL:
            return
        path = os.path.join(case_dir, "constant", "turbulenceProperties")
        with open(path) as f:
            text = f.read()
        text, n = re.subn(
            r"(\n(\s*)wallTu\s+on;)",
            lambda m: f"{m.group(1)}\n{m.group(2)}cFloor       {c_floor:.8g};",
            text,
        )
        assert n == 1, f"{path} has no wallTu entry"
        with open(path, "w") as f:
            f.write(text)

    of.write_model_coeffs = write_model_coeffs
    spec = importlib.util.spec_from_file_location("run_t3", "scripts/run-t3.py")
    t3 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(t3)
    t3.MODELS[KEY] = t3.MODELS["clip-le"]
    with open(t3.SETUP) as f:
        setup = json.load(f)["cases"][case]
    t3.run(case, KEY, setup)
    print(f"{case} {KEY} done", flush=True)


if __name__ == "__main__":
    main()
