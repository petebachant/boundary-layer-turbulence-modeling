#!/usr/bin/env python
"""The one refit allowed SST with the leading-edge clipping closure's onset
mechanism: its threshold LambdaC, on the JHTDB plate alone.

Run with the closure's own LambdaC, the plate's C_f error from x = 30 is
above 1.1 times the closure's, the condition fixed beforehand for this
refit (docs/ideas-log.md, section 7.15). The scan (clip-sst-plate-scan)
runs the model over a grid of LambdaC; the value with the lowest plate C_f
error, scored as scripts/test-lm-streak.py scores the plate, is the one
every other run uses. A best value at either end of the grid is flagged,
since the grid would then not bracket it.

Outputs
-------
results/clip-sst-calibration.json
"""

from __future__ import annotations

import importlib.util
import json
import os
import re

OUT = "results/clip-sst-calibration.json"
CASES = "sim/cases"
PATTERN = re.compile(r"^clip-sst-plate-lambda-([\d.]+)$")


def main():
    spec = importlib.util.spec_from_file_location(
        "streak", "scripts/test-lm-streak.py"
    )
    streak = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(streak)
    errors = {}
    for name in sorted(os.listdir(CASES)):
        m = PATTERN.match(name)
        if m is None:
            continue
        err, _ = streak.plate_cf_error(os.path.join(CASES, name))
        errors[float(m.group(1))] = err
    grid = sorted(errors)
    best = min(grid, key=errors.get)
    out = {
        "lambda_c": best,
        "plate_cf_err": errors[best],
        "grid": grid,
        "plate_cf_err_by_lambda_c": {f"{k:g}": errors[k] for k in grid},
        "at_grid_edge": best in (grid[0], grid[-1]),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(out)


if __name__ == "__main__":
    main()
