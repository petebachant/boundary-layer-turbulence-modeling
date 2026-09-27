#!/usr/bin/env python
"""Score the OpenFOAM runs of Wu et al.'s bypass-transition flows exactly
as the Python tier scores them.

Each run samples the velocity at stations along its plate
(sim/run-wu.sh). Those profiles are interpolated onto the Python case's
own grid, in y at each station and then linearly in x between stations,
and handed to the case's own score, so C_f, theta, H and the velocity
profiles are measured against Wu et al.'s DNS the same way in both tiers.
Stations upstream of the first sampled one are left out by construction,
since the case already skips the first tenth of the plate.

Outputs
-------
results/wu-openfoam.json
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

from pypkg import registry

MODELS = ("k-omega-sst", "k-omega-sst-lm", "kkl-omega", "clip-k-gamma")
TAGS = ("WM075", "WM150", "WM225", "WM300", "WM600")
OUT = "results/wu-openfoam.json"


def read_stations(case_dir):
    root = os.path.join(case_dir, "postProcessing", "sample")
    times = sorted(glob.glob(os.path.join(root, "*")),
                   key=lambda p: float(os.path.basename(p)))
    if not times:
        return None
    out = []
    for path in glob.glob(os.path.join(times[-1], "x*.csv")):
        x = float(re.search(r"x(\d+)", os.path.basename(path)).group(1))
        df = pd.read_csv(path)
        out.append((x, df["y"].to_numpy(), df["U_0"].to_numpy()))
    return sorted(out, key=lambda r: r[0])


def to_case_grid(case, stations):
    """U on the case's (y, x) grid, NaN outside the sampled x range."""
    xs = np.array([s[0] for s in stations])
    prof = np.array([np.interp(case.y, s[1], s[2]) for s in stations])
    U = np.full((len(case.y), len(case.x)), np.nan)
    inside = (case.x >= xs[0]) & (case.x <= xs[-1])
    for j in range(len(case.y)):
        U[j, inside] = np.interp(case.x[inside], xs, prof[:, j])
    return U, inside


def main():
    cases = registry.cases()
    out = {}
    for tag in TAGS:
        name = "wu-bypass-tu" + tag[2:]
        case = cases[name].build()
        for model in MODELS:
            st = read_stations(os.path.join("sim", "cases",
                                            f"wu-{tag}-{model}"))
            if not st:
                out.setdefault(name, {})[model] = {"missing": True}
                continue
            U, inside = to_case_grid(case, st)
            # Columns beyond the first and last stations take the nearest
            # station's profile; the first is inside the tenth of the plate
            # the case does not score, the last within half a station
            # spacing of the end
            first = int(np.argmax(inside))
            last = len(inside) - 1 - int(np.argmax(inside[::-1]))
            U[:, :first] = U[:, [first]]
            U[:, last + 1:] = U[:, [last]]
            sc = case.score({"U": U})
            out.setdefault(name, {})[model] = sc
            print(f"{name} {model}: {sc.get('normalized'):.3f}", flush=True)
    means = {m: float(np.mean([out[n][m]["normalized"] for n in out
                               if "normalized" in out[n][m]]))
             for m in MODELS}
    with open(OUT, "w") as f:
        json.dump({"models": list(MODELS), "cases": out, "mean": means},
                  f, indent=2)
        f.write("\n")
    print(means)


if __name__ == "__main__":
    main()
