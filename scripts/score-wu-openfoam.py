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

Each run's transition onset is also recorded, as the first station where
its shape factor falls below H_ONSET (laminar is about 2.6 here, turbulent
about 1.4), beside the DNS's by the same definition, with the Re_theta
there, so an error in the score can be told apart as onset in the wrong
place or something after it. The C_f minimum was tried first and is not
defined where transition starts at the inlet or near the outlet.

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
H_ONSET = 2.2


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


def onset(x, H):
    """x and index of the first station where H falls below H_ONSET, or
    None if it never does (no transition in the domain)."""
    below = np.flatnonzero(H < H_ONSET)
    if len(below) == 0:
        return None, None
    i = int(below[0])
    return float(x[i]), i


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
            _, th, H = case._metrics(U)
            xo, i = onset(case.x, H)
            sc["x_onset"] = xo
            sc["re_theta_onset"] = (float(th[i] * case.re_theta0)
                                    if i is not None else None)
            out.setdefault(name, {})[model] = sc
            print(f"{name} {model}: {sc.get('normalized'):.3f}", flush=True)
    dns_onset = {}
    for tag in TAGS:
        name = "wu-bypass-tu" + tag[2:]
        case = cases[name].build()
        xo, i = onset(case.x, np.interp(case.x, case.x_H, case.H_ref))
        dns_onset[name] = {
            "x_onset": xo,
            "re_theta_onset": (float(np.interp(xo, case.x_th, case.theta_ref)
                                     * case.re_theta0)
                               if xo is not None else None)}
    ratios = {}
    for m in MODELS:
        r = [out[n][m]["re_theta_onset"] / dns_onset[n]["re_theta_onset"]
             for n in out
             if out[n][m].get("re_theta_onset") is not None
             and dns_onset[n]["re_theta_onset"] is not None]
        ratios[m] = {"n": len(r),
                     "min": float(min(r)) if r else None,
                     "max": float(max(r)) if r else None}
    onsets = {m: [out[n][m].get("re_theta_onset") for n in out] for m in MODELS}
    means = {m: float(np.mean([out[n][m]["normalized"] for n in out
                               if "normalized" in out[n][m]]))
             for m in MODELS}
    with open(OUT, "w") as f:
        json.dump({"models": list(MODELS), "cases": out, "mean": means,
                   "dns_onset": dns_onset,
                   "onset_ratio_to_dns": ratios,
                   "onset_re_theta_min": {m: min(v for v in o if v)
                                          for m, o in onsets.items()},
                   "onset_re_theta_max": {m: max(v for v in o if v)
                                          for m, o in onsets.items()}},
                  f, indent=2)
        f.write("\n")
    print(means)
    for name, d in dns_onset.items():
        print(name, "DNS", d["re_theta_onset"],
              {m: out[name][m].get("re_theta_onset") for m in MODELS})


if __name__ == "__main__":
    main()
