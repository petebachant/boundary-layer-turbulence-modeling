#!/usr/bin/env python
"""Does a two-component streak closure carry streak growth between flows?

Lift-up forced by the local free-stream k grows the closure's streaks in
the right shape, but its coefficients fitted on one flow do not carry to
another in either direction (results/streak-growth.json), and the
free-stream length scale does not explain why
(results/freestream-length-scale.json). Lift-up is non-normal, a coupling
between two components: the wall-normal motions that force the streaks,
and the streaks, which do not force them back. StreakKOmegaGamma carries
the forcing as a transported roll energy k_v, fed from the free stream,
spreading into the layer by its own motions, decaying at the free
stream's rate and blocked by the wall, so a station feels the free stream
upstream of it rather than the one overhead.

Both closures, with transition off and the closure's own lift-up off
(C_L = 0), are solved on every point of their grids (GRID_1 for the
one-component closure, forced by the local k_inf; GRID_2 for the
two-component one) on the three flows with DNS streaks before onset: the
JHTDB plate and Wu et al.'s 0.75 and 1.5 percent flows. A flow's error
is the log-rms error in the peak sqrt(k)/U_e over the layer at its
stations, as in results/streak-growth.json.

Test, fixed before any flow was solved with the two-component closure
(outside a check that it runs): leave each flow out in turn, choose each
closure's coefficients by the mean error over the other two, and score
the held-out flow. The two-component closure passes if its mean held-out
error is below the one-component closure's and below MAX_HELD_OUT.

Outputs
-------
results/two-component-streaks.json
"""

from __future__ import annotations

import itertools
import json
import warnings

import numpy as np

from pypkg import closures, registry
from pypkg.cases.wu_bypass import peak_sqrt_k
from pypkg.dns_case import load_dns

OUT = "results/two-component-streaks.json"
ONSET = "results/bypass-onset.json"
MECHANICS = "results/transition-mechanics.json"
BASE = "clip-k-omega-gamma"
PLATE = "jhtdb-transitional-bl"
WU = ("wu-bypass-tu075", "wu-bypass-tu150")
GRID_1 = {"Cf": [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3],
          "Cd": [6.0, 30.0, 75.0, 150.0, 300.0, 750.0]}
GRID_2 = {"Cf": [0.01, 0.02, 0.03, 0.05, 0.1],
          "Cd": [6.0, 30.0, 150.0, 750.0],
          "Cv": [0.1, 1.0],
          "Cb": [0.3, 1.0, 3.0, 10.0, 30.0]}
MAX_HELD_OUT = 0.25
STATIONS = 8
OFF = {"Lam_c": 1e9, "CL": 0.0, "local_liftup": False}


def targets(cases):
    """{flow: (case, x stations, DNS peak sqrt(k)/U_e there)}"""
    out = {}
    plate = cases[PLATE].build()
    d = load_dns()
    with open(MECHANICS) as f:
        x_on = json.load(f)["x_transition_onset"]
    px = plate.case.x
    xs = np.linspace(px[0] + 0.1 * (x_on - px[0]), x_on, STATIONS)
    amp = np.sqrt(np.maximum(d["k"], 0.0)).max(axis=0) / d["U"].max(axis=0)
    out[PLATE] = (plate, px, xs, np.interp(xs, d["x"], amp))
    with open(ONSET) as f:
        onset = json.load(f)["cases"]
    for name in WU:
        case = cases[name].build()
        tag = "WM" + name.split("tu")[-1]
        rth_on = onset[tag]["re_theta_onset"]
        x_lo = case.x[0] + 0.1 * (case.x[-1] - case.x[0])
        xs, dns = [], []
        for rth, a in peak_sqrt_k(tag):
            x = float(np.interp(rth, case.theta_ref * case.re_theta0,
                                case.x_th))
            if (rth_on is None or rth <= rth_on) and x_lo <= x <= case.x[-1]:
                xs.append(x)
                dns.append(a)
        out[name] = (case, case.x, np.array(xs), np.array(dns))
    return out


def error(cls, spec, flow, **kw):
    case, cx, xs, dns = flow
    a = spec.get_coeffs()
    a.update(case.closure_kwargs(spec))
    a.update(OFF)
    a.update(kw)
    sol = case.run(cls(**a))
    amp = (np.sqrt(np.maximum(sol["k"], 0.0)).max(axis=0)
           / np.maximum(sol["U"].max(axis=0), 1e-12))
    m = np.maximum(np.interp(xs, cx, amp), 1e-12)
    return float(np.sqrt(np.mean(np.log(m / dns) ** 2)))


def table(cls, spec, flows, grid):
    keys = list(grid)
    rows = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        kw = dict(zip(keys, vals))
        errs = {n: error(cls, spec, f, **kw) for n, f in flows.items()}
        rows.append({"coeffs": kw, "errors": errs})
        print(" ", kw, {n: round(e, 3) for n, e in errs.items()}, flush=True)
    return rows


def leave_one_out(rows, names):
    folds = {}
    for held in names:
        train = [n for n in names if n != held]
        best = min(rows, key=lambda r: np.mean([r["errors"][n]
                                                for n in train]))
        folds[held] = {"coeffs": best["coeffs"],
                       "train_error": float(np.mean([best["errors"][n]
                                                     for n in train])),
                       "held_out_error": best["errors"][held]}
    joint = min(rows, key=lambda r: np.mean(list(r["errors"].values())))
    return {"folds": folds,
            "mean_held_out": float(np.mean([f["held_out_error"]
                                            for f in folds.values()])),
            "joint": {"coeffs": joint["coeffs"], "errors": joint["errors"],
                      "mean": float(np.mean(list(joint["errors"].values())))}}


def main():
    warnings.simplefilter("ignore")
    spec = registry.closures()[BASE]
    flows = targets(registry.cases())
    names = list(flows)
    print("one-component")
    rows_1 = table(closures.ClipKOmegaGamma, spec, flows, GRID_1)
    print("two-component")
    rows_2 = table(closures.StreakKOmegaGamma, spec, flows, GRID_2)
    one = leave_one_out(rows_1, names)
    two = leave_one_out(rows_2, names)
    result = {
        "flows": names, "grid_1": GRID_1, "grid_2": GRID_2,
        "max_held_out": MAX_HELD_OUT,
        "one_component": one, "two_component": two,
        "one_mean_held_out": one["mean_held_out"],
        "two_mean_held_out": two["mean_held_out"],
        "one_joint_mean": one["joint"]["mean"],
        "two_joint_mean": two["joint"]["mean"],
        "passes": bool(two["mean_held_out"] < one["mean_held_out"]
                       and two["mean_held_out"] < MAX_HELD_OUT),
        "table_1": rows_1, "table_2": rows_2,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for label, r in (("one", one), ("two", two)):
        print(label, "held out", {n: round(f["held_out_error"], 3)
                                  for n, f in r["folds"].items()},
              "mean", round(r["mean_held_out"], 3),
              "joint", round(r["joint"]["mean"], 3), r["joint"]["coeffs"])
    print("passes", result["passes"])


if __name__ == "__main__":
    main()
