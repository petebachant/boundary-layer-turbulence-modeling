#!/usr/bin/env python
"""Does the clipping closure transfer across free-stream intensity in
OpenFOAM when it reads the leading-edge intensity from the wall?

In the marching solver, scaling the closure's transition threshold as
1/Tu_in, the streak onset rule, makes it transfer across Wu et al.'s flows
(results/onset-rule-closure.json), with the inlet intensity handed to it.
Carried as a local scalar the intensity fails on the plate: the outer
layer, where the threshold acts, holds the intensity at which its fluid
entered, not the leading edge's (results/clip-le-calibration.json).
clipKGammaLE with wallTu on reads the scalar instead in the cell next to
each cell's nearest wall, where the fluid came from the start of the
layer, carried across the layer by the mesh wave that computes wall
distance. TuRef is the plate's inlet intensity, so on the plate the
threshold is the calibrated one and nothing new is fitted. It is run on
the plate and Wu et al.'s five flows set up exactly as clip-k-gamma is.

Tests, fixed before any of Wu et al.'s flows was run with this model:

  plate  its plate C_f error is within PLATE_TOL of the standard
         closure's
  wu     it beats the standard clipping closure's mean normalized score
         over Wu et al.'s flows in OpenFOAM (results/wu-openfoam.json) and
         wins on at least MIN_WINS of the five

Also reported: Langtry-Menter's and the marching solver's scores, and each
flow's onset Re_theta against the DNS's, from the rise of C_f as in
results/lm-onset-transfer.json.

Outputs
-------
results/clip-wall.json
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np

from pypkg import registry

OUT = "results/clip-wall.json"
WU = "results/wu-openfoam.json"
FAST = "results/onset-rule-closure.json"
PLATE_TOL = 0.10
MIN_WINS = 3


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log_rms(pairs):
    pairs = [(m, d) for m, d in pairs if m and d]
    if not pairs:
        return None, 0
    m, d = (np.array(v) for v in zip(*pairs))
    return float(np.sqrt(np.mean(np.log(m / d) ** 2))), len(pairs)


def main():
    mod = load("score_wu", "scripts/score-wu-openfoam.py")
    gate = load("gate", "scripts/test-lm-onset-transfer.py")
    streak = load("streak", "scripts/test-lm-streak.py")
    le = load("le", "scripts/test-lm-leading-edge.py")
    with open(WU) as f:
        wu = json.load(f)
    with open(FAST) as f:
        fast = json.load(f)
    cases = registry.cases()
    rows = {}
    for tag in mod.TAGS:
        name = "wu-bypass-tu" + tag[2:]
        case = cases[name].build()
        U, on = le.onset_re_theta(mod, gate, case, f"sim/cases/clip-wall-{tag}")
        _, on_clip = le.onset_re_theta(
            mod, gate, case, f"sim/cases/wu-{tag}-clip-k-gamma"
        )
        _, on_lm = le.onset_re_theta(
            mod, gate, case, f"sim/cases/wu-{tag}-k-omega-sst-lm"
        )
        cf_d = np.interp(case.x, case.x_cf, case.cf_ref)
        th_d = np.interp(case.x, case.x_th, case.theta_ref)
        _, th_on_d = gate.onset(case.x, cf_d, th_d)
        sc = case.score({"U": U})
        base = wu["cases"][name]
        rows[name] = {
            "wall": sc,
            "wall_normalized": sc["normalized"],
            "clip_normalized": base["clip-k-gamma"]["normalized"],
            "lm_normalized": base["k-omega-sst-lm"]["normalized"],
            "fast_normalized": fast["cases"][name]["a_normalized"],
            "wall_re_theta_onset": on,
            "clip_re_theta_onset": on_clip,
            "lm_re_theta_onset": on_lm,
            "dns_re_theta_onset": (
                float(th_on_d * case.re_theta0) if th_on_d else None
            ),
        }
        print(name, {k: v for k, v in rows[name].items() if k != "wall"})
    col = lambda k: np.array([r[k] for r in rows.values()])  # noqa: E731
    s, c = col("wall_normalized"), col("clip_normalized")
    wins = int(np.sum(s < c))
    errs = {}
    for key in ("wall", "clip", "lm"):
        e, n = log_rms(
            (r[f"{key}_re_theta_onset"], r["dns_re_theta_onset"])
            for r in rows.values()
        )
        errs[key] = {"log_rms": e, "n": n}
    cf_w, st = streak.plate_cf_error("sim/cases/clip-wall-plate")
    cf_c, _ = streak.plate_cf_error("sim/cases/clip-k-gamma-dns-domain")
    plate_ok = bool(cf_w <= (1 + PLATE_TOL) * cf_c)
    wu_ok = bool(s.mean() < c.mean() and wins >= MIN_WINS)
    result = {
        "plate_tol": PLATE_TOL,
        "min_wins": MIN_WINS,
        "cases": rows,
        "mean_wall": float(s.mean()),
        "mean_clip": float(c.mean()),
        "mean_lm": float(col("lm_normalized").mean()),
        "mean_fast": float(col("fast_normalized").mean()),
        "n_wins": wins,
        "n_wins_vs_lm": int(np.sum(s < col("lm_normalized"))),
        "onset_errors": errs,
        "plate_cf_err_wall": cf_w,
        "plate_cf_err_clip": cf_c,
        "plate_stations_wall": st,
        "plate_ok": plate_ok,
        "wu_ok": wu_ok,
        "passes": plate_ok and wu_ok,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        {
            k: v
            for k, v in result.items()
            if k not in ("cases", "plate_stations_wall")
        }
    )


if __name__ == "__main__":
    main()
