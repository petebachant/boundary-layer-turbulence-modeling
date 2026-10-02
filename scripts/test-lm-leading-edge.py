#!/usr/bin/env python
"""Does Langtry-Menter place transition better across free-stream intensity
when its layer keeps the intensity at which its fluid entered it?

Applied to the DNS, the streak onset rule Re_theta,onset = C / Tu_in
predicts Wu et al.'s onsets far better than Langtry-Menter's correlation,
and at the local intensity it fails (results/onset-history.json): the
layer needs the intensity from near the leading edge. Langtry-Menter's
ReThetat equation already holds ReThetat at its free-stream value and
transports it inside the layer, but diffuses the decayed values above
into it. The leading-edge variant is Langtry-Menter with the streak
correlation and that diffusion switched off (sigmaThetat = 0), so each
streamline keeps the value it entered the layer with, starting from the
correlation's own value at the inlet, C / Tu_in; calibrated by
running the model on the plate (results/lm-le-calibration.json) and run
on Wu et al.'s five flows set up exactly as the standard model.

Test, fixed before any of Wu et al.'s flows was run with this model: with
C calibrated on the plate, it beats standard Langtry-Menter's mean
normalized score on Wu et al.'s flows (results/wu-openfoam.json), the
same test the variant with diffusion took (results/lm-streak.json). Also
reported: its onset Re_theta against the DNS's on each flow, onset taken
from the rise of C_f as in results/lm-onset-transfer.json, beside the
standard model's and the variant with diffusion's, the plate's C_f
error. Whether the variants put different values of ReThetat in the
layer is checked from saved fields in results/lm-onset-gate.json.

Outputs
-------
results/lm-leading-edge.json
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

from pypkg import registry

OUT = "results/lm-leading-edge.json"
WU = "results/wu-openfoam.json"
CAL = "results/lm-le-calibration.json"
STREAK = "results/lm-streak.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log_err(pairs):
    pairs = [(m, d) for m, d in pairs if m and d]
    if not pairs:
        return None, 0
    m, d = map(np.array, zip(*pairs))
    return float(np.sqrt(np.mean(np.log(m / d) ** 2))), len(pairs)


def onset_re_theta(mod, gate, case, case_dir):
    st = mod.read_stations(case_dir)
    U, inside = mod.to_case_grid(case, st)
    first = int(np.argmax(inside))
    last = len(inside) - 1 - int(np.argmax(inside[::-1]))
    U[:, :first] = U[:, [first]]
    U[:, last + 1 :] = U[:, [last]]
    cf, th, _ = case._metrics(U)
    _, th_on = gate.onset(case.x, cf, th)
    return U, (float(th_on * case.re_theta0) if th_on else None)


def main():
    mod = load("score_wu", "scripts/score-wu-openfoam.py")
    gate = load("gate", "scripts/test-lm-onset-transfer.py")
    streak = load("streak", "scripts/test-lm-streak.py")
    with open(WU) as f:
        wu = json.load(f)
    cases = registry.cases()
    rows = {}
    for tag in mod.TAGS:
        name = "wu-bypass-tu" + tag[2:]
        case = cases[name].build()
        U, on = onset_re_theta(
            mod, gate, case, os.path.join("sim", "cases", f"lm-le-{tag}")
        )
        sc = case.score({"U": U})
        _, on_lm = onset_re_theta(
            mod, gate, case, f"sim/cases/wu-{tag}-k-omega-sst-lm"
        )
        _, on_st = onset_re_theta(mod, gate, case, f"sim/cases/lm-streak-{tag}")
        cf_d = np.interp(case.x, case.x_cf, case.cf_ref)
        th_d = np.interp(case.x, case.x_th, case.theta_ref)
        _, th_on_d = gate.onset(case.x, cf_d, th_d)
        rows[name] = {
            "le": sc,
            "le_normalized": sc["normalized"],
            "lm_normalized": wu["cases"][name]["k-omega-sst-lm"]["normalized"],
            "le_re_theta_onset": on,
            "lm_re_theta_onset": on_lm,
            "streak_re_theta_onset": on_st,
            "dns_re_theta_onset": (
                float(th_on_d * case.re_theta0) if th_on_d else None
            ),
        }
        print(name, {k: v for k, v in rows[name].items() if k != "le"})
    s = np.array([r["le_normalized"] for r in rows.values()])
    lm = np.array([r["lm_normalized"] for r in rows.values()])
    errs = {}
    for key in ("le", "lm", "streak"):
        e, n = log_err(
            (r[f"{key}_re_theta_onset"], r["dns_re_theta_onset"])
            for r in rows.values()
        )
        errs[key] = {"log_rms": e, "n": n}
    # How far apart the variants put onset, on the flows where all three
    # have one
    spread = []
    for r in rows.values():
        ons = [r[f"{k}_re_theta_onset"] for k in ("le", "lm", "streak")]
        if all(ons):
            spread.append(max(ons) / min(ons) - 1.0)
    with open(CAL) as f:
        c = json.load(f)["c_streak"]
    with open(STREAK) as f:
        mean_streak = json.load(f)["mean_streak"]
    cf_lm, _ = streak.plate_cf_error(streak.PLATE_LM)
    cf_le, st_le = streak.plate_cf_error(f"sim/cases/lm-le-plate-C{c}")
    result = {
        "c_streak": c,
        "cases": rows,
        "mean_le": float(s.mean()),
        "mean_lm": float(lm.mean()),
        "mean_streak": mean_streak,
        "n_wins": int(np.sum(s < lm)),
        "n_cases": len(rows),
        "onset_errors": errs,
        "max_onset_spread": max(spread) if spread else None,
        "plate_cf_err_le": cf_le,
        "plate_cf_err_lm": cf_lm,
        "plate_stations_le": st_le,
        "passes": bool(s.mean() < lm.mean()),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        {
            k: v
            for k, v in result.items()
            if k not in ("cases", "plate_stations_le")
        }
    )


if __name__ == "__main__":
    main()
