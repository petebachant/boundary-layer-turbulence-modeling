#!/usr/bin/env python
"""Does an onset floor fitted to the DNS move the leading-edge clipping
closure's T3B onset toward the LES's, at no cost where it was built?

The closure's onset falls where Tu_LE^2 Re_x reaches a fixed value, so it
transitions T3B, whose onset is at a larger value than any DNS's, at
0.37 of the LES's onset (results/t3-rans.json,
results/clip-le-onset-bias.json). The floor (scripts/fit-onset-floor.py)
adds a lowest onset Re_x, fitted with the rule's constant to the six
zero-pressure-gradient DNS alone, and carried to the closure as cFloor,
which leaves its threshold at the plate's leading-edge intensity as
calibrated (sim/clip-le-floor.patch). If the closure's onset moves by the
floor's factor alone, its T3B ratio becomes about 0.6 and its T3A ratio
moves from 0.56 to about 0.6 (results/onset-floor.json): T3A's offset is
not one the DNS show, and is a documented limitation (docs/ideas-log.md,
section 7.9).

The closure with the floor is run on T3A and T3B as scripts/run-t3.py
runs the closure (scripts/run-t3-floor.py), and on the JHTDB plate from
x = 30 and Wu et al.'s five flows as sim/run-clip-wall.sh does
(sim/run-clip-floor.sh). Nothing in it is fitted to T3A or T3B, to the
plate's onset or to the experiments.

Tests, fixed before any run with the floor:

  t3b    its T3B onset ratio, on test-t3-rans.py's measure and with its
         set-up check passed, has |ln ratio| at most T3B_SHARE of the
         closure's own on the same plate
  wu     its mean normalized score over Wu et al.'s five flows is within
         TOL of the closure's (results/clip-wall.json)
  plate  its plate C_f error from x = 30 is within TOL of the closure's

Also reported: its T3A ratio and C_f errors; on each run the factor by
which onset moved from the closure's against the factor the fit
predicts; Wu et al.'s onset Re_theta log-rms; and, out of sample, its
T3A and T3B onset Re_x against the ERCOFTAC experiments'
(results/onset-inflow.json), with the closure's.

Outputs
-------
results/onset-floor-test.json
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

from pypkg import registry

OUT = "results/onset-floor-test.json"
FLOOR = "results/onset-floor.json"
T3 = "results/t3-rans.json"
CLIP_WALL = "results/clip-wall.json"
INFLOW = "results/onset-inflow.json"
SETUP = "results/t3-rans-setup.json"
KEY = "clip-le-floor"
T3B_SHARE = 2.0 / 3.0
TOL = 0.10


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def t3_row(t3, case, s):
    """The floor run on a T3 plate, scored as scripts/test-t3-rans.py
    scores a run."""
    les = s["les"]
    lx, lue = np.array(les["x"]), np.array(les["ue"])
    px, pcf = np.array(les["plate_x"]), np.array(les["cf"])
    x_on, x_end = les["x_onset"], s["x_sponge"]
    path = os.path.join(
        "sim", "cases", f"t3-{case}-{KEY}", "postProcessing", "t3.json"
    )
    with open(path) as f:
        r = json.load(f)
    x, cf, ue = (np.array(r["plate"][n]) for n in ("x", "cf", "ue"))
    xo, done = t3.onset(x, cf, ue, x_end)
    ratio = (xo if done else x_end) / x_on
    m = (x >= t3.X_SKIP) & (x <= x_on)
    ue_err = float(np.max(np.abs(ue[m] / np.interp(x[m], lx, lue) - 1)))
    m = (x >= t3.X_SKIP) & (x <= x_end)
    cf_err = float(np.mean(np.abs(cf[m] / np.interp(x[m], px, pcf) - 1)))
    return {
        "transitioned": done,
        "x_onset": xo if done else None,
        "onset_ratio": float(ratio),
        "ln_onset_ratio": float(np.log(ratio)),
        "ue_err": ue_err,
        "setup_ok": ue_err <= t3.UE_TOL,
        "cf_err": cf_err,
        "iterations": r["iterations"],
        "converged": r["converged"],
        "settled": r["settled"],
    }


def main():
    t3 = load("t3_rans", "scripts/test-t3-rans.py")
    mod = load("score_wu", "scripts/score-wu-openfoam.py")
    gate = load("gate", "scripts/test-lm-onset-transfer.py")
    streak = load("streak", "scripts/test-lm-streak.py")
    le = load("le", "scripts/test-lm-leading-edge.py")
    with open(FLOOR) as f:
        floor = json.load(f)
    with open(T3) as f:
        base_t3 = json.load(f)["cases"]
    with open(CLIP_WALL) as f:
        base_wu = json.load(f)
    with open(INFLOW) as f:
        inflow = json.load(f)["flows"]
    with open(SETUP) as f:
        setup = json.load(f)["cases"]
    pred = floor["predicted"]

    plates = {}
    for case in t3.ZPG:
        row = t3_row(t3, case, setup[case])
        base = base_t3[case]["models"][t3.LE_MODEL]
        row["closure_onset_ratio"] = base["onset_ratio"]
        row["closure_cf_err"] = base["cf_err"]
        row["lm_onset_ratio"] = base_t3[case]["models"]["lm"]["onset_ratio"]
        row["factor"] = row["onset_ratio"] / base["onset_ratio"]
        row["predicted_factor"] = pred[case]["factor"]
        exp = inflow[f"{case}-exp"]["re_x_onset"]
        on = row["x_onset"] or setup[case]["x_sponge"]
        row["re_x_onset"] = on / t3.NU
        row["exp_re_x_onset"] = exp
        row["exp_onset_ratio"] = on / t3.NU / exp
        row["closure_exp_onset_ratio"] = (
            (base["x_onset"] or setup[case]["x_sponge"]) / t3.NU / exp
        )
        plates[case] = row
        print(case, row)

    cases = registry.cases()
    wu = {}
    for tag in mod.TAGS:
        name = "wu-bypass-tu" + tag[2:]
        case = cases[name].build()
        U, on = le.onset_re_theta(
            mod, gate, case, f"sim/cases/clip-floor-{tag}"
        )
        cf_d = np.interp(case.x, case.x_cf, case.cf_ref)
        th_d = np.interp(case.x, case.x_th, case.theta_ref)
        _, th_on_d = gate.onset(case.x, cf_d, th_d)
        sc = case.score({"U": U})
        base = base_wu["cases"][name]
        wu[name] = {
            "floor": sc,
            "floor_normalized": sc["normalized"],
            "closure_normalized": base["wall_normalized"],
            "floor_re_theta_onset": on,
            "closure_re_theta_onset": base["wall_re_theta_onset"],
            "dns_re_theta_onset": (
                float(th_on_d * case.re_theta0) if th_on_d else None
            ),
            "predicted_factor_re_x": pred[tag]["factor"],
        }
        print(name, {k: v for k, v in wu[name].items() if k != "floor"})
    s = np.array([r["floor_normalized"] for r in wu.values()])
    pairs = [
        (r["floor_re_theta_onset"], r["dns_re_theta_onset"])
        for r in wu.values()
        if r["floor_re_theta_onset"] and r["dns_re_theta_onset"]
    ]
    m_, d_ = (np.array(v) for v in zip(*pairs))
    onset_log_rms = float(np.sqrt(np.mean(np.log(m_ / d_) ** 2)))
    cf_plate, _ = streak.plate_cf_error("sim/cases/clip-floor-plate")

    t3b = plates["t3b"]
    t3b_ok = bool(
        t3b["setup_ok"]
        and abs(t3b["ln_onset_ratio"])
        <= T3B_SHARE * abs(np.log(t3b["closure_onset_ratio"]))
    )
    wu_ok = bool(s.mean() <= (1 + TOL) * base_wu["mean_wall"])
    plate_ok = bool(cf_plate <= (1 + TOL) * base_wu["plate_cf_err_wall"])
    result = {
        "t3b_share": T3B_SHARE,
        "tol": TOL,
        "c_floor": floor["c_floor"],
        "plates": plates,
        "wu": wu,
        "wu_mean": float(s.mean()),
        "wu_mean_closure": base_wu["mean_wall"],
        "wu_mean_lm": base_wu["mean_lm"],
        "wu_onset_log_rms": onset_log_rms,
        "wu_onset_n": len(pairs),
        "wu_onset_log_rms_closure": base_wu["onset_errors"]["wall"]["log_rms"],
        "plate_cf_err": cf_plate,
        "plate_cf_err_closure": base_wu["plate_cf_err_wall"],
        "t3b_ok": t3b_ok,
        "wu_ok": wu_ok,
        "plate_ok": plate_ok,
        "passes": t3b_ok and wu_ok and plate_ok,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        {k: v for k, v in result.items() if k not in ("plates", "wu")},
    )


if __name__ == "__main__":
    main()
