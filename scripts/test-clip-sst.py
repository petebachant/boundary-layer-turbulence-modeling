#!/usr/bin/env python
"""Does the leading-edge clipping closure's onset mechanism, carried on SST,
keep its bypass-transition gains and lose its fully turbulent losses?

clipKOmegaSSTLE (sim/clipSST) is SST whose k production is gated by the
leading-edge clipping closure's activation fraction, with the closure's
onset coefficients, TuRef and onset floor (docs/ideas-log.md, section
7.15). Tests, fixed before any run:

  turbulent  its mean score over the eight ducts and hills is at most
             TURB_TOL times SST's, and on the bump and the airfoil each at
             most BODY_TOL times SST's
  wu         its mean normalized score over Wu et al.'s five flows is
             below Langtry and Menter's
  t3         its mean |ln onset ratio| over T3A and T3B, set-up checks
             passed, is below Langtry and Menter's
  plate      its JHTDB plate C_f error from x = 30 is at most PLATE_TOL
             times the leading-edge closure's

Also reported: T3C2 to T3C5, Wu et al.'s onset Re_theta log-rms, and
every score against SST's, Langtry and Menter's and the leading-edge
closure's.

Outputs
-------
results/clip-sst.json
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

from pypkg import registry

OUT = "results/clip-sst.json"
T3 = "results/t3-rans.json"
SETUP = "results/t3-rans-setup.json"
CLIP_WALL = "results/clip-wall.json"
BENCH = {
    "turbulent": (
        "results/benchmark-openfoam.json",
        "results/benchmark-openfoam-clip-sst.json",
    ),
    "bump": (
        "results/benchmark-openfoam-bump.json",
        "results/benchmark-openfoam-bump-clip-sst.json",
    ),
    "airfoil": (
        "results/benchmark-openfoam-airfoil.json",
        "results/benchmark-openfoam-airfoil-clip-sst.json",
    ),
}
KEY = "clip-sst"
CLOSURE = "clip-sst-le"
OTHERS = {
    "sst": "k-omega-sst",
    "lm": "k-omega-sst-lm",
    "clip_le": "clip-k-omega-gamma-le",
}
TURB_TOL = 1.05
BODY_TOL = 1.10
PLATE_TOL = 1.10
T3_CASES = ("t3a", "t3b", "t3c2", "t3c3", "t3c4", "t3c5")
ZPG = ("t3a", "t3b")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mean_score(path, closure):
    with open(path) as f:
        res = json.load(f)["results"]
    s = [r[closure]["normalized"] for r in res.values() if closure in r]
    return float(np.mean(s)), len(s)


def t3_row(t3, case, s):
    """The run on a T3 plate, scored as scripts/test-t3-rans.py scores a
    run."""
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
    with open(T3) as f:
        base_t3 = json.load(f)["cases"]
    with open(SETUP) as f:
        setup = json.load(f)["cases"]
    with open(CLIP_WALL) as f:
        base_wu = json.load(f)

    bench = {}
    for name, (base, new) in BENCH.items():
        row = {"clip_sst": mean_score(new, CLOSURE)[0]}
        for k, c in OTHERS.items():
            row[k], row["n"] = mean_score(base, c)
        bench[name] = row
    turb_ok = bool(
        bench["turbulent"]["clip_sst"] <= TURB_TOL * bench["turbulent"]["sst"]
    )
    body_ok = bool(
        all(
            bench[n]["clip_sst"] <= BODY_TOL * bench[n]["sst"]
            for n in ("bump", "airfoil")
        )
    )

    plates = {}
    for case in T3_CASES:
        row = t3_row(t3, case, setup[case])
        for k in ("sst", "lm", "clip-le"):
            b = base_t3[case]["models"][k]
            row[f"{k.replace('-', '_')}_onset_ratio"] = b["onset_ratio"]
            row[f"{k.replace('-', '_')}_cf_err"] = b["cf_err"]
        plates[case] = row
        print(case, row)
    t3_mean = float(np.mean([abs(plates[c]["ln_onset_ratio"]) for c in ZPG]))
    t3_mean_lm = float(
        np.mean([abs(np.log(plates[c]["lm_onset_ratio"])) for c in ZPG])
    )
    t3_mean_le = float(
        np.mean([abs(np.log(plates[c]["clip_le_onset_ratio"])) for c in ZPG])
    )
    t3_ok = bool(
        all(plates[c]["setup_ok"] for c in ZPG) and t3_mean < t3_mean_lm
    )

    cases = registry.cases()
    wu = {}
    for tag in mod.TAGS:
        name = "wu-bypass-tu" + tag[2:]
        case = cases[name].build()
        U, on = le.onset_re_theta(mod, gate, case, f"sim/cases/clip-sst-{tag}")
        cf_d = np.interp(case.x, case.x_cf, case.cf_ref)
        th_d = np.interp(case.x, case.x_th, case.theta_ref)
        _, th_on_d = gate.onset(case.x, cf_d, th_d)
        sc = case.score({"U": U})
        base = base_wu["cases"][name]
        wu[name] = {
            "score": sc,
            "normalized": sc["normalized"],
            "clip_le_normalized": base["wall_normalized"],
            "lm_normalized": base["lm_normalized"],
            "re_theta_onset": on,
            "clip_le_re_theta_onset": base["wall_re_theta_onset"],
            "dns_re_theta_onset": (
                float(th_on_d * case.re_theta0) if th_on_d else None
            ),
        }
        print(name, {k: v for k, v in wu[name].items() if k != "score"})
    s = np.array([r["normalized"] for r in wu.values()])
    pairs = [
        (r["re_theta_onset"], r["dns_re_theta_onset"])
        for r in wu.values()
        if r["re_theta_onset"] and r["dns_re_theta_onset"]
    ]
    onset_log_rms = (
        float(
            np.sqrt(
                np.mean(
                    [np.log(m / d) ** 2 for m, d in pairs],
                )
            )
        )
        if pairs
        else None
    )
    wu_ok = bool(s.mean() < base_wu["mean_lm"])

    cf_plate, _ = streak.plate_cf_error("sim/cases/clip-sst-plate")
    plate_ok = bool(cf_plate <= PLATE_TOL * base_wu["plate_cf_err_wall"])

    result = {
        "turb_tol": TURB_TOL,
        "body_tol": BODY_TOL,
        "plate_tol": PLATE_TOL,
        "bench": bench,
        "plates": plates,
        "t3_mean_abs_ln": t3_mean,
        "t3_mean_abs_ln_lm": t3_mean_lm,
        "t3_mean_abs_ln_clip_le": t3_mean_le,
        "wu": wu,
        "wu_mean": float(s.mean()),
        "wu_mean_clip_le": base_wu["mean_wall"],
        "wu_mean_lm": base_wu["mean_lm"],
        "wu_onset_log_rms": onset_log_rms,
        "wu_onset_n": len(pairs),
        "wu_onset_log_rms_clip_le": base_wu["onset_errors"]["wall"]["log_rms"],
        "wu_onset_log_rms_lm": base_wu["onset_errors"]["lm"]["log_rms"],
        "plate_cf_err": cf_plate,
        "plate_cf_err_clip_le": base_wu["plate_cf_err_wall"],
        "turbulent_ok": turb_ok,
        "body_ok": body_ok,
        "wu_ok": wu_ok,
        "t3_ok": t3_ok,
        "plate_ok": plate_ok,
        "passes": turb_ok and body_ok and wu_ok and t3_ok and plate_ok,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items() if k not in ("plates", "wu")})


if __name__ == "__main__":
    main()
