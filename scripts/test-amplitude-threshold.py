#!/usr/bin/env python
"""A transition threshold on the closure's own streak amplitude.

Bypass transition begins near a fixed peak streak amplitude whatever the
free-stream intensity (results/streak-headroom.json). A closure carries a
fluctuation energy, so it can switch on where its own streak amplitude,
sqrt(k_s)/U_e, crosses a fixed value, with nothing to know about an inlet;
the "amp" driver of pypkg.closures.THRESHOLD_PARAMS.

clip-k-omega-gamma is given that driver in place of the vorticity Reynolds
number, with every other coefficient as calibrated on the JHTDB plate, and
its threshold is calibrated on the plate alone, by a scan over GRID. It is
then scored on Wu et al.'s five bypass-transition flows, which it never
saw, against the same closure with its calibrated Re_v threshold.

Test, fixed before any Wu flow was solved with the amplitude threshold:
the amplitude closure beats the Re_v closure on the mean normalized score
over the five flows, and on at least MIN_WINS of them.

Diagnostic, recorded alongside: with transition switched off, how the
closure's own peak streak amplitude grows along each flow, with lift-up
driven by its own k and by the free stream, against the DNS's peak
sqrt(k) at the stations where the dataset gives profiles, and at onset
(intermittency 0.1, results/bypass-onset.json). A threshold on the
model's amplitude can only work if that amplitude grows like the DNS's.

Outputs
-------
results/amplitude-threshold.json
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from pypkg import registry
from pypkg.cases.wu_bypass import peak_sqrt_k

OUT = "results/amplitude-threshold.json"
ONSET = "results/bypass-onset.json"
CLOSURE = "clip-k-omega-gamma"
PLATE = "jhtdb-transitional-bl"
GRID = np.geomspace(0.02, 0.4, 25)
MIN_WINS = 3


def main():
    warnings.simplefilter("ignore")
    spec = registry.closures()[CLOSURE]
    cases = registry.cases()
    plate = cases[PLATE].build()
    scan = []
    for a in GRID:
        sc = plate.evaluate(spec.build(case=plate, param="amp", Lam_c=a))
        scan.append({"threshold": float(a),
                     "normalized": sc.get("normalized")})
        print(f"  amp threshold {a:.4f}: plate {sc.get('normalized')}")
    finite = [r for r in scan if r["normalized"] is not None
              and np.isfinite(r["normalized"])]
    best = min(finite, key=lambda r: r["normalized"])
    a = best["threshold"]
    rev_plate = plate.evaluate(spec.build(case=plate))
    rows = {}
    for name in sorted(n for n in cases if n.startswith("wu-bypass-tu")):
        case = cases[name].build()
        rev = case.evaluate(spec.build(case=case))
        amp = case.evaluate(spec.build(case=case, param="amp", Lam_c=a))
        rows[name] = {"tu_inlet_percent": case.tu_inlet_percent,
                      "rev": rev, "amp": amp,
                      "rev_normalized": rev.get("normalized"),
                      "amp_normalized": amp.get("normalized")}
        print(f"{name}: Re_v {rev.get('normalized'):.3f}, amplitude "
              f"{amp.get('normalized'):.3f}")
    # Diagnostic: the model's streak amplitude with transition off
    with open(ONSET) as f:
        onset = json.load(f)["cases"]
    growth, dns_onset = {}, []
    for name in sorted(rows):
        case = cases[name].build()
        tag = "WM" + name.split("tu")[-1]
        amps = {}
        for mode, local in (("local", True), ("freestream", False)):
            sol = case.run(spec.build(case=case, param="amp", Lam_c=10.0,
                                      local_liftup=local))
            amps[mode] = np.sqrt(np.maximum(sol["k"], 0.0)).max(axis=0)
        stations = peak_sqrt_k(tag)
        rth_on = onset[tag]["re_theta_onset"]
        rs = [r for r, _ in stations]
        if rth_on is not None and min(rs) <= rth_on <= max(rs):
            dns_onset.append(float(np.interp(rth_on, rs,
                                             [a for _, a in stations])))
        pts = []
        for rth, amp in stations:
            xs = float(np.interp(rth,
                                 case.theta_ref * case.re_theta0, case.x_th))
            if not case.x[0] <= xs <= case.x[-1]:
                continue
            i = int(np.argmin(np.abs(case.x - xs)))
            pts.append({"re_theta": rth, "dns": amp,
                        "model_local": float(amps["local"][i]),
                        "model_freestream": float(amps["freestream"][i])})
        growth[name] = pts
    sol = plate.run(spec.build(case=plate, param="amp", Lam_c=10.0))
    plate_amp_max = float((np.sqrt(np.maximum(sol["k"], 0.0)).max(axis=0)
                           / np.maximum(sol["U"].max(axis=0), 1e-12)).max())
    at415 = {n: next((p for p in pts if p["re_theta"] == 415.0), None)
             for n, pts in growth.items()}
    r_s = np.array([r["rev_normalized"] for r in rows.values()])
    a_s = np.array([r["amp_normalized"] for r in rows.values()])
    wins = int(np.sum(a_s < r_s))
    result = {
        "closure": CLOSURE, "min_wins": MIN_WINS,
        "calibration_scan": scan,
        "amp_threshold": a,
        "plate_amp_normalized": best["normalized"],
        "plate_rev_normalized": rev_plate.get("normalized"),
        "cases": rows, "n_cases": len(rows),
        "mean_rev": float(r_s.mean()), "mean_amp": float(a_s.mean()),
        "n_wins": wins,
        "amp_transfers": bool(a_s.mean() < r_s.mean() and wins >= MIN_WINS),
        "streak_growth": growth,
        "at_re_theta_415": at415,
        "plate_model_amp_max": plate_amp_max,
        "dns_amp_onset": dns_onset,
        "dns_amp_onset_mean": float(np.mean(dns_onset)),
    }
    for key in ("dns", "model_local", "model_freestream"):
        v = [p[key] for p in at415.values() if p is not None]
        result[f"{key}_at_415_min"] = float(min(v))
        result[f"{key}_at_415_max"] = float(max(v))
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"threshold {a:.4f}; plate amp {best['normalized']:.3f} vs Re_v "
          f"{rev_plate.get('normalized'):.3f}; Wu mean Re_v {r_s.mean():.3f}"
          f" amp {a_s.mean():.3f}, wins {wins}/{len(rows)}, transfers "
          f"{result['amp_transfers']}")


if __name__ == "__main__":
    main()
