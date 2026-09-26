#!/usr/bin/env python
"""Can the closure's own streaks stand in for the inlet intensity that the
inlet-scaled transition threshold has to be told?

Scaling the clipping closure's Re_v threshold with the inlet intensity,
Lambda_c (Tu_in / Tu_in,JHTDB)^(-m), transfers across Wu et al.'s five
flows (results/threshold-closure.json), but Tu_in is not a local quantity.
A threshold on the streak amplitude alone fails even on the plate
(results/split-streaks.json): Re_v places transition well, the amplitude
does not. The streaks do carry the free stream's history, and they grow
with its intensity. So keep Re_v as the trigger and put the closure's own
streak amplitude where the inlet intensity was:

    Lambda = Re_v (A / A_ref)^m / Lambda_c

with A the peak of sqrt(k_s)/U_e over the layer at the station, from
SplitStreakKOmegaGamma with the streak coefficients of
results/split-streaks.json, A_ref the plate's A at the DNS onset (so the
plate is scaled by about one where it transitions), and Lambda_c the
calibrated threshold. m is chosen exactly as in the inlet-scaled test:
for each of Wu et al.'s flows, the power-law exponent of onset Re_v
against inlet intensity over the others (results/bypass-onset.json).

Test, fixed before any of Wu et al.'s flows was solved with this trigger
(the plate was checked first): it beats the fixed Re_v threshold on the
mean normalized score over the five flows and on at least MIN_WINS of
them. The inlet-scaled closure's scores are reported beside it. The
0.75 and 1.5 percent flows' pre-onset streaks were in the streak fit.

Outputs
-------
results/streak-scaled-threshold.json
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from pypkg import closures, registry

OUT = "results/streak-scaled-threshold.json"
ONSET = "results/bypass-onset.json"
MECHANICS = "results/transition-mechanics.json"
STREAKS = "results/split-streaks.json"
SCALED = "results/threshold-closure.json"
BASE = "clip-k-omega-gamma"
PLATE = "jhtdb-transitional-bl"
UNSEEN = ("wu-bypass-tu225", "wu-bypass-tu300", "wu-bypass-tu600")
MIN_WINS = 3


def build(spec, case, **kw):
    a = spec.get_coeffs()
    a.update(case.closure_kwargs(spec))
    a.update(kw)
    return closures.SplitStreakKOmegaGamma(**a)


def main():
    warnings.simplefilter("ignore")
    spec = registry.closures()[BASE]
    cases = registry.cases()
    with open(STREAKS) as f:
        coeffs = json.load(f)["streak_coeffs"]
    with open(ONSET) as f:
        onset = json.load(f)["cases"]
    with open(SCALED) as f:
        scaled = json.load(f)
    with open(MECHANICS) as f:
        x_on = json.load(f)["x_transition_onset"]
    plate = cases[PLATE].build()
    sol = plate.run(build(spec, plate, trigger="amp", Lam_c=1e9, **coeffs))
    amp = (np.sqrt(np.maximum(sol["ks"], 0.0)).max(axis=0)
           / np.maximum(sol["U"].max(axis=0), 1e-12))
    a_ref = float(np.interp(x_on, plate.case.x, amp))
    rev_plate = plate.evaluate(spec.build(case=plate)).get("normalized")
    m_all, _ = np.polyfit(
        np.log([c["tu_inlet_percent"] for c in onset.values()
                if c["re_v_max_onset"] is not None]),
        np.log([c["re_v_max_onset"] for c in onset.values()
                if c["re_v_max_onset"] is not None]), 1)
    plate_streak = plate.evaluate(build(
        spec, plate, trigger="rev_streak", A_ref=a_ref, m=float(-m_all),
        **coeffs)).get("normalized")
    print(f"A_ref {a_ref:.4f}; plate Re_v {rev_plate:.3f}, streak-scaled "
          f"(m over all five) {plate_streak:.3f}", flush=True)
    rows = {}
    for name in sorted(n for n in cases if n.startswith("wu-bypass-tu")):
        case = cases[name].build()
        tag = "WM" + name.split("tu")[-1]
        train = [c for t, c in onset.items()
                 if t != tag and c["re_v_max_onset"] is not None]
        slope, _ = np.polyfit(np.log([c["tu_inlet_percent"] for c in train]),
                              np.log([c["re_v_max_onset"] for c in train]), 1)
        m = float(-slope)
        rev = case.evaluate(spec.build(case=case))
        st = case.evaluate(build(spec, case, trigger="rev_streak",
                                 A_ref=a_ref, m=m, **coeffs))
        rows[name] = {"tu_inlet_percent": case.tu_inlet_percent,
                      "m_left_out": m, "rev": rev, "streak_scaled": st,
                      "rev_normalized": rev.get("normalized"),
                      "streak_scaled_normalized": st.get("normalized"),
                      "inlet_scaled_normalized":
                          scaled["cases"][name]["scaled_normalized"]}
        print(name, round(m, 3), round(rev.get("normalized"), 3),
              round(st.get("normalized"), 3),
              round(rows[name]["inlet_scaled_normalized"], 3), flush=True)
    r_s = np.array([r["rev_normalized"] for r in rows.values()])
    s_s = np.array([r["streak_scaled_normalized"] for r in rows.values()])
    i_s = np.array([r["inlet_scaled_normalized"] for r in rows.values()])
    wins = int(np.sum(s_s < r_s))
    unseen = [rows[n] for n in UNSEEN]
    result = {
        "streak_coeffs": coeffs, "A_ref": a_ref,
        "plate_rev_normalized": rev_plate,
        "plate_streak_scaled_normalized": plate_streak,
        "cases": rows, "n_cases": len(rows), "min_wins": MIN_WINS,
        "mean_rev": float(r_s.mean()),
        "mean_streak_scaled": float(s_s.mean()),
        "mean_inlet_scaled": float(i_s.mean()),
        "n_wins": wins,
        "n_wins_vs_inlet_scaled": int(np.sum(s_s < i_s)),
        "unseen_mean_rev": float(np.mean([r["rev_normalized"]
                                          for r in unseen])),
        "unseen_mean_streak_scaled": float(np.mean(
            [r["streak_scaled_normalized"] for r in unseen])),
        "unseen_wins": int(sum(r["streak_scaled_normalized"]
                               < r["rev_normalized"] for r in unseen)),
        "passes": bool(s_s.mean() < r_s.mean() and wins >= MIN_WINS),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items() if k != "cases"})


if __name__ == "__main__":
    main()
