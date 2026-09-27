#!/usr/bin/env python
"""Does the streak-scaled transition threshold survive being made
pointwise, as a general CFD code needs?

The streak-scaled threshold (results/streak-scaled-threshold.json) scales
the calibrated Re_v threshold by the peak streak amplitude over the
station, which the parabolic solver has but a general CFD code does not.
Make it pointwise:

    Lambda(y) = Re_v(y) (A(y) / A_ref)^m / Lambda_c,   A = sqrt(k_s) / U_e

with A_ref the plate's A, at the DNS onset, where Re_v peaks, and
everything else as in the station-peak test: the streak coefficients of
results/split-streaks.json, m chosen leaving each of Wu et al.'s flows
out.

Test, fixed before any flow was solved with this trigger: it beats the
fixed Re_v threshold on the mean normalized score over Wu et al.'s five
flows and on at least MIN_WINS of them. The station-peak version's
scores are reported beside it.

Outputs
-------
results/streak-local-threshold.json
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from pypkg import closures, registry

OUT = "results/streak-local-threshold.json"
ONSET = "results/bypass-onset.json"
MECHANICS = "results/transition-mechanics.json"
STREAKS = "results/split-streaks.json"
SCALED = "results/threshold-closure.json"
PEAK = "results/streak-scaled-threshold.json"
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
    with open(PEAK) as f:
        peak = json.load(f)
    with open(MECHANICS) as f:
        x_on = json.load(f)["x_transition_onset"]
    plate = cases[PLATE].build()
    sol = plate.run(build(spec, plate, trigger="amp", Lam_c=1e9, **coeffs))
    i_on = int(np.argmin(np.abs(plate.case.x - x_on)))
    u_on = sol["U"][:, i_on]
    rev_on = plate.case.y ** 2 * np.abs(np.gradient(u_on, plate.case.y))
    j = int(np.argmax(rev_on))
    a_ref = float(np.sqrt(max(sol["ks"][j, i_on], 0.0)) / u_on.max())
    rev_plate = plate.evaluate(spec.build(case=plate)).get("normalized")
    m_all, _ = np.polyfit(
        np.log([c["tu_inlet_percent"] for c in onset.values()
                if c["re_v_max_onset"] is not None]),
        np.log([c["re_v_max_onset"] for c in onset.values()
                if c["re_v_max_onset"] is not None]), 1)
    plate_streak = plate.evaluate(build(
        spec, plate, trigger="rev_streak_local", A_ref=a_ref, m=float(-m_all),
        **coeffs)).get("normalized")
    print(f"A_ref {a_ref:.4f}; plate Re_v {rev_plate:.3f}, pointwise "
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
        st = case.evaluate(build(spec, case, trigger="rev_streak_local",
                                 A_ref=a_ref, m=m, **coeffs))
        rows[name] = {"tu_inlet_percent": case.tu_inlet_percent,
                      "m_left_out": m, "rev": rev, "local": st,
                      "rev_normalized": rev.get("normalized"),
                      "local_normalized": st.get("normalized"),
                      "inlet_scaled_normalized":
                          scaled["cases"][name]["scaled_normalized"],
                      "peak_normalized":
                          peak["cases"][name]["streak_scaled_normalized"]}
        print(name, round(m, 3), round(rev.get("normalized"), 3),
              round(st.get("normalized"), 3),
              round(rows[name]["inlet_scaled_normalized"], 3), flush=True)
    r_s = np.array([r["rev_normalized"] for r in rows.values()])
    s_s = np.array([r["local_normalized"] for r in rows.values()])
    i_s = np.array([r["inlet_scaled_normalized"] for r in rows.values()])
    wins = int(np.sum(s_s < r_s))
    unseen = [rows[n] for n in UNSEEN]
    result = {
        "streak_coeffs": coeffs, "A_ref": a_ref,
        "plate_rev_normalized": rev_plate,
        "plate_local_normalized": plate_streak,
        "cases": rows, "n_cases": len(rows), "min_wins": MIN_WINS,
        "mean_rev": float(r_s.mean()),
        "mean_local": float(s_s.mean()),
        "mean_peak": peak["mean_streak_scaled"],
        "plate_peak": peak["plate_streak_scaled_normalized"],
        "unseen_mean_peak": peak["unseen_mean_streak_scaled"],
        "mean_inlet_scaled": float(i_s.mean()),
        "n_wins": wins,
        "n_wins_vs_inlet_scaled": int(np.sum(s_s < i_s)),
        "unseen_mean_rev": float(np.mean([r["rev_normalized"]
                                          for r in unseen])),
        "unseen_mean_local": float(np.mean(
            [r["local_normalized"] for r in unseen])),
        "unseen_wins": int(sum(r["local_normalized"]
                               < r["rev_normalized"] for r in unseen)),
        "passes": bool(s_s.mean() < r_s.mean() and wins >= MIN_WINS),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items() if k != "cases"})


if __name__ == "__main__":
    main()
