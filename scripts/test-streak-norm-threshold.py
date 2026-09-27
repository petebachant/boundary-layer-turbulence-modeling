#!/usr/bin/env python
"""Does normalizing the streak amplitude by the distance it has grown
over recover the plate without losing the transfer?

Scaling the Re_v threshold by the closure's own streak amplitude A
transfers across Wu et al.'s flows but costs the plate
(results/streak-scaled-threshold.json): A grows along a plate, where the
inlet intensity it stands in for is fixed. Lift-up grows streaks like
Tu Re_x^(1/2) (in the DNS, A / Re_theta per unit inlet Tu stays within
about 6-10 x 10^-3 across Wu et al.'s flows and stations), and Re_v,max
grows like Re_x^(1/2) in a laminar layer, so the ratio

    T = A / Re_v,max

measures the intensity with the distance taken out. Replace A with T:

    Lambda = Re_v (T / T_ref)^m / Lambda_c

T_ref is the plate's T at the DNS onset; m is chosen leaving each of Wu
et al.'s flows out, exactly as before; the streak coefficients are those
of results/split-streaks.json.

Test, fixed before any flow was solved with this trigger: it beats the
fixed Re_v threshold on the mean normalized score over the five flows and
on at least MIN_WINS of them, and beats the unnormalized streak-scaled
threshold both on the plate and on the mean over the five.

Outputs
-------
results/streak-norm-threshold.json
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from pypkg import closures, registry

OUT = "results/streak-norm-threshold.json"
ONSET = "results/bypass-onset.json"
MECHANICS = "results/transition-mechanics.json"
STREAKS = "results/split-streaks.json"
UNNORM = "results/streak-scaled-threshold.json"
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
    with open(UNNORM) as f:
        unnorm = json.load(f)
    with open(MECHANICS) as f:
        x_on = json.load(f)["x_transition_onset"]
    plate = cases[PLATE].build()
    sol = plate.run(build(spec, plate, trigger="amp", Lam_c=1e9, **coeffs))
    amp = (np.sqrt(np.maximum(sol["ks"], 0.0)).max(axis=0)
           / np.maximum(sol["U"].max(axis=0), 1e-12))
    dudy = np.abs(np.gradient(sol["U"], plate.case.y, axis=0))
    rev_max = (plate.case.y[:, None] ** 2 * dudy / plate.case.nu).max(axis=0)
    a_ref = float(np.interp(x_on, plate.case.x, amp / rev_max))
    rev_plate = plate.evaluate(spec.build(case=plate)).get("normalized")
    m_all, _ = np.polyfit(
        np.log([c["tu_inlet_percent"] for c in onset.values()
                if c["re_v_max_onset"] is not None]),
        np.log([c["re_v_max_onset"] for c in onset.values()
                if c["re_v_max_onset"] is not None]), 1)
    plate_streak = plate.evaluate(build(
        spec, plate, trigger="rev_streak_norm", A_ref=a_ref, m=float(-m_all),
        **coeffs)).get("normalized")
    print(f"T_ref {a_ref:.3g}; plate Re_v {rev_plate:.3f}, normalized "
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
        st = case.evaluate(build(spec, case, trigger="rev_streak_norm",
                                 A_ref=a_ref, m=m, **coeffs))
        rows[name] = {"tu_inlet_percent": case.tu_inlet_percent,
                      "m_left_out": m, "rev": rev, "norm": st,
                      "rev_normalized": rev.get("normalized"),
                      "norm_normalized": st.get("normalized"),
                      "inlet_scaled_normalized":
                          scaled["cases"][name]["scaled_normalized"],
                      "unnormalized_normalized":
                          unnorm["cases"][name]["streak_scaled_normalized"]}
        print(name, round(m, 3), round(rev.get("normalized"), 3),
              round(st.get("normalized"), 3),
              round(rows[name]["inlet_scaled_normalized"], 3), flush=True)
    r_s = np.array([r["rev_normalized"] for r in rows.values()])
    s_s = np.array([r["norm_normalized"] for r in rows.values()])
    i_s = np.array([r["inlet_scaled_normalized"] for r in rows.values()])
    wins = int(np.sum(s_s < r_s))
    unseen = [rows[n] for n in UNSEEN]
    result = {
        "streak_coeffs": coeffs, "T_ref": a_ref,
        "plate_rev_normalized": rev_plate,
        "plate_norm_normalized": plate_streak,
        "cases": rows, "n_cases": len(rows), "min_wins": MIN_WINS,
        "mean_rev": float(r_s.mean()),
        "mean_norm": float(s_s.mean()),
        "mean_inlet_scaled": float(i_s.mean()),
        "n_wins": wins,
        "n_wins_vs_inlet_scaled": int(np.sum(s_s < i_s)),
        "unseen_mean_rev": float(np.mean([r["rev_normalized"]
                                          for r in unseen])),
        "unseen_mean_norm": float(np.mean(
            [r["norm_normalized"] for r in unseen])),
        "unseen_wins": int(sum(r["norm_normalized"]
                               < r["rev_normalized"] for r in unseen)),
        "mean_unnormalized": unnorm["mean_streak_scaled"],
        "plate_unnormalized": unnorm["plate_streak_scaled_normalized"],
        "unseen_mean_unnormalized": unnorm["unseen_mean_streak_scaled"],
        "passes": bool(s_s.mean() < r_s.mean() and wins >= MIN_WINS
                       and plate_streak < unnorm["plate_streak_scaled_normalized"]
                       and s_s.mean() < unnorm["mean_streak_scaled"]),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items() if k != "cases"})


if __name__ == "__main__":
    main()
