#!/usr/bin/env python
"""With the streaks carried apart from the turbulence, does a transition
threshold on streak amplitude predict bypass transition across free-stream
intensity without being told the inlet?

The two-component closure's streaks grow as the DNS's do, but with
transition on it spoils the plate: the streaks need dissipation suppressed
strongly and the turbulent layer hardly at all, and one k cannot be both
(results/streak-dissipation.json). SplitStreakKOmegaGamma keeps the
turbulence exactly as calibrated and carries the forcing k_v and the
streaks k_s as two more energies beside it; the threshold reads the
streaks.

  1. streak coefficients (Cf_s, Cds, Cv, Cb) chosen on GRID by the mean
     error in peak sqrt(k_s)/U_e before onset against the DNS's peak
     sqrt(k), over the three flows with DNS streaks there (the JHTDB plate,
     Wu et al.'s 0.75 and 1.5 percent), with transition off
  2. the streak-amplitude threshold, and a factor on the activation rate
     C_gam (calibrated for the Re_v threshold, whose overshoot is of order
     one where an amplitude's is a few percent), chosen on the plate alone
     by its normalized score over AMP_GRID and CGAM_FACTORS
  3. all five of Wu et al.'s flows scored with it, against the clipping
     closure with its calibrated Re_v threshold

Test, fixed before any of Wu et al.'s flows was solved with this closure
(a check on the plate alone came first): the split closure beats the Re_v
closure on the mean normalized score over the five flows and on at least
MIN_WINS of them. The three flows whose streaks were not in step 1 (2.25,
3 and 6 percent) are reported on their own too.

The first run of this stage was invalid: the closure took param="Rev"
from the calibrated coefficients, so its "amplitude" threshold was a Re_v
threshold of order 0.1 and every flow turned turbulent at its inlet. The
closure now always reads the streaks.

Outputs
-------
results/split-streaks.json
"""

from __future__ import annotations

import itertools
import json
import warnings

import numpy as np

from pypkg import closures, registry
from pypkg.cases.wu_bypass import peak_sqrt_k
from pypkg.dns_case import load_dns

OUT = "results/split-streaks.json"
ONSET = "results/bypass-onset.json"
MECHANICS = "results/transition-mechanics.json"
BASE = "clip-k-omega-gamma"
PLATE = "jhtdb-transitional-bl"
STREAK_FLOWS = (PLATE, "wu-bypass-tu075", "wu-bypass-tu150")
UNSEEN = ("wu-bypass-tu225", "wu-bypass-tu300", "wu-bypass-tu600")
GRID = {"Cf_s": [0.03, 0.1, 0.3, 1.0],
        "Cds": [30.0, 150.0, 750.0, 3750.0],
        "Cv": [0.3, 1.0, 3.0],
        "Cb": [3.0, 10.0, 30.0]}
AMP_GRID = np.geomspace(0.03, 0.2, 15)
CGAM_FACTORS = [1.0, 3.0, 10.0, 30.0]
MIN_WINS = 3
STATIONS = 8


def build(spec, case, **kw):
    a = spec.get_coeffs()
    a.update(case.closure_kwargs(spec))
    a.update(kw)
    return closures.SplitStreakKOmegaGamma(**a)


def targets(cases):
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
    for name in STREAK_FLOWS[1:]:
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


def streak_error(spec, flow, **kw):
    case, cx, xs, dns = flow
    sol = case.run(build(spec, case, Lam_c=1e9, **kw))
    amp = (np.sqrt(np.maximum(sol["ks"], 0.0)).max(axis=0)
           / np.maximum(sol["U"].max(axis=0), 1e-12))
    m = np.maximum(np.interp(xs, cx, amp), 1e-12)
    return float(np.sqrt(np.mean(np.log(m / dns) ** 2)))


def main():
    warnings.simplefilter("ignore")
    spec = registry.closures()[BASE]
    cases = registry.cases()
    flows = targets(cases)
    cgam = float(spec.get_coeffs()["Cgam"])
    # 1. streak coefficients, transition off
    rows = []
    keys = list(GRID)
    for vals in itertools.product(*(GRID[k] for k in keys)):
        kw = dict(zip(keys, vals))
        errs = {n: streak_error(spec, f, **kw) for n, f in flows.items()}
        rows.append({"coeffs": kw, "errors": errs,
                     "mean": float(np.mean(list(errs.values())))})
        print(" ", kw, round(rows[-1]["mean"], 3), flush=True)
    best = min(rows, key=lambda r: r["mean"])
    coeffs = best["coeffs"]
    print("streak coefficients", coeffs, best["errors"])
    # 2. threshold and activation rate, on the plate alone
    plate = flows[PLATE][0]
    scan = []
    for fac in CGAM_FACTORS:
        for a in AMP_GRID:
            sc = plate.evaluate(build(spec, plate, Lam_c=float(a),
                                      Cgam=cgam * fac, **coeffs))
            scan.append({"threshold": float(a), "cgam_factor": fac,
                         "normalized": sc.get("normalized")})
            print(f"  Cgam x{fac:g}, threshold {a:.4f}: plate "
                  f"{sc.get('normalized')}", flush=True)
    ok = [r for r in scan if r["normalized"] is not None
          and np.isfinite(r["normalized"])]
    cal = min(ok, key=lambda r: r["normalized"])
    tuned = dict(coeffs, Lam_c=cal["threshold"],
                 Cgam=cgam * cal["cgam_factor"])
    rev_plate = plate.evaluate(spec.build(case=plate)).get("normalized")
    # 3. Wu et al.'s flows
    out = {}
    for name in sorted(n for n in cases if n.startswith("wu-bypass-tu")):
        case = cases[name].build()
        rev = case.evaluate(spec.build(case=case))
        st = case.evaluate(build(spec, case, **tuned))
        out[name] = {"tu_inlet_percent": case.tu_inlet_percent,
                     "rev": rev, "split": st,
                     "rev_normalized": rev.get("normalized"),
                     "split_normalized": st.get("normalized")}
        print(name, round(rev.get("normalized"), 3),
              round(st.get("normalized"), 3), flush=True)
    r_s = np.array([c["rev_normalized"] for c in out.values()])
    s_s = np.array([c["split_normalized"] for c in out.values()])
    wins = int(np.sum(s_s < r_s))
    unseen_r = np.array([out[n]["rev_normalized"] for n in UNSEEN])
    unseen_s = np.array([out[n]["split_normalized"] for n in UNSEEN])
    with open("results/threshold-closure.json") as f:
        scaled_mean = json.load(f)["mean_scaled"]
    no_transition = max(r["normalized"] for r in ok)
    result = {
        "grid": GRID, "streak_table": rows,
        "streak_coeffs": coeffs, "streak_errors": best["errors"],
        "streak_mean_error": best["mean"],
        "plate_scan": scan, "amp_threshold": cal["threshold"],
        "cgam_factor": cal["cgam_factor"],
        "plate_split_normalized": cal["normalized"],
        "plate_rev_normalized": rev_plate,
        "plate_worst_in_scan": no_transition,
        "cases": out, "n_cases": len(out), "min_wins": MIN_WINS,
        "mean_rev": float(r_s.mean()), "mean_split": float(s_s.mean()),
        "n_wins": wins,
        "unseen": list(UNSEEN),
        "unseen_mean_rev": float(unseen_r.mean()),
        "unseen_mean_split": float(unseen_s.mean()),
        "unseen_wins": int(np.sum(unseen_s < unseen_r)),
        "inlet_scaled_mean": scaled_mean,
        "passes": bool(s_s.mean() < r_s.mean() and wins >= MIN_WINS),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items()
           if k not in ("streak_table", "plate_scan", "cases", "grid")})


if __name__ == "__main__":
    main()
