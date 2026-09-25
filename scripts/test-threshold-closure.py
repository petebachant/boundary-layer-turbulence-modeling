#!/usr/bin/env python
"""Does a free-stream-dependent threshold make the clipping closure transfer
across free-stream intensity?

clip-k-omega-gamma was calibrated on the JHTDB plate, at one intensity, with
a fixed threshold Lambda_c. Onset data at five intensities (Wu et al.) say
the threshold falls as the intensity rises, roughly as a power of the inlet
intensity (results/threshold-law.json). So scale the calibrated threshold,

    Lambda_c(Tu_in) = Lambda_c,JHTDB (Tu_in / Tu_in,JHTDB)^(-m),

which leaves the JHTDB plate exactly as calibrated and adds one number, m,
and fit m leaving each of Wu et al.'s intensities out in turn: for each
case, m comes from the power law of onset Re_v against inlet intensity over
the other cases (results/bypass-onset.json), and the case is then solved
with the scaled threshold. The 0.75 percent case has no onset Re_v inside
its profiles, so it never contributes to a fit but is still tested.

Test, fixed before any case was solved with the scaled threshold: the
scaled closure beats the fixed one on the mean normalized score over the
five cases, and on at least MIN_WINS of them.

Outputs
-------
results/threshold-closure.json
"""

from __future__ import annotations

import json
import time
import warnings

import numpy as np

from pypkg import registry

ONSET = "results/bypass-onset.json"
INLET = "results/inlet-profiles.json"
OUT = "results/threshold-closure.json"
CLOSURE = "clip-k-omega-gamma"
MIN_WINS = 3


def main():
    with open(ONSET) as f:
        onset = json.load(f)["cases"]
    with open(INLET) as f:
        tu_jhtdb = json.load(f)["Tu_inlet_percent"]
    spec = registry.closures()[CLOSURE]
    lam_jhtdb = float(spec.get_coeffs().get("Lam_c", 440.0))
    cases = {n: c for n, c in registry.cases().items()
             if n.startswith("wu-bypass-tu")}
    rows = {}
    for name, cspec in sorted(cases.items()):
        case = cspec.build()
        tag = "WM" + name.split("tu")[-1]
        train = [c for t, c in onset.items()
                 if t != tag and c["re_v_max_onset"] is not None]
        slope, _ = np.polyfit(np.log([c["tu_inlet_percent"] for c in train]),
                              np.log([c["re_v_max_onset"] for c in train]), 1)
        m = float(-slope)
        lam = lam_jhtdb * (case.tu_inlet_percent / tu_jhtdb) ** (-m)
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fixed = case.evaluate(spec.build(case=case))
            scaled = case.evaluate(spec.build(case=case, Lam_c=lam))
        rows[name] = {
            "tu_inlet_percent": case.tu_inlet_percent,
            "m_left_out": m, "n_train": len(train),
            "lam_c_fixed": lam_jhtdb, "lam_c_scaled": lam,
            "fixed": fixed, "scaled": scaled,
            "fixed_normalized": fixed.get("normalized"),
            "scaled_normalized": scaled.get("normalized"),
            "wall_time_s": round(time.time() - t0, 2),
        }
        print(f"{name}: Tu {case.tu_inlet_percent}%, m {m:.3f}, Lambda_c "
              f"{lam_jhtdb:.0f} -> {lam:.0f}; score fixed "
              f"{fixed.get('normalized'):.3f}, scaled "
              f"{scaled.get('normalized'):.3f}")
    f_s = np.array([r["fixed_normalized"] for r in rows.values()])
    s_s = np.array([r["scaled_normalized"] for r in rows.values()])
    wins = int(np.sum(s_s < f_s))
    result = {
        "closure": CLOSURE, "tu_inlet_jhtdb_percent": tu_jhtdb,
        "lam_c_jhtdb": lam_jhtdb, "min_wins": MIN_WINS,
        "cases": rows, "n_cases": len(rows),
        "mean_fixed": float(f_s.mean()), "mean_scaled": float(s_s.mean()),
        "n_wins": wins,
        "scaled_transfers": bool(s_s.mean() < f_s.mean()
                                 and wins >= MIN_WINS),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"mean fixed {f_s.mean():.3f}, scaled {s_s.mean():.3f}, wins "
          f"{wins}/{len(rows)}, transfers {result['scaled_transfers']}")


if __name__ == "__main__":
    main()
