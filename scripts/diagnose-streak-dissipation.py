#!/usr/bin/env python
"""Why does the two-component streak closure spoil the plate once
transition is on?

The streak-threshold closure scores far worse on the JHTDB plate than the
clipping closure it is built on (results/streak-threshold.json). Solve the
plate with the calibrated Re_v threshold, so the threshold is not in
question, changing one thing at a time on the way from the calibrated
closure to the streak closure's coefficients:

  calibrated          the clipping closure as calibrated
  no_own_liftup       its own lift-up off (C_L = 0), as in the streak closure
  streak              the streak closure's coefficients (C_f, C_d, C_v, C_b)
  streak_calib_cd     the same, with C_d back at its calibrated value

C_d suppresses dissipation where the flow has not activated, (1 - gamma)
S / omega large. The streak fit wants it large, streaks that barely
dissipate; if the turbulent layer needs it small, one k is being asked to
be two reservoirs.

Outputs
-------
results/streak-dissipation.json
"""

from __future__ import annotations

import json
import warnings

from pypkg import closures, registry

OUT = "results/streak-dissipation.json"
COEFFS = "results/streak-threshold.json"
BASE = "clip-k-omega-gamma"
PLATE = "jhtdb-transitional-bl"


def main():
    warnings.simplefilter("ignore")
    spec = registry.closures()[BASE]
    plate = registry.cases()[PLATE].build()
    with open(COEFFS) as f:
        streak = json.load(f)["streak_coeffs"]
    cd_calib = float(spec.get_coeffs()["Cd"])
    off = {"CL": 0.0, "local_liftup": False}
    configs = {
        "calibrated": (closures.ClipKOmegaGamma, {}),
        "no_own_liftup": (closures.ClipKOmegaGamma, off),
        "streak": (closures.StreakKOmegaGamma, {**off, **streak}),
        "streak_calib_cd": (closures.StreakKOmegaGamma,
                            {**off, **streak, "Cd": cd_calib}),
    }
    rows = {}
    for name, (cls, kw) in configs.items():
        a = spec.get_coeffs()
        a.update(plate.closure_kwargs(spec))
        a.update(kw)
        sc = plate.evaluate(cls(**a))
        rows[name] = {"normalized": sc.get("normalized"),
                      "cf_rel_rms": sc.get("cf_rel_rms"),
                      "theta_rel_rms": sc.get("theta_rel_rms")}
        print(name, rows[name])
    result = {"cd_calibrated": cd_calib, "cd_streak": streak["Cd"],
              "configs": rows}
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
