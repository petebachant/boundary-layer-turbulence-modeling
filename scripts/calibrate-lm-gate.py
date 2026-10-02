#!/usr/bin/env python
"""Can Langtry-Menter transition the JHTDB plate once only its onset
correlation can start transition?

On Wu et al.'s flows Langtry-Menter's transition is started by its
eddy-viscosity route, R_T reaching 2.5 in the layer, before Re_v reaches
the correlation's critical value (results/lm-onset-gate.json). The gated
variant (kOmegaSSTLMGate with rtOnset off, sim/run_lm_gate.py) closes that
route on top of the leading-edge variant: streak correlation, no diffusion
of ReThetat, C / Tu_in at the inlet. As for the other variants, C is
chosen by running the model on the plate over a grid and keeping the one
with the lowest mean C_f error over the sampled stations.

Trial runs at four values of C, made before this stage, showed no value
bringing the plate near standard Langtry-Menter, so the grid is those
four, and Wu et al.'s flows are run only if the best plate C_f error is
within PROCEED times standard Langtry-Menter's. That threshold was set
after the trials, and is reported as such.

Outputs
-------
results/lm-gate-calibration.json
"""

from __future__ import annotations

import importlib.util
import json

GRID = (250, 330, 430, 610)
PROCEED = 1.5
OUT = "results/lm-gate-calibration.json"


def main():
    spec = importlib.util.spec_from_file_location(
        "t", "scripts/test-lm-streak.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    scan = []
    for c in GRID:
        err, rows = mod.plate_cf_error(f"sim/cases/lm-gate-plate-C{c}")
        ratios = [r["cf"] / r["cf_dns"] for r in rows]
        scan.append(
            {
                "c_streak": c,
                "plate_cf_err": err,
                "cf_ratio_max": max(ratios),
                "cf_ratio_min_downstream": min(ratios[len(ratios) // 2 :]),
            }
        )
        print(c, round(err, 4))
    lm, _ = mod.plate_cf_error(mod.PLATE_LM)
    with open("results/lm-le-calibration.json") as f:
        le = json.load(f)
    best = min(scan, key=lambda r: r["plate_cf_err"])
    out = {
        "grid": list(GRID),
        "scan": scan,
        "c_streak": best["c_streak"],
        "plate_cf_err": best["plate_cf_err"],
        "plate_cf_err_lm": lm,
        "plate_cf_err_le": le["plate_cf_err"],
        "error_over_lm": best["plate_cf_err"] / lm,
        "proceed_factor": PROCEED,
        "proceed": bool(best["plate_cf_err"] <= PROCEED * lm),
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(out)


if __name__ == "__main__":
    main()
