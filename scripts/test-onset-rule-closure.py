#!/usr/bin/env python
"""Does the clipping closure transfer across free-stream intensity with its
transition threshold set by the streak onset rule rather than fitted?

Applied to the DNS, onset at Re_theta = C / Tu_in, with C from the JHTDB
plate, predicts Wu et al.'s onsets and Bienner et al.'s
(results/onset-history.json). The clipping closure activates where
Re_v = y^2 |dU/dy| / nu exceeds Lambda_c, and in a laminar layer the peak
Re_v is 2.193 Re_theta, so the rule is a threshold inversely proportional
to the inlet intensity. Two forms, with nothing fitted on Wu et al.'s
flows:

  A  Lambda_c = Lambda_c,JHTDB (Tu_in / Tu_in,JHTDB)^(-1): the calibrated
     plate threshold scaled with the rule's exponent, 1, in place of the
     exponent results/threshold-closure.json fits on Wu et al.'s onsets
  B  Lambda_c = 2.193 C / Tu_in with C from the DNS, no calibration at all

In the marching solver the inlet intensity is the leading edge's, so this
is the threshold a closure carrying the leading-edge intensity as an
advected scalar would apply on these flows.

Test, fixed before any case was solved with either form: form A beats the
fixed threshold on the mean normalized score over Wu et al.'s five flows
and on at least MIN_WINS of them, the test the fitted exponent took. Also
reported: form B, and both against the fitted exponent's scores.

Outputs
-------
results/onset-rule-closure.json
"""

from __future__ import annotations

import json
import warnings

import numpy as np

from pypkg import registry

#: The plate's leading-edge intensity (scripts/plate-leading-edge-intensity.py),
#: rather than the free stream's at x = 30, where the database starts and
#: which this read until 2026-10-06
INLET = "results/plate-leading-edge-intensity.json"
FITTED = "results/threshold-closure.json"
HISTORY = "results/onset-history.json"
OUT = "results/onset-rule-closure.json"
CLOSURE = "clip-k-omega-gamma"
MIN_WINS = 3
REV_OVER_RETHETA = 2.193


def main():
    with open(INLET) as f:
        tu_jhtdb = json.load(f)["tu_le_percent"]
    with open(FITTED) as f:
        fitted = json.load(f)
    with open(HISTORY) as f:
        c_rule = json.load(f)["c_inlet"]
    spec = registry.closures()[CLOSURE]
    lam_jhtdb = float(spec.get_coeffs().get("Lam_c", 440.0))
    cases = {
        n: c
        for n, c in registry.cases().items()
        if n.startswith("wu-bypass-tu")
    }
    rows = {}
    for name, cspec in sorted(cases.items()):
        case = cspec.build()
        tu = case.tu_inlet_percent
        lam_a = lam_jhtdb * tu_jhtdb / tu
        lam_b = REV_OVER_RETHETA * c_rule / tu
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = case.evaluate(spec.build(case=case, Lam_c=lam_a))
            b = case.evaluate(spec.build(case=case, Lam_c=lam_b))
        prior = fitted["cases"][name]
        rows[name] = {
            "tu_inlet_percent": tu,
            "lam_c_a": lam_a,
            "lam_c_b": lam_b,
            "lam_c_fitted": prior["lam_c_scaled"],
            "a": a,
            "b": b,
            "a_normalized": a.get("normalized"),
            "b_normalized": b.get("normalized"),
            "fixed_normalized": prior["fixed_normalized"],
            "fitted_normalized": prior["scaled_normalized"],
        }
        print(
            f"{name}: Lambda_c A {lam_a:.0f}, B {lam_b:.0f}, fitted "
            f"{prior['lam_c_scaled']:.0f}; score A "
            f"{a.get('normalized'):.3f}, B {b.get('normalized'):.3f}, "
            f"fixed {prior['fixed_normalized']:.3f}, fitted "
            f"{prior['scaled_normalized']:.3f}"
        )
    col = lambda k: np.array([r[k] for r in rows.values()])  # noqa: E731
    a, b = col("a_normalized"), col("b_normalized")
    fixed, fit = col("fixed_normalized"), col("fitted_normalized")
    wins = int(np.sum(a < fixed))
    result = {
        "closure": CLOSURE,
        "tu_inlet_jhtdb_percent": tu_jhtdb,
        "lam_c_jhtdb": lam_jhtdb,
        "c_rule": c_rule,
        "lam_c_b_jhtdb": REV_OVER_RETHETA * c_rule / tu_jhtdb,
        "min_wins": MIN_WINS,
        "cases": rows,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_fixed": float(fixed.mean()),
        "mean_fitted": float(fit.mean()),
        "n_wins": wins,
        "n_wins_b": int(np.sum(b < fixed)),
        "n_wins_a_vs_fitted": int(np.sum(a < fit)),
        "passes": bool(a.mean() < fixed.mean() and wins >= MIN_WINS),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items() if k != "cases"})


if __name__ == "__main__":
    main()
