#!/usr/bin/env python
"""Fit the momentum-term library's coefficients against gym cases by
Bayesian optimization, a posteriori.

The library (pypkg/momentum_library.py) adds streamwise forces beyond the
eddy-viscosity stress to a base closure. Its coefficients are found here by
running the resulting closure on the chosen cases and minimizing the mean
normalized benchmark score, so the objective is the same number the
leaderboard reports and a coefficient set is judged by the flow it produces,
not by how well it fits a stress a priori. A diverged or non-converged run is
a legitimate answer -- that coefficient set is unstable -- and the optimizer
is told so rather than having it hidden.

Two things are reported besides the optimum, because the optimum alone is
the least useful number here (ideas-log 7.2):

* the score of the fitted closure on every case, in- and out-of-sample, so
  the transfer penalty is visible before the closure enters the leaderboard;
* for each coefficient, the range it takes over every evaluation within a
  tolerance of the best score -- a cheap identifiability interval. A
  coefficient whose interval spans its bounds is not determined by the fit.

Outputs
-------
results/momentum-library.json (or --out)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from pypkg import registry  # noqa: E402
from pypkg.bayesopt import bayes_opt  # noqa: E402
from pypkg.momentum_library import (  # noqa: E402
    DEFAULT_BOUNDS,
    TERM_NAMES,
    MomentumLibraryClosure,
)


def score_on(case, coeffs, cap):
    closure = MomentumLibraryClosure(cap=cap, **coeffs, **case.closure_kwargs())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sc = case.evaluate(closure)
    n = sc.get("normalized")
    return sc, (float(n) if n is not None and np.isfinite(n) else float("inf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit-cases", default="jhtdb-transitional-bl",
                    help="comma-separated cases the coefficients are fitted on")
    ap.add_argument("--n-init", type=int, default=12)
    ap.add_argument("--n-iter", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bound", type=float, default=0.5,
                    help="symmetric coefficient bound for every term")
    ap.add_argument("--cap", type=float, default=5.0,
                    help="cap on |library force| relative to the eddy-"
                         "viscosity term; None to disable")
    ap.add_argument("--near-best", type=float, default=0.10,
                    help="relative tolerance defining the near-best set "
                         "for the identifiability intervals")
    ap.add_argument("--out", default="results/momentum-library.json")
    args = ap.parse_args()

    fit_names = [s.strip() for s in args.fit_cases.split(",") if s.strip()]
    all_cases = registry.cases()
    fit_cases = {n: all_cases[n].build() for n in fit_names}
    bounds = [(-args.bound, args.bound)] * len(TERM_NAMES)
    cap = None if args.cap <= 0 else args.cap
    history = []

    def objective(x):
        coeffs = dict(zip(TERM_NAMES, (float(v) for v in x)))
        per_case = {}
        total = 0.0
        for name, case in fit_cases.items():
            sc, n = score_on(case, coeffs, cap)
            per_case[name] = n
            total += n
        val = total / len(fit_cases)
        history.append({"coeffs": coeffs, "per_case": per_case,
                        "objective": val if np.isfinite(val) else None})
        return val

    t0 = time.time()

    def log(i, x, val):
        print(f"  eval {i:3d}: {'diverged' if not np.isfinite(val) else f'{val:8.3f}'}"
              f"  best so far {min((h['objective'] for h in history if h['objective'] is not None), default=float('nan')):.3f}")

    # Start from the base closure itself and a few small perturbations of
    # it, so the surrogate has feasible points to build on before the random
    # design explores the corners, where O(1) forces tend to diverge
    rng = np.random.default_rng(args.seed)
    dim = len(TERM_NAMES)
    x0 = [np.zeros(dim)] + [
        rng.normal(0.0, 0.1 * args.bound, dim) for _ in range(5)
    ]
    res = bayes_opt(objective, bounds, n_init=args.n_init, n_iter=args.n_iter,
                    seed=args.seed, callback=log, x0=x0)
    best = dict(zip(TERM_NAMES, (float(v) for v in res["best_x"])))
    print(f"best objective {res['best_y']:.4f} after {len(history)} evaluations "
          f"({res['n_infeasible']} diverged), {time.time() - t0:.0f} s")
    # Identifiability: the range of each coefficient over the near-best set
    finite = [h for h in history if h["objective"] is not None]
    if not finite:
        raise SystemExit("every evaluation diverged; nothing to report")
    thresh = res["best_y"] * (1.0 + args.near_best)
    near = [h for h in finite if h["objective"] <= thresh]
    intervals = {
        name: [min(h["coeffs"][name] for h in near),
               max(h["coeffs"][name] for h in near)]
        for name in TERM_NAMES
    }
    # The fitted closure on every case, so the transfer is visible here
    scores = {}
    base_scores = {}
    for name, spec in sorted(all_cases.items()):
        case = spec.build()
        sc, n = score_on(case, best, cap)
        scores[name] = {**{k: v for k, v in sc.items()
                           if isinstance(v, (int, float, bool))},
                        "in_sample": name in fit_cases}
        sc0, n0 = score_on(case, {k: 0.0 for k in TERM_NAMES}, cap)
        base_scores[name] = n0
        print(f"  {name:30s} {'in ' if name in fit_cases else 'OUT'}"
              f"  library {n:8.3f}   base {n0:8.3f}")
    ins = [scores[n]["normalized"] for n in scores
           if scores[n]["in_sample"] and np.isfinite(scores[n]["normalized"])]
    outs = [scores[n]["normalized"] for n in scores
            if not scores[n]["in_sample"] and np.isfinite(scores[n]["normalized"])]
    payload = {
        "base_closure": "launder-sharma",
        "terms": list(TERM_NAMES),
        "fit_cases": fit_names,
        "bounds": dict(zip(TERM_NAMES, bounds)),
        "cap": cap,
        "n_init": args.n_init, "n_iter": args.n_iter, "seed": args.seed,
        "coeffs": best,
        "best_objective": res["best_y"],
        "n_evaluations": len(history),
        "n_diverged": res["n_infeasible"],
        "near_best_tolerance": args.near_best,
        "n_near_best": len(near),
        "coefficient_intervals": intervals,
        "scores": scores,
        "base_scores": base_scores,
        "in_sample_mean": float(np.mean(ins)) if ins else None,
        "out_of_sample_mean": float(np.mean(outs)) if outs else None,
        "history": history,
        "wall_time_s": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
