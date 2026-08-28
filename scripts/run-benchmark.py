#!/usr/bin/env python
"""Run every registered closure against every registered case.

This is the harness the rest of the project has been missing. The point is not
that it runs many models -- it is that it splits **in-sample from
out-of-sample by construction**. Each closure declares in the registry which
cases its coefficients were fitted on, and the leaderboard reports those two
numbers separately and never averages them together.

That constraint exists because this project's own result is that in-sample
agreement did not distinguish a constitutive law from a curve fit. A benchmark
that reported one aggregate score would make the same mistake easy to repeat,
so the split is not a reporting preference here; it is the whole point.

Reading the output
------------------
``normalized`` is the mean over a case's metrics of (error / target), where
each target is the error at which that quantity counts as matching the data.
So 1.0 means "matches the data by inspection on every metric" and the number
means the same thing on every case, which is what makes a row comparable
across a transitional plate and a fully turbulent layer.

Outputs
-------
results/benchmark.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypkg import registry  # noqa: E402


def run_one(case, spec):
    t0 = time.time()
    try:
        closure = spec.build(case=case)
    except Exception as e:  # a closure that cannot even be built
        return {"error": f"{type(e).__name__}: {e}", "wall_time_s": 0.0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        score = case.evaluate(closure)
    score["wall_time_s"] = round(time.time() - t0, 3)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/benchmark.json")
    ap.add_argument("--cases", default="", help="comma-separated subset")
    ap.add_argument("--closures", default="", help="comma-separated subset")
    ap.add_argument("--tier", default=registry.TIER_PYTHON)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    all_cases = registry.cases(tier=args.tier)
    all_closures = registry.closures()
    if args.cases:
        want = [s.strip() for s in args.cases.split(",")]
        all_cases = {k: v for k, v in all_cases.items() if k in want}
    if args.closures:
        want = [s.strip() for s in args.closures.split(",")]
        all_closures = {k: v for k, v in all_closures.items() if k in want}
    if not all_cases or not all_closures:
        raise SystemExit("nothing to run: check --cases / --closures")

    results = {}
    case_info = {}
    for cname, cspec in sorted(all_cases.items()):
        case = cspec.build()
        case_info[cname] = {
            **case.describe(),
            "family": cspec.family,
            "tier": cspec.tier,
            "description": cspec.description,
        }
        results[cname] = {}
        for mname, mspec in sorted(all_closures.items()):
            sc = run_one(case, mspec)
            sc["in_sample"] = mspec.is_in_sample(cname)
            results[cname][mname] = sc
            if not args.quiet:
                tag = "in " if sc["in_sample"] else "OUT"
                n = sc.get("normalized")
                nstr = "diverged" if n is None or n != n or n == float("inf") \
                    else f"{n:8.3f}"
                print(f"  {cname:24s} {mname:22s} [{tag}] {nstr}"
                      f"  ({sc.get('wall_time_s', 0):.1f}s)")

    payload = {
        "cases": case_info,
        "closures": {
            k: {"description": v.description,
                "calibrated_on": list(v.calibrated_on),
                "openfoam_model": v.openfoam_model,
                "reference": v.reference}
            for k, v in sorted(all_closures.items())
        },
        "results": results,
        "leaderboard": leaderboard(results, all_closures),
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    if not args.quiet:
        print()
        print_matrix(results, all_closures)
        print()
        print_leaderboard(payload["leaderboard"])
        print(f"\nwrote {args.out}")


def leaderboard(results, closures):
    """Per-closure summary, in-sample and out-of-sample kept apart.

    Ranked on the out-of-sample score, because that is the question the
    harness exists to answer. A closure with no out-of-sample case is
    unranked rather than ranked first.
    """
    rows = []
    for mname in closures:
        ins, out = [], []
        diverged = []
        for cname, per_case in results.items():
            sc = per_case.get(mname, {})
            n = sc.get("normalized")
            if n is None or n != n or n == float("inf"):
                diverged.append(cname)
                continue
            (ins if sc.get("in_sample") else out).append(n)
        rows.append({
            "closure": mname,
            "in_sample_mean": _mean(ins),
            "in_sample_n": len(ins),
            "out_of_sample_mean": _mean(out),
            "out_of_sample_n": len(out),
            # How much worse the model gets when it leaves the data it was
            # fitted on. This is the number the paper is about.
            "transfer_penalty": (
                _mean(out) / _mean(ins)
                if _mean(ins) and _mean(out) is not None and _mean(ins) > 0
                else None),
            "diverged_on": diverged,
        })
    rows.sort(key=lambda r: (r["out_of_sample_mean"] is None,
                             r["out_of_sample_mean"] or 0.0))
    return rows


def _mean(v):
    return float(sum(v) / len(v)) if v else None


def print_matrix(results, closures):
    """Per-case scores, which are the actual content.

    Read this before the leaderboard. The aggregate below averages over cases
    of very different difficulty -- a k-epsilon model has no transition
    mechanism at all, so its score on a transitional plate says nothing about
    its skill on an equilibrium layer -- and the per-case row is the honest
    view. `*` marks a case the closure was calibrated on.
    """
    cases = sorted(results)
    print("per-case normalized score (1.0 = matches the data on every metric;"
          " * = in-sample)")
    print(f"{'closure':24s}" + "".join(f"{c[:22]:>24s}" for c in cases))
    print("-" * (24 + 24 * len(cases)))
    for m in sorted(closures):
        cells = []
        for c in cases:
            sc = results[c].get(m, {})
            n = sc.get("normalized")
            if n is None or n != n or n == float("inf"):
                txt = "crashed" if sc.get("error") else "diverged"
            else:
                txt = f"{n:.3f}" + ("*" if sc.get("in_sample") else "")
            cells.append(f"{txt:>24s}")
        print(f"{m:24s}" + "".join(cells))


def print_leaderboard(rows):
    print(f"{'closure':24s}{'in-sample':>12s}{'out-of-sample':>16s}"
          f"{'penalty':>10s}")
    print("-" * 62)
    for r in rows:
        def f(x):
            return "  --  " if x is None else f"{x:.3f}"
        print(f"{r['closure']:24s}{f(r['in_sample_mean']):>12s}"
              f"{f(r['out_of_sample_mean']):>16s}"
              f"{f(r['transfer_penalty']):>10s}")


if __name__ == "__main__":
    main()
