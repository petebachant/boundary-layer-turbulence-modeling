#!/usr/bin/env python
"""Evolve the STRUCTURE of the RANS closure, not just its coefficients.

Searches the grammar in pypkg/grammar.py by mutation and crossover,
fitting each candidate structure's coefficients before scoring it, and
penalising term count so the result is a Pareto front rather than the most
elaborate model that fits.

Outputs
-------
results/closure-evolution.json   ranked archive of every structure tried
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pypkg.evolve import evolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=6)
    ap.add_argument("--population", type=int, default=14)
    ap.add_argument("--n-coeff", type=int, default=35)
    ap.add_argument("--x-stride", type=int, default=12)
    ap.add_argument("--parsimony", type=float, default=0.02)
    ap.add_argument("--max-terms", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/closure-evolution.json")
    args = ap.parse_args()

    archive = evolve(
        n_gen=args.generations,
        pop=args.population,
        n_coeff=args.n_coeff,
        x_stride=args.x_stride,
        parsimony=args.parsimony,
        max_terms=args.max_terms,
        seed=args.seed,
    )

    out = []
    for rec in archive:
        cand = rec["cand"]
        out.append({
            "structure": cand.key(),
            "rate": cand.rate,
            "shape": cand.shape,
            "terms": [{"driver": d, "response": r} for d, r in cand.terms],
            "n_terms": cand.n_terms(),
            "total": rec["total"],
            "fitness": rec["fitness"],
            "score": {k: float(v) for k, v in (rec["score"] or {}).items()},
            "coeffs": rec["coeffs"],
        })

    # Pareto front over (error, complexity)
    front, best_by_n = [], {}
    for r in sorted(out, key=lambda r: r["total"]):
        n = r["n_terms"]
        if n not in best_by_n:
            best_by_n[n] = r
    seen = float("inf")
    for n in sorted(best_by_n):
        r = best_by_n[n]
        if r["total"] < seen:
            front.append({"n_terms": n, "structure": r["structure"],
                          "total": r["total"]})
            seen = r["total"]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"pareto_front": front, "archive": out}, f, indent=2)

    print(f"\nevaluated {len(out)} structures")
    print("\nPareto front (error vs complexity):")
    for p in front:
        print(f"  {p['n_terms']} term(s)  total={p['total']:.4f}  {p['structure']}")
    print("\ntop 8 by fitness:")
    for r in sorted(out, key=lambda r: r["fitness"])[:8]:
        print(f"  {r['fitness']:.4f} (total {r['total']:.4f})  {r['structure']}")


if __name__ == "__main__":
    main()
