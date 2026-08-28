"""Evolutionary structural search over RANS closure PDEs.

Two nested loops:

  outer   discrete search over PDE STRUCTURE (which operators and derived
          quantities appear) -- mutation + crossover over the grammar
  inner   continuous fit of the coefficients that structure requires

The outer objective is the best a-posteriori error the structure can reach
after its inner fit, plus a parsimony penalty. Fitting coefficients inside
the loop is what makes the structural comparison fair: a good structure with
bad constants would otherwise look worse than a bad structure with tuned ones.

A parsimony penalty is essential. Without it the search always adds terms,
because extra terms can only reduce training error. The interesting models
sit at the knee of the error-vs-complexity Pareto front, not at its floor.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .grammar import Candidate, crossover, mutate, random_candidate

_CASE = None

# Coefficient bounds for the grammar's continuous parameters
BASE_BOUNDS = {
    "Cgam": (1e-3, 50.0, "log"),
    "gseed": (1e-4, 0.2, "log"),
    "CL": (0.002, 0.3, "log"),
    "Cnu": (0.2, 20.0, "log"),
    "Cs_cap": (0.05, 1.0),
}
TERM_BOUNDS = {"t": (20.0, 3000.0, "log"), "p": (0.25, 3.0)}


def _get_case(root, x_stride):
    global _CASE
    if _CASE is None:
        from .dns_case import Case
        _CASE = Case(root=root, x_stride=x_stride)
    return _CASE


def bounds_for(cand):
    b = dict(BASE_BOUNDS)
    for i in range(cand.n_terms()):
        b[f"t{i}"] = TERM_BOUNDS["t"]
        b[f"p{i}"] = TERM_BOUNDS["p"]
    return b


def _sample(rng, bounds):
    c = {}
    for name, spec in bounds.items():
        lo, hi, *rest = spec
        if rest and rest[0] == "log":
            c[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            c[name] = float(rng.uniform(lo, hi))
    return c


def _score_one(cand, coeffs, root, x_stride):
    from .closures import GrammarKOmegaGamma
    case = _get_case(root, x_stride)
    model_kw = {k: coeffs[k] for k in ("CL", "Cnu", "Cs_cap") if k in coeffs}
    gc = {k: v for k, v in coeffs.items() if k not in model_kw}
    try:
        closure = GrammarKOmegaGamma(
            candidate=cand, gcoeffs=gc, k_inf=case.kinf_fn(), **model_kw
        )
        res = case.solve(closure)
        U = res["U"]
        if not np.all(np.isfinite(U)):
            return math.inf, {}
        sc = case.score(U)
        tot = sc["total"]
        return (tot if np.isfinite(tot) else math.inf), sc
    except Exception:
        return math.inf, {}


def _fit_worker(args):
    cand, n_coeff, root, x_stride, seed = args
    rng = np.random.default_rng(seed)
    b = bounds_for(cand)
    best, best_c, best_sc = math.inf, None, {}
    for _ in range(n_coeff):
        c = _sample(rng, b)
        t, sc = _score_one(cand, c, root, x_stride)
        if t < best:
            best, best_c, best_sc = t, c, sc
    # local refinement around the incumbent
    if best_c is not None:
        for shrink in (0.3, 0.12):
            for _ in range(max(8, n_coeff // 3)):
                c = {}
                for name, val in best_c.items():
                    lo, hi, *rest = b[name]
                    if rest and rest[0] == "log":
                        c[name] = float(np.clip(
                            val * np.exp(rng.normal(0, shrink * 2)), lo, hi))
                    else:
                        c[name] = float(np.clip(
                            val + rng.normal(0, shrink * (hi - lo)), lo, hi))
                t, sc = _score_one(cand, c, root, x_stride)
                if t < best:
                    best, best_c, best_sc = t, c, sc
    return cand.key(), best, best_c, best_sc, cand


def evolve(n_gen=6, pop=16, n_coeff=40, root=".", x_stride=12, seed=0,
           parsimony=0.02, max_terms=3, workers=None, log=print):
    """Run the structural search. Returns the ranked archive."""
    rng = np.random.default_rng(seed)
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    population = [random_candidate(rng, max_terms=max_terms) for _ in range(pop)]
    archive = {}

    for gen in range(n_gen):
        todo = [c for c in population if c.key() not in archive]
        args = [(c, n_coeff, root, x_stride, int(rng.integers(1 << 30)))
                for c in todo]
        if args:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                for key, val, cf, sc, cand in ex.map(_fit_worker, args,
                                                     chunksize=1):
                    archive[key] = {"total": val, "coeffs": cf, "score": sc,
                                    "cand": cand,
                                    "fitness": val + parsimony * cand.n_terms()}
        ranked = sorted(archive.values(), key=lambda r: r["fitness"])
        log(f"gen {gen}: archive={len(archive)} "
            f"best_fitness={ranked[0]['fitness']:.4f} "
            f"best_total={ranked[0]['total']:.4f} "
            f"[{ranked[0]['cand'].key()}]")

        # Next generation: elites, mutants, crossovers, and fresh blood
        elites = [r["cand"] for r in ranked[: max(2, pop // 4)]]
        nxt = list(elites)
        while len(nxt) < pop:
            r = rng.random()
            if r < 0.55 and elites:
                nxt.append(mutate(elites[int(rng.integers(len(elites)))], rng,
                                  max_terms=max_terms))
            elif r < 0.85 and len(elites) >= 2:
                i, j = rng.choice(len(elites), size=2, replace=False)
                nxt.append(crossover(elites[i], elites[j], rng))
            else:
                nxt.append(random_candidate(rng, max_terms=max_terms))
        population = nxt

    return sorted(archive.values(), key=lambda r: r["fitness"])
