"""Coefficient search for candidate RANS closures.

Evaluates a closure over the whole plate with the fast parabolic solver, so a
few hundred candidates can be screened in minutes rather than days of CFD.
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_CASE = None


def _get_case(root, x_stride):
    global _CASE
    if _CASE is None:
        from .dns_case import Case
        _CASE = Case(root=root, x_stride=x_stride)
    return _CASE


def freestream_error(res, case):
    """Relative RMS error in the SOLVED free-stream k against the DNS.

    Measured from the solution at the top of the domain, not from the analytic
    decay law. The two are not the same: the model's actual k equation gates
    dissipation by gamma, which in the free stream is small, so the solved
    decay can be far slower than the law predicts. Scoring the law instead of
    the solution is how a free stream 15x too energetic passed unnoticed and
    then ruined the elliptic run.
    """
    dns = np.array([float(case.kinf_fn()(x)) for x in case.x])
    mod = res["k"][-1, :] if "k" in res else None
    if mod is None:
        return 0.0
    return float(np.sqrt(np.mean(np.log(np.maximum(mod, 1e-30)
                                        / np.maximum(dns, 1e-30)) ** 2)))


def evaluate(cls, coeffs, root=".", x_stride=8, extra=None, w_fs=1.0):
    """Score one candidate. Returns +inf if it blows up."""
    case = _get_case(root, x_stride)
    kw = dict(coeffs)
    if extra:
        kw.update(extra)
    kw["k_inf"] = case.kinf_fn()
    try:
        closure = cls(**kw)
        res = case.solve(closure)
        U = res["U"]
        if not np.all(np.isfinite(U)):
            return math.inf, {}
        sc = case.score(U)
        if not np.isfinite(sc["total"]):
            return math.inf, {}
        fs = freestream_error(res, case)
        sc["freestream_rel_rms"] = fs
        sc["total"] = sc["total"] + w_fs * fs / case.TARGETS["freestream_rel_rms"]
        if not np.isfinite(sc["total"]):
            return math.inf, {}
        return sc["total"], sc
    except Exception:
        return math.inf, {}


def _worker(args):
    cls, coeffs, root, x_stride, extra = args
    total, sc = evaluate(cls, coeffs, root, x_stride, extra)
    return total, coeffs, sc


def random_search(cls, bounds, n=200, root=".", x_stride=8, seed=0,
                  extra=None, workers=None, log_every=25):
    """Latin-hypercube-ish random search over the coefficient bounds."""
    rng = np.random.default_rng(seed)
    keys = list(bounds)
    samples = []
    for _ in range(n):
        c = {}
        for kname in keys:
            lo, hi, *rest = bounds[kname]
            log = bool(rest and rest[0] == "log")
            if log:
                c[kname] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
            else:
                c[kname] = float(rng.uniform(lo, hi))
        samples.append(c)
    args = [(cls, c, root, x_stride, extra) for c in samples]
    out = []
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_worker, args, chunksize=2)):
            out.append(r)
            if log_every and (i + 1) % log_every == 0:
                best = min(out, key=lambda t: t[0])[0]
                print(f"  {i+1}/{n} evaluated, best={best:.4f}", flush=True)
    out.sort(key=lambda t: t[0])
    return out


def refine(cls, bounds, start, n=150, shrink=0.35, root=".", x_stride=8,
           seed=1, extra=None, workers=None):
    """Gaussian search around an incumbent, clipped to the bounds."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n):
        c = {}
        for kname, val in start.items():
            lo, hi, *_ = bounds[kname]
            span = (hi - lo) * shrink
            c[kname] = float(np.clip(val + rng.normal(0, span), lo, hi))
        samples.append(c)
    args = [(cls, c, root, x_stride, extra) for c in samples]
    out = []
    workers = workers or max(1, (os.cpu_count() or 4) - 1)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_worker, args, chunksize=2))
    out.sort(key=lambda t: t[0])
    return out
