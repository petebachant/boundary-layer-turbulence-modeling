#!/usr/bin/env python
"""Brute-force term discovery with no dimensional constraint.

Ideas-log section 7.16, design and test fixed there before this ran.
Thousands of monomials in the mean-flow primitives (M) and in those plus k
and eps (K), each raised to any power in {-2, -1, 1, 2, 3}, are regressed
by sequentially thresholded ridge least squares on four targets:

  S-M  the shear stress -uv, from M alone
  S-K  the shear stress -uv, from K
  E    eps's equation in implicit form, eps^2/k on the left
  T    k's turbulent transport, adv(k) - d2k/dy2 - P + eps

Every case is in units where its reference speed and nu are 1, so lengths
are in nu / U_ref and a dimensionally free coefficient means the same thing
in every flow. Each target is fitted leaving one flow family out at a time,
the threshold set on the training families to leave at most 10 terms, and
scored on the family left out against a baseline refitted the same way.

Outputs
-------
results/big-library.json
"""

from __future__ import annotations

import itertools
import json
import re

import numpy as np
import pandas as pd
import scipy.linalg as sla

IN_H5 = "results/transport-targets.h5"
OUT_JSON = "results/big-library.json"
BETA_STAR = 0.09
POWERS = (-2, -1, 1, 2, 3)
MAX_FACTORS = 3
MAX_DEGREE = 4
N_PER_FAMILY = 3000
RIDGE = 1e-5
BUDGET = 10
FRONT = (1, 2, 3, 5, 10, 20, 50)
THRESH_STEP = 10 ** (1 / 12)
N_BISECT = 30
PASS_RATIO = 0.8
SEED = 0

M_VARS = ("y", "U", "V", "Uy", "Uyy", "Px")
K_VARS = M_VARS + ("k", "eps", "ky", "epsy", "kyy", "epsyy")

#: (velocity, length) exponents of each quantity, for the change to units
#: with U_ref = nu = 1
DIMS = {
    "y": (0, 1),
    "U": (1, 0),
    "V": (1, 0),
    "Uy": (1, -1),
    "Uyy": (1, -2),
    "Px": (2, -1),
    "k": (2, 0),
    "eps": (3, -1),
    "ky": (2, -1),
    "kyy": (2, -2),
    "epsy": (3, -2),
    "epsyy": (3, -3),
    "uv": (2, 0),
    "adv_k": (3, -1),
    "adv_eps": (4, -2),
}


def _grad(q, y):
    if len(y) < 3:
        return np.full_like(q, np.nan)
    return np.gradient(q, y, edge_order=2)


def primitives(df):
    """Every row with its primitives, in units U_ref = nu = 1, wall rows
    dropped."""
    out = []
    for case, g in df.groupby("case", sort=True):
        fam = g["family"].iloc[0]
        nu = float(g["nu"].iloc[0])
        u_ref = float(g["U"].max()) if fam == "channel" else 1.0
        g = g.sort_values(["station", "y"]).copy()
        omega = g["eps"] / (BETA_STAR * g["k"])
        g["adv_eps"] = (
            BETA_STAR * omega * g["adv_k"]
            + BETA_STAR * g["k"] * g["adv_omega_eps"]
        )
        g = g.rename(columns={"dUdy": "Uy"})
        parts = []
        for _, s in g.groupby("station", sort=True):
            s = s.copy()
            y = s["y"].to_numpy()
            s["Uyy"] = _grad(s["Uy"].to_numpy(), y)
            s["ky"] = _grad(s["k"].to_numpy(), y)
            s["kyy"] = _grad(s["ky"].to_numpy(), y)
            s["epsy"] = _grad(s["eps"].to_numpy(), y)
            s["epsyy"] = _grad(s["epsy"].to_numpy(), y)
            s["Ue"] = s["U"].iloc[-1]
            parts.append(s)
        g = pd.concat(parts)
        if fam == "channel":
            re_tau = float(re.search(r"retau-(\d+)", case).group(1))
            g["Px"] = -1.0 / re_tau  # wall units: u_tau = nu = 1, h = Re_tau
        elif fam == "zpg-tbl":
            g["Px"] = 0.0
        else:
            st = g.groupby("station")["Ue"].first()
            x = st.index.to_numpy(float)
            px = -st.to_numpy() * np.gradient(st.to_numpy(), x)
            g["Px"] = g["station"].map(dict(zip(st.index, px)))
        length = nu / u_ref
        for name, (a, b) in DIMS.items():
            g[name] = g[name] / (u_ref**a * length**b)
        g = g[g["y"] > 1e-6]  # the wall, to round-off
        out.append(g)
    return pd.concat(out, ignore_index=True)


def monomials(variables, exclude=()):
    terms = [()]
    for n in range(1, MAX_FACTORS + 1):
        for vs in itertools.combinations(variables, n):
            for ps in itertools.product(POWERS, repeat=n):
                if sum(abs(p) for p in ps) <= MAX_DEGREE:
                    terms.append(tuple(zip(vs, ps)))
    return [t for t in terms if t not in exclude]


def term_name(t):
    if t == ():
        return "1"
    if isinstance(t, str):
        return t
    return " ".join(v if p == 1 else f"{v}^{p}" for v, p in t)


def evaluate(t, d):
    if isinstance(t, str):
        return SPECIAL[t](d)
    col = np.ones(len(d))
    with np.errstate(all="ignore"):
        for v, p in t:
            col = col * d[v].to_numpy(float) ** p
    return col


def _ddy_k2_over_eps(q, qy, qyy):
    """d/dy(k^2/eps dq/dy), expanded in the primitives."""

    def f(d):
        k, e, ky, ey = d["k"], d["eps"], d["ky"], d["epsy"]
        return (
            k**2 / e * d[qyy]
            + 2 * k * ky * d[qy] / e
            - k**2 * ey * d[qy] / e**2
        ).to_numpy(float)

    return f


SPECIAL = {
    "y^2 Uy|Uy|": lambda d: (d["y"] ** 2 * d["Uy"] * d["Uy"].abs()).to_numpy(
        float
    ),
    "k^2/eps Uy": lambda d: (d["k"] ** 2 / d["eps"] * d["Uy"]).to_numpy(float),
    "k Uy^2": lambda d: (d["k"] * d["Uy"] ** 2).to_numpy(float),
    "ddy(k^2/eps epsy)": _ddy_k2_over_eps("eps", "epsy", "epsyy"),
    "ddy(k^2/eps ky)": _ddy_k2_over_eps("k", "ky", "kyy"),
    "adv_eps - epsyy": lambda d: (d["adv_eps"] - d["epsyy"]).to_numpy(float),
}

TARGETS = {
    "S-M": {
        "target": lambda d: -d["uv"].to_numpy(float),
        "vars": M_VARS,
        "needs": ("uv",),
        "extra": (),
        "exclude": (),
        "baseline": ("y^2 Uy|Uy|",),
        "drop_cases": ("jhtdb-transitional-bl-sampled",),
    },
    "S-K": {
        "target": lambda d: -d["uv"].to_numpy(float),
        "vars": K_VARS,
        "needs": ("uv",),
        "extra": (),
        "exclude": (),
        "baseline": ("k^2/eps Uy",),
        "drop_cases": (),
    },
    "E": {
        "target": lambda d: (d["eps"] ** 2 / d["k"]).to_numpy(float),
        "vars": K_VARS,
        "needs": ("adv_eps",),
        "extra": ("adv_eps - epsyy",),
        "exclude": ((("k", -1), ("eps", 2)),),
        "baseline": ("k Uy^2", "ddy(k^2/eps epsy)", "adv_eps - epsyy"),
        "drop_cases": (),
    },
    "T": {
        "target": lambda d: (
            d["adv_k"] - d["kyy"] + d["uv"] * d["Uy"] + d["eps"]
        ).to_numpy(float),
        "vars": K_VARS,
        "needs": ("adv_k", "uv"),
        "extra": (),
        "exclude": (),
        "baseline": ("ddy(k^2/eps ky)",),
        "drop_cases": (),
    },
}


def rows_for(prim, spec):
    cols = list(spec["vars"]) + list(spec["needs"])
    d = prim[~prim["case"].isin(spec["drop_cases"])]
    ok = np.isfinite(d[cols].to_numpy(float)).all(axis=1)
    d = d[ok].copy()
    d["_t"] = spec["target"](d)
    d = d[np.isfinite(d["_t"])]
    rms = d.groupby("case")["_t"].transform(lambda s: np.sqrt(np.mean(s**2)))
    d["_w"] = 1.0 / rms
    return d[np.isfinite(d["_w"])]


def sample(d, fams, rng):
    picks = []
    for f in fams:
        cases = sorted(d.loc[d["family"] == f, "case"].unique())
        n = N_PER_FAMILY // len(cases)
        for c in cases:
            idx = np.flatnonzero((d["case"] == c).to_numpy())
            picks.append(rng.choice(idx, n, replace=len(idx) < n))
    return d.iloc[np.concatenate(picks)]


def design(terms, d):
    """Weighted columns scaled to unit rms; terms not finite on every row,
    or zero, are dropped."""
    w = d["_w"].to_numpy()
    cols, keep = [], []
    for t in terms:
        c = evaluate(t, d) * w
        if not np.all(np.isfinite(c)):
            continue
        s = np.sqrt(np.mean(c**2))
        if not np.isfinite(s) or s == 0:
            continue
        cols.append(c / s)
        keep.append((t, s))
    return np.column_stack(cols), keep


def _solve(G, b, S):
    A = G[np.ix_(S, S)] + RIDGE * np.eye(len(S))
    return sla.solve(A, b[S], assume_a="pos")


def _stlsq(G, b, S, c, thr):
    """Threshold and refit from support S with coefficients c until no
    coefficient is below thr."""
    for _ in range(200):
        keep = np.abs(c) >= thr
        if keep.all():
            return S, c
        if not keep.any():
            return np.array([], int), np.array([])
        S = S[keep]
        c = _solve(G, b, S)
    return S, c


def threshold_path(X, yw):
    """STLSQ along increasing thresholds, each started from the previous
    support, until no term is left. Returns the path, each step
    (threshold, support, coefficients), and the normal equations."""
    n = len(yw)
    G = X.T @ X / n
    b = X.T @ yw / n
    S = np.arange(X.shape[1])
    c = _solve(G, b, S)
    path = []
    thr = 1e-7
    while len(S):
        S, c = _stlsq(G, b, S, c, thr)
        path.append((float(thr), S.copy(), c.copy()))
        thr *= THRESH_STEP
    return path, G, b


def pick(path, G, b, budget):
    """The smallest threshold leaving at most `budget` terms: the first step
    of the path that does, then bisection between it and the step before,
    each trial started from that step's support."""
    i = next(i for i, (_, S, _) in enumerate(path) if len(S) <= budget)
    if i == 0:
        return path[0]
    lo_thr, S0, c0 = path[i - 1]
    best = path[i]
    hi_thr = best[0]
    for _ in range(N_BISECT):
        mid = float(np.sqrt(lo_thr * hi_thr))
        S, c = _stlsq(G, b, S0, c0, mid)
        if len(S) <= budget:
            best, hi_thr = (mid, S, c), mid
        else:
            lo_thr = mid
    return best


def case_errors(model, d):
    """rms(prediction - target)/rms(target) per case; inf where the model
    is not finite."""
    out = {}
    for case, g in d.groupby("case", sort=True):
        t = g["_t"].to_numpy()
        p = np.zeros(len(g))
        for term, coef in model:
            p = p + coef * evaluate(term, g)
        if not np.all(np.isfinite(p)):
            out[case] = float("inf")
            continue
        out[case] = float(np.sqrt(np.mean((p - t) ** 2) / np.mean(t**2)))
    return out


def family_error(model, d, fam):
    g = d[d["family"] == fam]
    errs = case_errors(model, g)
    return float(np.mean(list(errs.values()))), errs


def fit_model(terms, d, fams, rng, budgets):
    s = sample(d[d["family"].isin(fams)], fams, rng)
    X, keep = design(terms, s)
    yw = s["_t"].to_numpy() * s["_w"].to_numpy()
    path, G, b = threshold_path(X, yw)
    models = {}
    for k in budgets:
        thr, S, c = pick(path, G, b, k)
        models[k] = {
            "threshold": thr,
            "terms": [
                (keep[j][0], float(c[i] / keep[j][1])) for i, j in enumerate(S)
            ],
        }
    return models, len(keep)


def fit_baseline(names, d, fams, rng):
    s = sample(d[d["family"].isin(fams)], fams, rng)
    w = s["_w"].to_numpy()
    X = np.column_stack([evaluate(t, s) * w for t in names])
    scale = np.sqrt(np.mean(X**2, axis=0))
    coef, *_ = np.linalg.lstsq(X / scale, s["_t"].to_numpy() * w, rcond=None)
    return list(zip(names, map(float, coef / scale)))


def describe(model):
    return [{"term": term_name(t), "coef": c} for t, c in model]


def run_target(name, spec, prim, rng):
    d = rows_for(prim, spec)
    fams = sorted(d["family"].unique())
    terms = monomials(spec["vars"], spec["exclude"]) + list(spec["extra"])
    out = {
        "families": fams,
        "n_rows": int(len(d)),
        "n_library": len(terms),
        "folds": {},
    }
    for held in fams:
        train = [f for f in fams if f != held]
        models, n_used = fit_model(terms, d, train, rng, FRONT)
        base = fit_baseline(spec["baseline"], d, train, rng)
        fold = {"n_library_finite": n_used, "front": {}}
        for k, m in models.items():
            err, per_case = family_error(m["terms"], d, held)
            ins = np.mean([family_error(m["terms"], d, f)[0] for f in train])
            fold["front"][str(k)] = {
                "n_terms": len(m["terms"]),
                "threshold": m["threshold"],
                "held_out_err": err,
                "held_out_by_case": per_case,
                "in_sample_err": float(ins),
                "model": describe(m["terms"]),
            }
        berr, bcase = family_error(base, d, held)
        bins = np.mean([family_error(base, d, f)[0] for f in train])
        fold["baseline"] = {
            "held_out_err": berr,
            "held_out_by_case": bcase,
            "in_sample_err": float(bins),
            "model": describe(base),
        }
        out["folds"][held] = fold
        print(
            f"{name} held {held}: {BUDGET}-term "
            f"{fold['front'][str(BUDGET)]['held_out_err']:.3g} "
            f"(in {fold['front'][str(BUDGET)]['in_sample_err']:.3g}), "
            f"baseline {berr:.3g} (in {bins:.3g}), "
            f"{n_used} finite terms",
            flush=True,
        )
    pooled, _ = fit_model(terms, d, fams, rng, (BUDGET,))
    out["pooled"] = describe(pooled[BUDGET]["terms"])
    errs = np.array(
        [out["folds"][f]["front"][str(BUDGET)]["held_out_err"] for f in fams]
    )
    berrs = np.array(
        [out["folds"][f]["baseline"]["held_out_err"] for f in fams]
    )
    wins = int(np.sum(errs < berrs))
    med, bmed = float(np.median(errs)), float(np.median(berrs))
    sets = [
        frozenset(
            t["term"] for t in out["folds"][f]["front"][str(BUDGET)]["model"]
        )
        for f in fams
    ]
    jac = [
        len(a & b) / len(a | b)
        for a, b in itertools.combinations(sets, 2)
        if a | b
    ]
    freq = {}
    for s in sets:
        for t in s:
            freq[t] = freq.get(t, 0) + 1
    out.update(
        {
            "wins": wins,
            "n_folds": len(fams),
            "median_held_out_err": med,
            "median_baseline_err": bmed,
            "ratio": med / bmed if bmed > 0 else float("inf"),
            "mean_jaccard": float(np.mean(jac)) if jac else 0.0,
            "term_frequency": dict(sorted(freq.items(), key=lambda kv: -kv[1])),
            "passes": bool(wins >= len(fams) - 1 and med <= PASS_RATIO * bmed),
        }
    )
    return out


def _clean(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def main():
    rng = np.random.default_rng(SEED)
    prim = primitives(pd.read_hdf(IN_H5, "targets"))
    res = {
        "powers": POWERS,
        "max_factors": MAX_FACTORS,
        "max_degree": MAX_DEGREE,
        "n_per_family": N_PER_FAMILY,
        "ridge": RIDGE,
        "budget": BUDGET,
        "pass_ratio": PASS_RATIO,
        "targets": {},
    }
    for name, spec in TARGETS.items():
        res["targets"][name] = run_target(name, spec, prim, rng)
        r = res["targets"][name]
        print(
            f"== {name}: wins {r['wins']}/{r['n_folds']}, median "
            f"{r['median_held_out_err']:.3g} vs {r['median_baseline_err']:.3g}"
            f", jaccard {r['mean_jaccard']:.2f}, passes {r['passes']}",
            flush=True,
        )
    res["any_passes"] = any(r["passes"] for r in res["targets"].values())
    with open(OUT_JSON, "w") as f:
        json.dump(_clean(res), f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
