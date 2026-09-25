#!/usr/bin/env python
"""Sparse regression of an omega transport equation across the DNS families.

Ideas-log section 7.5. Given the omega fields of build-transport-targets,
which terms does the equation they obey need, and are those the same terms
in every flow? The question is stability of the term set, not in-sample
fit, so the output is how often each term is selected when each flow family
is left out, next to how well the fit predicts the family it did not see.

The equation, in the thin-shear form the profile data allow (wall-normal
derivatives only, advection along the mean flow on the left):

    U.grad(omega) - d/dy(nu domega/dy) = sum_i c_i theta_i - beta omega^2

In every family this is nearly a local balance: the left side, divided by
omega^2, is two orders of magnitude smaller than the sources. Regressing it
directly asks the fit to explain noise, and SST's own coefficients score
R^2 of -0.1 to -4000 on it. So the destruction term is fixed instead, the
implicit (SINDy-PI) form: divide by beta omega^2 and regress

    1 = sum_i (c_i/beta) theta_i/omega^2 + (1/beta) (D_nu - adv)/omega^2

The coefficients are ratios to beta, the transport term is one more column
the fit may keep or drop (dropping it says the flow is in local
equilibrium), and the error is the rms of the unexplained part as a fraction
of the destruction, where SST's published equation is a meaningful
baseline. The library starts from SST's and adds candidates, each divided
by omega^2:

    S2            S^2                      production (SST: gamma/beta 5.3-7.4)
    diff_t        d/dy(nu_t domega/dy)     turbulent diffusion (SST: sigma_w/beta)
    cross         dk/dy domega/dy / omega  cross-diffusion (SST: 2 sigma_w2/beta)
    transport     D_nu - U.grad(omega)     molecular diffusion minus advection,
                                           with coefficient 1/beta
    S_omega       S omega                  production linear in the strain
    S2_rev        S^2 max(0, 1 - 440/Re_v) production switched on by the
                                           vorticity Reynolds number, the
                                           rectifier our ablation found
                                           load-bearing
    omega2_lowre  omega^2 nu/nu_t          low-Reynolds-number destruction
    S2_lowre      S^2 nu/nu_t              low-Reynolds-number production
    grad_k2       omega nu_t (dk/dy / k)^2 gradient of k, squared
    grad_omega2   nu_t (domega/dy)^2/omega gradient of omega, squared
    wall          omega nu / y^2           explicit wall proximity

with nu_t = k/omega. Dividing each row by omega^2 also makes flows at
different scales and Reynolds numbers weigh alike.

Regression is STLSQ from PySINDy on normalized columns, bagged: each of
N_BOOT fits draws the same number of points from every family, so a family
with a million points does not outvote one with a thousand. A term's
inclusion probability is the fraction of fits that keep it.

Designs, for each of the two omega targets:
  pooled        every family together
  per family    each family alone: does each flow want the same terms?
  leave-one-out every family but one, scored on the one, against the same
                fit restricted to SST's terms, against SST's published inner
                and outer coefficients, and against the two columns that are
                functions of the structure parameter alone (A1_TERMS), so the
                extra terms have to beat SST's structure, and a constant a_1,
                out of sample to count

Outputs
-------
results/transport-equation-fit.json
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pysindy as ps

IN_H5 = "results/transport-targets.h5"
OUT_JSON = "results/transport-equation-fit.json"
TARGETS = ("omega_frozen", "omega_eps")
TERMS = ("S2", "diff_t", "cross", "transport", "S_omega", "S2_rev",
         "omega2_lowre", "S2_lowre", "grad_k2", "grad_omega2", "wall")
SST_TERMS = ("S2", "diff_t", "cross", "transport")
#: For the frozen target, S/omega = -uv/k exactly, the structure parameter,
#: so these two columns are a function of a_1 alone. A fit that does no
#: better than they do has found that a_1 is nearly constant, which is known,
#: rather than a transport equation: the trivial-solution guard
A1_TERMS = ("S2", "S_omega")
#: SST's published omega equation, as ratios to its beta: the inner
#: (k-omega) and outer (transformed k-epsilon) coefficient sets
SST_STANDARD = {
    "inner": {"S2": 0.5532 / 0.075, "diff_t": 0.5 / 0.075, "cross": 0.0,
              "transport": 1 / 0.075},
    "outer": {"S2": 0.4403 / 0.0828, "diff_t": 0.856 / 0.0828,
              "cross": 2 * 0.856 / 0.0828, "transport": 1 / 0.0828},
}
RE_V_CRIT = 440.0
N_BOOT = 100
N_PER_FAMILY = 2000
THRESHOLD = 0.05
#: Sparsity sweep: the same leave-one-out design at harder thresholds, for
#: the trade between the number of terms and the held-out error. Reported
#: beside the main result, which stays at THRESHOLD as set before the fit
SWEEP = (0.05, 0.1, 0.2, 0.4, 0.8)
RIDGE = 1e-3
#: A term selected in at least this fraction of fits counts as selected
SELECTED = 0.8
#: Wall region, and a floor on k relative to the case's peak, where the
#: targets are too singular or too noisy to regress
Y_PLUS_MIN = 30.0
K_FLOOR = 1e-3
#: Rows beyond this percentile of any column, within a family, are dropped:
#: a dimensionless row is a ratio, and a few near-zero omegas otherwise
#: decide the fit
CLIP_PCT = 99.5
SEED = 0


def station_rows(g, target):
    """Library rows along one wall-normal profile, divided by omega^2."""
    g = g.sort_values("y")
    y = g["y"].to_numpy()
    if len(y) < 5:
        return None
    w = g[target].to_numpy()
    k = g["k"].to_numpy()
    nu = float(g["nu"].iloc[0])
    S = np.abs(g["dUdy"].to_numpy())
    adv = g[f"adv_{target}"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        nut = k / w
        dw = np.gradient(w, y)
        dk = np.gradient(k, y)
        D_nu = np.gradient(nu * dw, y)
        re_v = y ** 2 * S / nu
        cols = {
            "S2": S ** 2,
            "diff_t": np.gradient(nut * dw, y),
            "cross": dk * dw / w,
            "transport": D_nu - adv,
            "S_omega": S * w,
            "S2_rev": S ** 2 * np.maximum(0.0, 1.0 - RE_V_CRIT / re_v),
            "omega2_lowre": w ** 2 * nu / nut,
            "S2_lowre": S ** 2 * nu / nut,
            "grad_k2": w * nut * (dk / k) ** 2,
            "grad_omega2": nut * dw ** 2 / w,
            "wall": w * nu / y ** 2,
        }
        # The wall region, from the station's own friction velocity
        u_tau = np.sqrt(nu * S[0]) if S[0] > 0 else np.nan
        ok = (np.isfinite(adv) & np.isfinite(w) & (w > 0)
              & (k > K_FLOOR * g["k_case_max"].iloc[0])
              & (y * u_tau / nu > Y_PLUS_MIN))
        if g["y_d99"].notna().any():
            ok &= g["y_d99"].to_numpy() < 0.9
        X = np.column_stack([cols[n] for n in TERMS]) / (w ** 2)[:, None]
    ok &= np.all(np.isfinite(X), axis=1)
    if ok.sum() == 0:
        return None
    # The fixed destruction term, beta omega^2 / (beta omega^2)
    return pd.DataFrame(X[ok], columns=TERMS).assign(
        t=1.0, family=g["family"].iloc[0], case=g["case"].iloc[0])


def build_rows(df, target):
    df = df.copy()
    df["k_case_max"] = df.groupby("case")["k"].transform("max")
    parts = [station_rows(g, target)
             for _, g in df[df[target].notna()].groupby(["case", "station"])]
    rows = pd.concat([p for p in parts if p is not None], ignore_index=True)
    keep = np.ones(len(rows), bool)
    for fam, g in rows.groupby("family"):
        vals = g[list(TERMS)].abs()
        lim = vals.quantile(CLIP_PCT / 100)
        keep[g.index] = (vals <= lim).all(axis=1).to_numpy()
    return rows[keep].reset_index(drop=True)


def bagged_fit(rows, terms, rng, threshold=THRESHOLD):
    """Inclusion probability and median coefficient of each term."""
    fams = sorted(rows["family"].unique())
    idx = {f: np.flatnonzero(rows["family"].to_numpy() == f) for f in fams}
    coefs = []
    for _ in range(N_BOOT):
        pick = np.concatenate([rng.choice(idx[f], N_PER_FAMILY) for f in fams])
        X = rows.loc[pick, list(terms)].to_numpy()
        t = rows.loc[pick, "t"].to_numpy()
        opt = ps.STLSQ(threshold=threshold, alpha=RIDGE,
                       normalize_columns=True)
        opt.fit(X, t)
        coefs.append(np.ravel(opt.coef_))
    C = np.array(coefs)
    nonzero = C != 0
    median = [float(np.median(C[nonzero[:, i], i])) if nonzero[:, i].any()
              else 0.0 for i in range(len(terms))]
    return {
        "inclusion": dict(zip(terms, nonzero.mean(axis=0).round(3).tolist())),
        "median_coefficient": dict(zip(terms, median)),
    }


def residual(rows, terms, coef):
    """rms of what the equation leaves unexplained, as a fraction of the
    destruction: 0 is exact, and 1 is no better than predicting no source
    at all."""
    X = rows[list(terms)].to_numpy()
    pred = X @ np.array([coef.get(n, 0.0) for n in terms])
    return float(np.sqrt(np.mean((rows["t"].to_numpy() - pred) ** 2)))


def selected_model(fit, terms):
    """The model a fit nominates: its frequently selected terms, at their
    median coefficients, and zero for the rest."""
    return {n: (fit["median_coefficient"][n]
                if fit["inclusion"][n] >= SELECTED else 0.0) for n in terms}


def run_target(df, target, rng):
    rows = build_rows(df, target)
    fams = sorted(rows["family"].unique())
    out = {
        "families": fams,
        "n_families": len(fams),
        "n_terms": len(TERMS),
        "n_rows": {f: int((rows["family"] == f).sum()) for f in fams},
    }
    if len(fams) < 2:
        out["skipped"] = "fewer than two families have this target"
        return out
    pooled = bagged_fit(rows, TERMS, rng)
    out["pooled"] = pooled
    out["pooled_residual"] = residual(rows, TERMS, selected_model(pooled, TERMS))
    out["per_family"] = {
        f: bagged_fit(rows[rows["family"] == f].reset_index(drop=True),
                      TERMS, rng)["inclusion"]
        for f in fams
    }
    lofo = {}
    for f in fams:
        train = rows[rows["family"] != f].reset_index(drop=True)
        test = rows[rows["family"] == f]
        full = bagged_fit(train, TERMS, rng)
        sst = bagged_fit(train, SST_TERMS, rng)
        a1 = bagged_fit(train, A1_TERMS, rng)
        lofo[f] = {
            "inclusion": full["inclusion"],
            "coefficients": selected_model(full, TERMS),
            "residual_in_sample": residual(
                train, TERMS, selected_model(full, TERMS)),
            "residual_held_out": residual(
                test, TERMS, selected_model(full, TERMS)),
            "residual_held_out_sst_terms": residual(
                test, SST_TERMS, selected_model(sst, SST_TERMS)),
            "residual_held_out_a1_only": residual(
                test, A1_TERMS, selected_model(a1, A1_TERMS)),
            **{f"residual_sst_{name}": residual(test, SST_TERMS, coef)
               for name, coef in SST_STANDARD.items()},
        }
    out["leave_one_family_out"] = lofo
    shared = [n for n in TERMS
              if all(v["inclusion"][n] >= SELECTED for v in lofo.values())]
    out["shared_terms"] = shared
    out["n_shared_terms"] = len(shared)
    out["n_shared_beyond_sst"] = len([n for n in shared
                                      if n not in SST_TERMS])
    held = [v["residual_held_out"] for v in lofo.values()]
    held_sst = [v["residual_held_out_sst_terms"] for v in lofo.values()]
    held_std = [min(v["residual_sst_inner"], v["residual_sst_outer"])
                for v in lofo.values()]
    out["median_residual_held_out"] = float(np.median(held))
    out["median_residual_held_out_sst_terms"] = float(np.median(held_sst))
    out["median_residual_sst_standard"] = float(np.median(held_std))
    held_a1 = [v["residual_held_out_a1_only"] for v in lofo.values()]
    out["median_residual_held_out_a1_only"] = float(np.median(held_a1))
    out["n_families_beating_a1_only"] = int(
        sum(a < b for a, b in zip(held, held_a1)))
    out["n_families_beating_sst_terms"] = int(
        sum(a < b for a, b in zip(held, held_sst)))
    out["sparsity_sweep"] = sweep(rows, fams, rng)
    out["n_families_beating_sst_standard"] = int(
        sum(a < b for a, b in zip(held, held_std)))
    return out


def sweep(rows, fams, rng):
    """Terms kept in every leave-one-out fit, and the median held-out
    residual, at each threshold of SWEEP."""
    out = []
    for th in SWEEP:
        fits, held = [], []
        for f in fams:
            train = rows[rows["family"] != f].reset_index(drop=True)
            fit = bagged_fit(train, TERMS, rng, threshold=th)
            fits.append(fit)
            held.append(residual(rows[rows["family"] == f], TERMS,
                                 selected_model(fit, TERMS)))
        shared = [n for n in TERMS
                  if all(fit["inclusion"][n] >= SELECTED for fit in fits)]
        out.append({
            "threshold": th,
            "shared_terms": shared,
            "n_shared_terms": len(shared),
            "median_residual_held_out": float(np.median(held)),
            "residual_held_out": dict(zip(fams, held)),
        })
    return out


def main():
    df = pd.read_hdf(IN_H5)
    rng = np.random.default_rng(SEED)
    result = {
        "terms": list(TERMS),
        "sst_terms": list(SST_TERMS),
        "settings": {
            "n_boot": N_BOOT, "n_per_family": N_PER_FAMILY,
            "threshold": THRESHOLD, "ridge": RIDGE, "selected": SELECTED,
            "y_plus_min": Y_PLUS_MIN, "k_floor": K_FLOOR,
            "clip_pct": CLIP_PCT, "re_v_crit": RE_V_CRIT, "seed": SEED,
            "sweep": list(SWEEP),
        },
        "targets": {t: run_target(df, t, rng) for t in TARGETS},
    }
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for t, r in result["targets"].items():
        print(f"== {t}: families {r['families']}")
        if "skipped" in r:
            print("  ", r["skipped"])
            continue
        print("  pooled inclusion:", r["pooled"]["inclusion"])
        print("  shared in every leave-one-out fit:", r["shared_terms"])
        print("  pooled coefficients:", {k: round(v, 3) for k, v in
              r["pooled"]["median_coefficient"].items()})
        for sw in r["sparsity_sweep"]:
            print(f"  threshold {sw['threshold']:<5} terms "
                  f"{sw['n_shared_terms']:2d} median held-out residual "
                  f"{sw['median_residual_held_out']:.3f} {sw['shared_terms']}")
        for f, v in r["leave_one_family_out"].items():
            print(f"  held out {f:18s} residual {v['residual_held_out']:.3f}"
                  f"   SST terms refit {v['residual_held_out_sst_terms']:.3f}"
                  f"   a1 only {v['residual_held_out_a1_only']:.3f}"
                  f"   SST inner {v['residual_sst_inner']:.3f}"
                  f"   outer {v['residual_sst_outer']:.3f}"
                  f"   in-sample {v['residual_in_sample']:.3f}")


if __name__ == "__main__":
    main()
