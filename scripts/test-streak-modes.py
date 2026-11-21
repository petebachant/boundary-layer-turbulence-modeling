#!/usr/bin/env python
"""Do free-stream-forced streak modes, carried on each flow's laminar mean,
predict the streaks' amplitude ahead of onset across flows, and the
pressure gradient's effect on it with nothing fitted to it?

The design is fixed in docs/ideas-log.md, section 7.14 (pypkg/streak_modes
for the marchers). Per flow: the laminar boundary layer under its edge
velocity; steady boundary-region modes over a grid of spanwise
wavenumbers, forced by a free-stream vortex of amplitude the local
free-stream rms and weighted by the von Karman transverse spectrum with
L_11 = L_eps / 4.3; u_rms summed over modes; A = u'_max / U_e below
Y_MAX delta_99. One constant, the receptivity, is fitted on the JHTDB
plate. Reference A as scripts/analyze-streak-write.py measures it.

Tests, fixed before any run:

  zpg    mean |ln(A_pred / A)| over the pooled pre-onset stations of Wu et
         al.'s five flows, T3A and T3B is at most TOL
  pg     the same over T3C2 to T3C5 is at most TOL, and below the same
         model's with the mean replaced by Blasius at the leading edge's
         U_e

Also reported: a constant B, A = B Tu_ref sqrt(Re_x) with B fitted on the
plate, the baseline the modes must beat.

Outputs
-------
results/streak-modes.json
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
from scipy.optimize import least_squares

from pypkg import streak_modes as sm
from pypkg.dns_case import load_dns

OUT = "results/streak-modes.json"
WRITE = "results/streak-write.json"
LENGTH = "results/freestream-length-scale.json"
INLETS = "results/wu-openfoam-inlets.json"
T3_LES = "results/t3-les.json"
TOL = 0.22
Y_MAX = 0.7
#: L_11 = L_eps / L_RATIO, from eps = 0.43 u'^3 / L_11 and k = 1.5 u'^2
L_RATIO = 4.3
#: Re_x where the leading-edge flows' marches start
RE_X0 = 2000.0
#: Wu et al.'s inflow holds no disturbance below this many delta_99
WU_EMPTY = 2.0
N_CELLS, N_X, N_BETA = 160, 400, 24
BETA_RANGE = (0.05, 20.0)
WU_TAGS = ("WM075", "WM150", "WM225", "WM300", "WM600")
ZPG_T3 = ("t3a", "t3b")
PG_T3 = ("t3c2", "t3c3", "t3c4", "t3c5")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_decay(t, k):
    """(ln A, t_v, n) of k = A (t - t_v)^(-n) in time of flight t, so
    L_eps = k^(3/2) / eps = sqrt(k) (t - t_v) / n."""

    def resid(p):
        logA, tv, n = p
        return logA - n * np.log(np.maximum(t - tv, 1e-12)) - np.log(k)

    r = least_squares(
        resid,
        [np.log(k[0]) + 1.3 * np.log(t[-1]), -0.5 * t[-1], 1.3],
        bounds=([-np.inf, -np.inf, 0.1], [np.inf, t[0] - 1e-12, 5.0]),
    )
    return r.x


def flow_jhtdb(write, length):
    d = load_dns()
    nu = d["nu"]
    fit = length["jhtdb"]["decay_fit"]
    A, xv, n = fit["A"], fit["x_v"], fit["n"]
    ue_x, ue_v = d["x"], d["U"].max(axis=0)
    x0 = RE_X0 * nu
    k0 = A * (x0 - xv) ** (-n)
    return {
        "nu": nu,
        "x0": x0,
        "x0_virtual": x0,
        "ue": lambda x: np.interp(x, ue_x, ue_v),
        "tu": lambda x: 100
        * np.sqrt(2 * A * (x - xv) ** (-n) / 3)
        / np.interp(x, ue_x, ue_v),
        "L_eps": float(np.sqrt(k0) * (x0 - xv) / n),
        "stations": write["flows"]["jhtdb"],
        "inside": None,
    }


def flow_wu(tag, write, length, inlets):
    nu = inlets[tag]["nu"]
    x_in = inlets[tag]["x_inlet"]
    fit = length["wu"][tag]["decay_fit"]
    A, xv, n = fit["A"], fit["x_v"], fit["n"]
    k0 = A * (x_in - xv) ** (-n)
    return {
        "nu": nu,
        "x0": x_in,
        "x0_virtual": x_in,
        "ue": lambda x: np.ones_like(x),
        "tu": lambda x: 100 * np.sqrt(2 * A * (x - xv) ** (-n) / 3),
        "L_eps": float(np.sqrt(k0) * (x_in - xv) / n),
        "stations": write["flows"][f"wu-{tag.lower()}"],
        "inside": "empty",
    }


def flow_t3(case, write, les, streak_write, no_pg=False):
    c = les["cases"][case]
    nu = c["nu_ref"]
    lx = np.array(c["line"]["x"])
    lue = np.array(c["line"]["ue"])
    ltu = np.array(c["line"]["tu"])
    m = lx >= 0
    lx, lue, ltu = lx[m], lue[m], ltu[m]
    x_on = c["x_cf_min"]
    # Time of flight from the leading edge, and the free stream's decay
    # from there to onset
    t = np.concatenate([[0.0], np.cumsum(np.diff(lx) / lue[1:])])
    k = 1.5 * (ltu * lue / 100) ** 2
    s = (lx > 0) & (lx <= x_on)
    logA, tv, n = fit_decay(t[s], k[s])
    k0 = np.exp(logA) * (0 - tv) ** (-n)
    ue0 = float(lue[0])
    if case in write["flows"]:
        stations = write["flows"][case]
    else:
        st = sorted(streak_write.t3(case))
        stations = {
            "stations": [{"re_x": r, "a": a} for r, a in st],
            "re_x_onset": c["rex_cf_min"],
        }
    ue = (
        (lambda x: np.full_like(x, ue0))
        if no_pg
        else (lambda x: np.interp(x, lx, lue))
    )
    return {
        "nu": nu,
        "x0": RE_X0 * nu / ue0,
        "x0_virtual": RE_X0 * nu / ue0,
        "ue": ue,
        # The forcing is the LES's free-stream rms, in both variants
        "eps": lambda x: np.interp(x, lx, ltu * lue / 100),
        "tu": lambda x: np.interp(x, lx, ltu),
        "L_eps": float(np.sqrt(k0) * (0 - tv) / n),
        "stations": stations,
        "inside": None,
        "x_of_rex": lambda r: np.interp(
            r, np.array(c["plate"]["rex"]), np.array(c["plate"]["x"])
        ),
    }


def predict(f):
    """A per unit receptivity at the flow's pre-onset stations."""
    nu = f["nu"]
    st = [
        s
        for s in f["stations"]["stations"]
        if s["re_x"] < f["stations"]["re_x_onset"]
    ]
    rex = np.array([s["re_x"] for s in st])
    a_ref = np.array([s["a"] for s in st])
    if "x_of_rex" in f:
        xs = f["x_of_rex"](rex)
    else:
        xs = rex * nu / f["ue"](np.array([f["x0"]]))[0]
    keep = xs > f["x0"]
    rex, a_ref, xs = rex[keep], a_ref[keep], xs[keep]
    x = np.geomspace(f["x0"], xs.max() * 1.001, N_X)
    ue = f["ue"](x)
    due = np.gradient(ue, x)
    eps = f["eps"](x) if "eps" in f else f["tu"](x) * ue / 100
    d_end = 5 * np.sqrt(nu * x[-1] / ue.min())
    d_0 = 5 * np.sqrt(nu * f["x0_virtual"] / ue[0])
    g = sm.Grid(N_CELLS, 10 * d_end, d_0 / 30)
    mean = sm.Mean(g, x, ue, due, nu, f["x0_virtual"])
    inside = None
    if f["inside"] == "empty":
        d99 = mean.delta99(0)
        inside = 0.5 * (1 + np.tanh((g.yc - WU_EMPTY * d99) / (0.25 * d99)))
    L11 = f["L_eps"] / L_RATIO
    betas = np.geomspace(*BETA_RANGE, N_BETA) / L11
    urms = sm.streak_rms(mean, eps, L11, betas, inside)
    a = np.empty(len(x))
    for i in range(len(x)):
        mm = g.yc <= Y_MAX * mean.delta99(i)
        a[i] = urms[i, mm].max() / ue[i]
    return {
        "re_x": rex.tolist(),
        "a_ref": a_ref.tolist(),
        "a_unit": np.interp(xs, x, a).tolist(),
        "tu_le": float(f["tu"](np.array([f["x0"]]))[0]),
        "L_eps": f["L_eps"],
        "separated_at": mean.separated_at,
    }


def score(rows, c):
    err = [
        abs(np.log(c * au / ar))
        for r in rows
        for au, ar in zip(r["a_unit"], r["a_ref"])
    ]
    return float(np.mean(err)), len(err)


def main():
    streak_write = load("streak_write", "scripts/analyze-streak-write.py")
    with open(WRITE) as f:
        write = json.load(f)
    with open(LENGTH) as f:
        length = json.load(f)
    with open(INLETS) as f:
        inlets = json.load(f)
    with open(T3_LES) as f:
        les = json.load(f)

    flows = {"jhtdb": flow_jhtdb(write, length)}
    for tag in WU_TAGS:
        flows[f"wu-{tag.lower()}"] = flow_wu(tag, write, length, inlets)
    for case in ZPG_T3 + PG_T3:
        flows[case] = flow_t3(case, write, les, streak_write)
    for case in PG_T3:
        flows[f"{case}-no-pg"] = flow_t3(
            case, write, les, streak_write, no_pg=True
        )

    rows = {}
    for name, f in flows.items():
        rows[name] = predict(f)
        print(
            name,
            {
                k: v
                for k, v in rows[name].items()
                if k not in ("re_x", "a_ref", "a_unit")
            },
            flush=True,
        )

    j = rows["jhtdb"]
    c = float(
        np.exp(np.mean(np.log(np.array(j["a_ref"]) / np.array(j["a_unit"]))))
    )
    for r in rows.values():
        r["a_pred"] = [c * v for v in r["a_unit"]]

    zpg = [f"wu-{t.lower()}" for t in WU_TAGS] + list(ZPG_T3)
    pg = list(PG_T3)
    zpg_err, zpg_n = score([rows[n] for n in zpg], c)
    pg_err, pg_n = score([rows[n] for n in pg], c)
    nopg_err, _ = score([rows[f"{n}-no-pg"] for n in pg], c)

    # The constant-B baseline, B fitted on the plate as the modes' c is
    def tu_ref(name):
        return write["flows"].get(name, {}).get("tu_ref_percent")

    b_rows = {}
    for name in ["jhtdb"] + zpg + pg:
        r = rows[name]
        tr = tu_ref(name)
        if tr is None:
            cse = les["cases"][name]
            tr = float(
                np.interp(
                    write["re_x_ref"],
                    np.array(cse["plate"]["rex"]),
                    np.interp(
                        np.array(cse["plate"]["x"]),
                        np.array(cse["line"]["x"]),
                        np.array(cse["line"]["tu"]),
                    ),
                )
            )
        b_rows[name] = {
            "a_unit": [tr / 100 * np.sqrt(x) for x in r["re_x"]],
            "a_ref": r["a_ref"],
        }
    jb = b_rows["jhtdb"]
    b = float(
        np.exp(np.mean(np.log(np.array(jb["a_ref"]) / np.array(jb["a_unit"]))))
    )
    zpg_b, _ = score([b_rows[n] for n in zpg], b)
    pg_b, _ = score([b_rows[n] for n in pg], b)

    result = {
        "tol": TOL,
        "l_ratio": L_RATIO,
        "receptivity": c,
        "constant_b": b,
        "zpg_err": zpg_err,
        "zpg_n": zpg_n,
        "pg_err": pg_err,
        "pg_n": pg_n,
        "pg_err_no_pg": nopg_err,
        "zpg_err_constant_b": zpg_b,
        "pg_err_constant_b": pg_b,
        "jhtdb_err": score([rows["jhtdb"]], c)[0],
        "per_flow_err": {n: score([r], c)[0] for n, r in rows.items()},
        "per_flow_err_constant_b": {
            n: score([r], b)[0] for n, r in b_rows.items()
        },
        "zpg_ok": bool(zpg_err <= TOL),
        "pg_ok": bool(pg_err <= TOL and pg_err < nopg_err),
        "flows": rows,
    }
    result["passes"] = result["zpg_ok"] and result["pg_ok"]
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items() if k != "flows"})


if __name__ == "__main__":
    main()
