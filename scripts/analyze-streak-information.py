#!/usr/bin/env python
"""Do the layers overflow into turbulence at a common information content
where they do not at a common streak amplitude?

The streak amplitude at onset, u'_max / U_e, is not common to the flows:
about 0.085 to 0.12 on Wu et al.'s DNS and 0.12 to 0.14 on the layers
grown from a leading edge (results/streak-write.json). In the reading of
transition as information moving between media (docs/ideas-log.md,
sections 7.12 and 7.13), amplitude is only the store's size; what it holds
is the structure the forcing has written, of which the single-point
measure is the u-v mutual information of a Gaussian pair,

    I = -1/2 ln(1 - R_uv^2),   R_uv = -<u'v'> / (u_rms v_rms),

zero for uncorrelated fluctuations, larger the more the wall has
organized them (it rises elevenfold through the JHTDB plate's transition,
section 1.5). Each station's I is its mean over Y_LO to Y_HI delta_99,
away from the wall, where v_rms vanishes, and from the edge; the value at
onset is interpolated between the stations either side, onset and the
flows as in results/streak-write.json:

  Wu et al. (DNS)        the R_uv profiles, at Re_theta 147 and above
  Bienner et al. (LES)   the uv, u_rms and v_rms profiles, air at Mach 0.1
  JHTDB plate (DNS)      the profiles, every JHTDB_EVERY-th, from x = 30
  T3A, T3B (LES)         the twelve profiles of results/t3-les-profiles.npz
  T3A, T3B, T3A- (exp.)  the U-V cross-wire traverses, at the single-wire
                         stations of the same run numbers; out of sample

Prediction, fixed before I was computed for any flow:

  information  I at onset varies by less than SPREAD (max over min, less
               one) across the simulations with stations either side of
               their onset, and by less than u'_max / U_e does across the
               same flows

Also reported: I and R_uv along each flow, so that how fast each layer
organizes can be compared with how fast it grows.

Outputs
-------
results/streak-information.json
"""

from __future__ import annotations

import glob
import json
import re

import numpy as np

from pypkg.dns_case import load_dns

INFLOW = "results/onset-inflow.json"
WRITE = "results/streak-write.json"
WU = "data/wu-bypass-transition/stats_{0}"
WU_TAGS = ("WM075", "WM150", "WM225", "WM300", "WM600")
BIENNER_EVOL = "data/bienner-bypass-les/evol_streamwise/stats_vs_x_{}.dat"
BIENNER_PROF = "data/bienner-bypass-les/profiles/{}"
BIENNER_SKIP = 19
BIENNER_M01 = ("low_Lf_Tu4_Air_M0p1", "high_Lf_Tu4_Air_M0p1")
T3_PROFILES = "results/t3-les-profiles.npz"
T3_LES = "results/t3-les.json"
EXP_S = "data/t3-ercoftac-exp/{}s.dat"
EXP_V = "data/t3-ercoftac-exp/{}v.dat"
EXP_CASES = ("t3a", "t3b", "t3am")
OUT = "results/streak-information.json"
Y_LO, Y_HI = 0.1, 0.7
JHTDB_EVERY = 40
SPREAD = 0.2
R_MAX = 0.99


def band(eta, r):
    """The mean over Y_LO to Y_HI delta_99 of I and of R_uv, with eta the
    height over delta_99."""
    m = (eta >= Y_LO) & (eta <= Y_HI) & np.isfinite(r)
    eta, r = eta[m], np.clip(r[m], -R_MAX, R_MAX)
    i = -0.5 * np.log(1.0 - r**2)
    w = np.trapezoid(np.ones_like(eta), eta)
    return float(np.trapezoid(i, eta) / w), float(np.trapezoid(r, eta) / w)


def delta99(y, U, ue):
    i = int(np.argmax(U >= 0.99 * ue))
    return float(y[i]) if i > 0 else float(y[-1])


def corr(uv, uu, vv):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -uv / np.sqrt(np.maximum(uu, 0.0) * np.maximum(vv, 0.0))


def wu(tag):
    d = WU.format(tag)
    rx = np.loadtxt(f"{d}/Re_x_versus_Re_theta_{tag}.dat")
    out = []
    pattern = (
        f"{d}/y_over_delta_versus_-uv_over_vrmsurms_at_Re_theta_*_{tag}.dat"
    )
    for f in sorted(glob.glob(pattern)):
        rth = float(re.search(r"Re_theta_(\d+)", f).group(1))
        p = np.loadtxt(f)
        out.append(
            (float(np.interp(rth, rx[:, 1], rx[:, 0])), *band(p[:, 0], p[:, 1]))
        )
    return out


def bienner(tag):
    evol = np.loadtxt(BIENNER_EVOL.format(tag), skiprows=BIENNER_SKIP)
    out = []
    for f in sorted(glob.glob(f"{BIENNER_PROF.format(tag)}/profile_*.dat")):
        rth = float(re.search(r"Rtheta_(\d+)", f).group(1))
        with open(f) as fh:
            lines = fh.read().splitlines()
        start = next(
            i
            for i, s in enumerate(lines)
            if s.strip().lower().startswith("y/d99")
        )
        p = np.loadtxt(lines[start + 1 :])
        r = corr(p[:, 7], p[:, 4] ** 2, p[:, 5] ** 2)
        out.append(
            (float(np.interp(rth, evol[:, 1], evol[:, 0])), *band(p[:, 0], r))
        )
    return out


def jhtdb():
    d = load_dns()
    x, y, U = d["x"], d["y"], d["U"]
    out = []
    for i in range(0, len(x), JHTDB_EVERY):
        ue = float(U[:, i].max())
        r = corr(d["uv"][:, i], d["uu"][:, i], d["vv"][:, i])
        eta = y / delta99(y, U[:, i], ue)
        out.append((float(x[i] * ue / d["nu"]), *band(eta, r)))
    return out


def t3(case):
    prof = np.load(T3_PROFILES)
    with open(T3_LES) as f:
        plate = json.load(f)["cases"][case]["plate"]
    out = []
    k = 0
    while f"{case}/{k}/x" in prof:
        p = {
            n: prof[f"{case}/{k}/{n}"]
            for n in ("x", "y", "U", "uu", "vv", "uv", "ue")
        }
        eta = p["y"] / delta99(p["y"], p["U"], float(p["ue"]))
        r = corr(p["uv"], p["uu"], p["vv"])
        rex = float(np.interp(p["x"], plate["x"], plate["rex"]))
        out.append((rex, *band(eta, r)))
        k += 1
    return out


def blocks(path):
    with open(path) as f:
        text = f.read()
    return {
        m.group(1)[2:]: m.group(2)
        for m in re.finditer(
            r"RUN NUMBER\s+(\S+)(.*?)(?=RUN NUMBER|\Z)", text, re.DOTALL
        )
    }


def experiment(case):
    single = blocks(EXP_S.format(case))
    out = []
    for run, block in blocks(EXP_V.format(case)).items():
        if run not in single:
            continue
        s = single[run]
        rex = 1e4 * float(re.search(r"X-REYNOLDS NUMBER =\s*([\d.]+)", s)[1])
        d99 = float(re.search(r"\(99\.0%\) =\s*([\d.]+)", s)[1])
        rows = []
        for line in block.splitlines():
            v = line.split()
            if len(v) == 10:
                try:
                    rows.append([float(t) for t in v])
                except ValueError:
                    pass
        r = np.array(rows)
        out.append((rex, *band(r[:, 0] / d99, -r[:, 9])))
    return out


def at_onset(st, rex_on):
    rex = np.array([s[0] for s in st])
    if not rex.min() <= rex_on <= rex.max():
        return None, None
    return (
        float(np.interp(rex_on, rex, [s[1] for s in st])),
        float(np.interp(rex_on, rex, [s[2] for s in st])),
    )


def main():
    with open(INFLOW) as f:
        ref = json.load(f)["flows"]
    with open(WRITE) as f:
        write = json.load(f)["flows"]
    sources = {f"wu-{t.lower()}": lambda t=t: wu(t) for t in WU_TAGS}
    for t in BIENNER_M01:
        sources[f"bienner-{t.lower()}"] = lambda t=t: bienner(t)
    sources["jhtdb"] = jhtdb
    for c in ("t3a", "t3b"):
        sources[c] = lambda c=c: t3(c)
    for c in EXP_CASES:
        sources[f"{c}-exp"] = lambda c=c: experiment(c)
    flows = {}
    for name, get in sources.items():
        st = sorted(get())
        rex_on = ref[name]["re_x_onset"]
        i_on, r_on = at_onset(st, rex_on)
        flows[name] = {
            "group": ref[name]["group"],
            "kind": ref[name]["kind"],
            "tu_ref_percent": ref[name]["tu_ref_percent"],
            "re_x_onset": rex_on,
            "i_onset": i_on,
            "r_uv_onset": r_on,
            "a_onset": write[name]["a_onset"],
            "stations": [{"re_x": s[0], "i": s[1], "r_uv": s[2]} for s in st],
        }
        print(
            name,
            {k: v for k, v in flows[name].items() if k != "stations"},
        )

    both = [
        n
        for n, f in flows.items()
        if f["kind"] == "simulation"
        and f["i_onset"] is not None
        and f["a_onset"] is not None
    ]

    def spread(key):
        v = [flows[n][key] for n in both]
        return float(max(v) / min(v) - 1)

    s_i, s_a = spread("i_onset"), spread("a_onset")
    result = {
        "y_band_over_delta": [Y_LO, Y_HI],
        "spread_tol": SPREAD,
        "flows": flows,
        "compared": both,
        "i_spread": s_i,
        "a_spread_same_flows": s_a,
        "information_ok": bool(s_i < SPREAD and s_i < s_a),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items() if k != "flows"})


if __name__ == "__main__":
    main()
