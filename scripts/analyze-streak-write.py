#!/usr/bin/env python
"""Is the leading-edge flows' later onset in how much of the free stream's
forcing their layers take in, or in how much their layers hold?

Pete's reading of transition as information moving between media
(docs/ideas-log.md, section 7.12): the free stream's disturbances are
written into the laminar layer as streaks, a slow-forgetting store,
which grows until it can hold no more and the flow moves its energy into
turbulence, a fast-forgetting one, from which dissipation erases it. At
matched free-stream intensity the flows that start from a leading edge
transition later than those started from a Blasius inflow, S0 = Tu^2 Re_x
at onset 1.36 to 1.5 times as large (results/onset-inflow.json), and
Wu et al.'s inflow puts its turbulence only above twice the layer's
thickness, so that layer starts empty (Wu2026). The offset is then either
in the writing or in the store:

  write     the streak amplitude A = u'_max / U_e, the largest u_rms
            below Y_MAX delta_99, grows as B Tu_ref sqrt(Re_x) while the
            layer is linear; B is how much of the forcing the layer takes
            in
  capacity  A at onset, how much the store holds when it overflows

Tu_ref and onset are those of results/onset-inflow.json (Tu_ref the
intensity at Re_x = RE_REF, onset the C_f minimum ahead of the first
station where C_f sqrt(Re_x) is twice Blasius's). Each flow's B is the
median of A / (Tu_ref sqrt(Re_x)) over its stations from RE_REF to
LINEAR times its onset Re_x. Flows and their stations:

  Wu et al. (DNS)        the u_rms^+ profiles, at Re_theta 147 and above
  Bienner et al. (LES)   the peak u_rms along the plate, air at Mach 0.1
  JHTDB plate (DNS)      the profiles, every JHTDB_EVERY-th, from x = 30
  T3A, T3B (LES)         the twelve profiles of results/t3-les-profiles.npz
  T3A, T3B, T3A- (exp.)  the single-wire traverses, out of sample

Predictions, fixed before A was computed for any flow but Wu et al.'s
(whose peak u_rms at onset was known, results/streak-headroom.json):

  capacity  A at onset varies by less than SPREAD (max over min, less one)
            across the simulations that have stations either side of
            their onset
  write     on each simulated leading-edge flow (the JHTDB plate, T3A,
            T3B) B is below the Blasius-inflow flows' B at its Tu_ref
            (interpolated in ln Tu_ref, held at the end values outside
            their range), and (B / B_Blasius)^-2, the S0 offset a fixed
            capacity would give, is within a factor WRITE_TOL of its
            measured S0 offset

If both hold, the leading edge's effect is on what the layer takes in, and
a RANS closure must carry how the layer began, not only the intensity it
began under. If capacity fails, the store itself differs. The experiments
are reported on the same measures.

Outputs
-------
results/streak-write.json
"""

from __future__ import annotations

import glob
import json
import re

import numpy as np

from pypkg.dns_case import load_dns

INFLOW = "results/onset-inflow.json"
WU = "data/wu-bypass-transition/stats_{0}"
WU_TAGS = ("WM075", "WM150", "WM225", "WM300", "WM600")
BIENNER = "data/bienner-bypass-les/evol_streamwise/stats_vs_x_{}.dat"
BIENNER_SKIP = 19
BIENNER_M01 = ("low_Lf_Tu4_Air_M0p1", "high_Lf_Tu4_Air_M0p1")
T3_PROFILES = "results/t3-les-profiles.npz"
T3_LES = "results/t3-les.json"
EXP = "data/t3-ercoftac-exp/{}s.dat"
EXP_CASES = ("t3a", "t3b", "t3am")
OUT = "results/streak-write.json"
RE_REF = (80.0 / 0.664) ** 2
Y_MAX = 0.7
LINEAR = 0.7
JHTDB_EVERY = 40
SPREAD = 0.2
WRITE_TOL = 1.25


def delta99(y, U, ue):
    i = int(np.argmax(U >= 0.99 * ue))
    return float(y[i]) if i > 0 else float(y[-1])


def wu(tag):
    d = WU.format(tag)
    utau = np.loadtxt(f"{d}/Re_theta_versus_utau_{tag}.dat")
    rx = np.loadtxt(f"{d}/Re_x_versus_Re_theta_{tag}.dat")
    out = []
    pattern = f"{d}/y_over_delta_versus_urms_plus_at_Re_theta_*_{tag}.dat"
    for f in sorted(glob.glob(pattern)):
        rth = float(re.search(r"Re_theta_(\d+)", f).group(1))
        p = np.loadtxt(f)
        ut = float(np.interp(rth, utau[:, 0], utau[:, 1]))
        a = float(p[p[:, 0] <= Y_MAX, 1].max()) * ut
        out.append((float(np.interp(rth, rx[:, 1], rx[:, 0])), a))
    return out


def bienner(tag):
    a = np.loadtxt(BIENNER.format(tag), skiprows=BIENNER_SKIP)
    return list(zip(a[:, 0].tolist(), a[:, 9].tolist()))


def jhtdb():
    d = load_dns()
    x, y, U, uu = d["x"], d["y"], d["U"], d["uu"]
    out = []
    for i in range(0, len(x), JHTDB_EVERY):
        ue = float(U[:, i].max())
        m = y <= Y_MAX * delta99(y, U[:, i], ue)
        a = float(np.sqrt(max(uu[m, i].max(), 0.0))) / ue
        out.append((float(x[i] * ue / d["nu"]), a))
    return out


def t3(case):
    prof = np.load(T3_PROFILES)
    with open(T3_LES) as f:
        plate = json.load(f)["cases"][case]["plate"]
    out = []
    k = 0
    while f"{case}/{k}/x" in prof:
        p = {n: prof[f"{case}/{k}/{n}"] for n in ("x", "y", "U", "uu", "ue")}
        ue = float(p["ue"])
        m = p["y"] <= Y_MAX * delta99(p["y"], p["U"], ue)
        a = float(np.sqrt(max(p["uu"][m].max(), 0.0))) / ue
        out.append((float(np.interp(p["x"], plate["x"], plate["rex"])), a))
        k += 1
    return out


def experiment(case):
    with open(EXP.format(case)) as f:
        text = f.read()
    out = []
    for block in text.split("AXIAL POSITION")[1:]:
        rex = 1e4 * float(
            re.search(r"X-REYNOLDS NUMBER =\s*([\d.]+)", block)[1]
        )
        rows = []
        for line in block.splitlines():
            v = line.split()
            if len(v) == 10:
                try:
                    rows.append([float(s) for s in v])
                except ValueError:
                    pass
        r = np.array(rows)
        out.append((rex, float(r[r[:, 1] <= Y_MAX, 7].max()) / 100.0))
    return out


def summarize(stations, rex_on, tu_ref):
    st = sorted(stations)
    rex = np.array([s[0] for s in st])
    a = np.array([s[1] for s in st])
    inside = rex.min() <= rex_on <= rex.max()
    m = (rex >= RE_REF) & (rex <= LINEAR * rex_on)
    b = a[m] / (tu_ref / 100.0 * np.sqrt(rex[m]))
    return {
        "stations": [{"re_x": float(r), "a": float(v)} for r, v in st],
        "re_x_onset": rex_on,
        "tu_ref_percent": tu_ref,
        "a_onset": float(np.interp(rex_on, rex, a)) if inside else None,
        "b": float(np.median(b)) if b.size else None,
        "b_n": int(b.size),
        "b_range": [float(b.min()), float(b.max())] if b.size else None,
    }


def main():
    with open(INFLOW) as f:
        ref = json.load(f)["flows"]
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
        r = ref[name]
        flows[name] = summarize(get(), r["re_x_onset"], r["tu_ref_percent"])
        flows[name]["group"] = r["group"]
        flows[name]["kind"] = r["kind"]
        flows[name]["s0_over_blasius"] = r.get("s0_over_blasius")
        print(name, {k: v for k, v in flows[name].items() if k != "stations"})

    sims = [n for n, f in flows.items() if f["kind"] == "simulation"]
    caps = {n: flows[n]["a_onset"] for n in sims if flows[n]["a_onset"]}
    spread = max(caps.values()) / min(caps.values()) - 1
    base = sorted(
        (f["tu_ref_percent"], f["b"])
        for f in flows.values()
        if f["group"] == "blasius" and f["b"] is not None
    )
    tu_b, b_b = (np.array(v) for v in zip(*base))
    for f in flows.values():
        if f["group"] != "blasius" and f["b"] is not None:
            bb = float(
                np.interp(np.log(f["tu_ref_percent"]), np.log(tu_b), b_b)
            )
            f["b_over_blasius"] = f["b"] / bb
            f["s0_offset_from_b"] = (f["b"] / bb) ** -2
            f["s0_offset_ratio"] = f["s0_offset_from_b"] / f["s0_over_blasius"]
    le = ("jhtdb", "t3a", "t3b")
    write_ok = all(
        flows[n]["b"] is not None
        and flows[n]["b_over_blasius"] < 1
        and abs(np.log(flows[n]["s0_offset_ratio"])) <= np.log(WRITE_TOL)
        for n in le
    )
    result = {
        "re_x_ref": RE_REF,
        "y_max_over_delta": Y_MAX,
        "linear": LINEAR,
        "spread_tol": SPREAD,
        "write_tol": WRITE_TOL,
        "flows": flows,
        "capacity": caps,
        "capacity_spread": float(spread),
        "capacity_ok": bool(spread < SPREAD),
        "blasius_b": dict(zip(tu_b.tolist(), b_b.tolist())),
        "write_ok": bool(write_ok),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items() if k != "flows"})
    for n, f in flows.items():
        if "b_over_blasius" in f:
            print(
                n,
                round(f["b_over_blasius"], 3),
                round(f["s0_offset_from_b"], 3),
                f["s0_over_blasius"],
            )


if __name__ == "__main__":
    main()
