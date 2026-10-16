#!/usr/bin/env python
"""Where along the plate does the leading-edge clipping closure's early
onset on T3A and T3B arise, against the JHTDB plate it was built on?

On the JHTDB plate with its nose the closure's onset is within a few
percent of the DNS's (results/nosed-plate.json), on T3A and T3B it is at
0.57 and 0.37 of the LES's (results/t3-rans.json), and neither the
threshold, the forcing accumulated by onset (results/onset-dose.json) nor
how the layer begins (results/onset-inflow.json) accounts for that. The
two tests also measure onset differently: the nosed plate's as Re_theta
where C_f has risen F of the way from its minimum to its peak
(scripts/test-lm-onset-transfer.py), the T3 plates' as x at the C_f
minimum (scripts/test-t3-rans.py).

So the closure's three runs, on the nosed plate and on T3A and T3B, are
traced along the plate, and each run and its reference (the JHTDB DNS, the
T3 LES) is put on both measures in x. Along each run, at stations DX
apart, the trace holds Re_theta over Blasius's, the largest y^2 |dU/dy| /
nu across the layer (the closure's Re_v with the vorticity taken as the
wall-normal shear), the threshold LambdaC TuRef / TuLE that Re_v must pass,
with TuLE in the cell next to the wall as the closure reads it, and the
largest activation fraction gamma. The crossing is where Re_v first
reaches the threshold; ahead of it gamma is not produced.

A T3 plate's onset ratio under the C_f-minimum measure then splits, as an
identity, into three factors against the nosed plate:

  measure   the nosed plate's own ratio under that measure, which the
            closure was not built on
  scaling   the crossing's move from the nosed plate to the T3 plate over
            the reference onset's move
  lag       the onset's distance behind the crossing on the T3 plate over
            that on the nosed plate

The first is a property of the closure on the flow it was built on; the
second compares how its crossing and the references' onsets move with the
intensity, which by the closure's form goes as TuLE^-2 in Re_x while the
layer is laminar (Re_v grows as sqrt(Re_x), the threshold as 1 / TuLE);
the third is how fast gamma, once produced, raises C_f. Each plate's
crossing exponent, ln of the crossing's x ratio over ln of the TuLE
ratio, and the references', ln of the onset's x ratio over ln of the
leading-edge intensity ratio, are reported with them; the free stream is
near unit speed on every plate, so x stands for Re_x.

The F-rise measure here has no minimum rise, unlike the nosed plate's
test, since on T3A the closure's C_f rises to only 1.4 times its minimum;
each run's peak over its minimum is reported. Windows are the tests': x
from 30 to X_END_NOSED on the nosed plate and its DNS, from the T3 test's
X_SKIP to the LES's sponge on the T3 plates.

Added after the split was first run, which put a fifth of the T3A bias
on the measure: the same closure on Wu et al.'s five flows, whose runs
start from their Blasius inflow (the clip-wall stage), scored on both
measures in Re_x from the stations they sample, against the DNS's C_f
(the F-rise level has no minimum rise here either). With every run and
reference, S0 = Tu_LE^2 Re_x at onset (the closure's rule; Tu_LE the
reference's leading-edge or first-station intensity, as in
scripts/analyze-onset-dose.py, and x / nu for Re_x on the nosed and T3
plates) is reported for the run and the reference under the C_f-minimum
measure, so each onset ratio reads as the run's S0 over the reference's.

The runs' fields are read from the latest time directory the sims stages
leave in each case; only their postProcessing is tracked, so this stage
runs after those stages and fails if the fields are absent.

Outputs
-------
results/clip-le-onset-bias.json
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import re

import numpy as np
import pandas as pd

from pypkg.cases import openfoam as of
from pypkg.dns_case import bl_metrics, load_dns

CASES = {
    "nosed": "sim/cases/nosed-plate-clip-le",
    "t3a": "sim/cases/t3-t3a-clip-le",
    "t3b": "sim/cases/t3-t3b-clip-le",
}
WU_CASE = "sim/cases/clip-wall-{}"
T3_SETUP = "results/t3-rans-setup.json"
T3_RESULT = "results/t3-rans.json"
PLATE_LE = "results/plate-leading-edge-intensity.json"
OUT = "results/clip-le-onset-bias.json"
NU = 1.0 / 800.0
X_END_NOSED = 600.0
DX = 5.0
#: Cells whose centers are this close to the plate are the wall row
Y_WALL = 0.006
#: A column must reach this high to give the free stream
Y_TOP = 25.0
FIELDS = ("C", "U", "gamma", "TuLE")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def coeffs(case_dir):
    """LambdaC and TuRef from the run's own turbulenceProperties."""
    path = os.path.join(case_dir, "constant", "turbulenceProperties")
    with open(path) as f:
        words = f.read().replace(";", " ").split()
    return tuple(float(words[words.index(k) + 1]) for k in ("LambdaC", "TuRef"))


def read_case(case_dir):
    t = of.latest_time_dir(case_dir)
    return {n: of.read_field(os.path.join(case_dir, t, n)) for n in FIELDS}


def column(f, xs):
    """The full-height column of cells nearest xs over the flat plate, by
    increasing y."""
    C = f["C"]
    w = (C[:, 1] > 0) & (C[:, 1] < Y_WALL)
    xw = np.unique(np.round(C[w, 0], 4))
    for i in np.argsort(np.abs(xw - xs)):
        x0 = xw[i]
        dx = np.diff(xw)[min(i, len(xw) - 2)]
        m = (np.abs(C[:, 0] - x0) < 0.4 * dx) & (C[:, 1] > 0)
        if C[m, 1].max() > Y_TOP:
            o = np.argsort(C[m, 1])
            return x0, {n: v[m][o] for n, v in f.items()}
    raise ValueError(f"no full column near x = {xs}")


def trace(case_dir, x0, x1):
    lam_c, tu_ref = coeffs(case_dir)
    f = read_case(case_dir)
    rows = []
    for xs in np.arange(x0, x1 + 0.5 * DX, DX):
        x, c = column(f, xs)
        y = np.r_[0.0, c["C"][:, 1]]
        u = np.r_[0.0, c["U"][:, 0]]
        cf, th, _ = bl_metrics(y, u, None, NU)
        ue = float(u.max())
        j = int(np.argmax(u))
        rev = y[:j] ** 2 * np.abs(np.gradient(u, y)[:j]) / NU
        rex = x * ue / NU
        rows.append(
            {
                "x": float(x),
                "re_x": float(rex),
                "cf": float(cf),
                "re_theta_over_blasius": float(th * ue / NU)
                / (0.664 * np.sqrt(rex)),
                "re_v_max": float(rev.max()),
                "threshold": float(lam_c * tu_ref / c["TuLE"][0]),
                "tu_le_wall": float(c["TuLE"][0]),
                "gamma_max": float(c["gamma"][:j].max()),
            }
        )
    return rows


def crossing(rows):
    """x, Re_x and TuLE where Re_v first reaches the threshold."""
    g = np.array([r["re_v_max"] - r["threshold"] for r in rows])
    i = int(np.flatnonzero(g >= 0)[0])
    if i == 0:
        raise ValueError("Re_v is past the threshold at the first station")
    w = -g[i - 1] / (g[i] - g[i - 1])
    return tuple(
        rows[i - 1][k] + w * (rows[i][k] - rows[i - 1][k])
        for k in ("x", "re_x", "tu_le_wall")
    )


def onsets(x, cf, f_rise):
    """x at the C_f minimum and where C_f has risen f_rise of the way
    from it to its peak downstream, with the peak over the minimum."""
    i0 = int(np.argmin(cf))
    peak = float(cf[i0:].max())
    level = cf[i0] + f_rise * (peak - cf[i0])
    j = i0 + int(np.flatnonzero(cf[i0:] >= level)[0])
    w = (level - cf[j - 1]) / (cf[j] - cf[j - 1])
    x_rise = x[j - 1] + w * (x[j] - x[j - 1])
    return float(x[i0]), float(x_rise), peak / float(cf[i0])


def model_cf(case_dir, x0, x1):
    """The run's C_f from its wall row, on the inflow's unit speed; only
    where it is least and where it rises are used, which a slowly varying
    edge velocity hardly moves."""
    f = read_case(case_dir)
    C = f["C"]
    w = np.flatnonzero((C[:, 1] > 0) & (C[:, 1] < Y_WALL))
    w = w[(C[w, 0] >= x0) & (C[w, 0] <= x1)]
    w = w[np.argsort(C[w, 0])]
    return C[w, 0], 2.0 * NU * f["U"][w, 0] / C[w, 1]


def wu_flow(dose, tag, f_rise):
    """The closure's onsets on one of Wu et al.'s flows and the DNS's, in
    Re_x, on both measures."""
    case_dir = WU_CASE.format(tag)
    with open(os.path.join(case_dir, "constant", "transportProperties")) as f:
        nu = float(re.search(r"nu\s+([\d.eE+-]+);", f.read()).group(1))
    root = os.path.join(case_dir, "postProcessing", "sample")
    t = max(
        glob.glob(os.path.join(root, "*")),
        key=lambda p: float(p.rsplit("/", 1)[1]),
    )
    rows = []
    for path in glob.glob(os.path.join(t, "x*.csv")):
        x = float(re.search(r"x(\d+)", os.path.basename(path)).group(1))
        df = pd.read_csv(path)
        y, u = df["y"].to_numpy(), df["U_0"].to_numpy()
        if y[0] > 0:
            y, u = np.r_[0.0, y], np.r_[0.0, u]
        rows.append((x / nu, bl_metrics(y, u, None, nu)[0]))
    rows.sort()
    rex, cf = (np.array(v) for v in zip(*rows))
    on_min = float(rex[dose.onset_index(rex, cf)])
    _, on_rise, rise = onsets(rex, cf, f_rise)
    dns = np.loadtxt(f"{dose.WU}/stats_{tag}/Re_x_versus_cf_{tag}.dat")
    ref_min = float(dns[dose.onset_index(dns[:, 0], dns[:, 1]), 0])
    _, ref_rise, ref_peak = onsets(dns[:, 0], dns[:, 1], f_rise)
    tu_le = dose.wu(tag)[2]
    return {
        "first_station_re_x": float(rex[0]),
        "re_x_onset_cf_min": on_min,
        "re_x_onset_f_rise": on_rise,
        "cf_peak_over_min": rise,
        "ref_re_x_onset_cf_min": ref_min,
        "ref_re_x_onset_f_rise": ref_rise,
        "ref_cf_peak_over_min": ref_peak,
        "ref_tu_le": tu_le,
        "ratio_cf_min": on_min / ref_min,
        "ratio_f_rise": on_rise / ref_rise,
        "s0": tu_le**2 * on_min,
        "ref_s0": tu_le**2 * ref_min,
    }


def main():
    gate = load("gate", "scripts/test-lm-onset-transfer.py")
    dose = load("dose", "scripts/analyze-onset-dose.py")
    t3t = load("t3t", "scripts/test-t3-rans.py")
    with open(T3_SETUP) as fh:
        setup = json.load(fh)["cases"]
    with open(T3_RESULT) as fh:
        t3r = json.load(fh)["cases"]
    with open(PLATE_LE) as fh:
        tu_le_dns = float(json.load(fh)["tu_le_percent"])

    d = load_dns()
    m = (d["x"] >= 30.0) & (d["x"] <= X_END_NOSED)
    xd = d["x"][m]
    cfd = np.array(
        [bl_metrics(d["y"], d["U"][:, i])[0] for i in np.flatnonzero(m)]
    )
    refs = {"nosed": (xd, cfd, tu_le_dns)}
    windows = {"nosed": (30.0, X_END_NOSED)}
    for c in ("t3a", "t3b"):
        les = setup[c]["les"]
        x1 = t3r[c]["les"]["x_sponge"]
        px, pcf = np.array(les["plate_x"]), np.array(les["cf"])
        k = (px >= t3t.X_SKIP) & (px <= x1)
        refs[c] = (px[k], pcf[k], t3r[c]["les"]["tu_le"])
        windows[c] = (t3t.X_SKIP, x1)

    flows = {}
    for name, case_dir in CASES.items():
        x0, x1 = windows[name]
        rows = trace(case_dir, x0, min(x1, 400.0))
        xc, rexc, tuc = crossing(rows)
        xm, cfm = model_cf(case_dir, x0, x1)
        on_min, on_rise, rise = onsets(xm, cfm, gate.F)
        xr, cfr, tu_le_ref = refs[name]
        ref_min, ref_rise, ref_peak = onsets(xr, cfr, gate.F)
        flows[name] = {
            "x_crossing": xc,
            "re_x_crossing": rexc,
            "tu_le_wall_at_crossing": tuc,
            "x_onset_cf_min": on_min,
            "x_onset_f_rise": on_rise,
            "cf_peak_over_min": rise,
            "ref_x_onset_cf_min": ref_min,
            "ref_x_onset_f_rise": ref_rise,
            "ref_cf_peak_over_min": ref_peak,
            "ref_tu_le": tu_le_ref,
            "ratio_cf_min": on_min / ref_min,
            "ratio_f_rise": on_rise / ref_rise,
            "trace": rows,
        }
        print(name, {k: v for k, v in flows[name].items() if k != "trace"})

    n = flows["nosed"]
    split = {}
    for c in ("t3a", "t3b"):
        t = flows[c]
        measure = n["ratio_cf_min"]
        scaling = (t["x_crossing"] / n["x_crossing"]) / (
            t["ref_x_onset_cf_min"] / n["ref_x_onset_cf_min"]
        )
        lag = (t["x_onset_cf_min"] / t["x_crossing"]) / (
            n["x_onset_cf_min"] / n["x_crossing"]
        )
        split[c] = {
            "ratio_cf_min": t["ratio_cf_min"],
            "measure": measure,
            "scaling": scaling,
            "lag": lag,
            "product": measure * scaling * lag,
            "crossing_exponent": float(
                np.log(t["x_crossing"] / n["x_crossing"])
                / np.log(
                    t["tu_le_wall_at_crossing"] / n["tu_le_wall_at_crossing"]
                )
            ),
            "reference_exponent": float(
                np.log(t["ref_x_onset_cf_min"] / n["ref_x_onset_cf_min"])
                / np.log(t["ref_tu_le"] / n["ref_tu_le"])
            ),
        }
    for name, r in flows.items():
        r["s0"] = r["ref_tu_le"] ** 2 * r["x_onset_cf_min"] / NU
        r["ref_s0"] = r["ref_tu_le"] ** 2 * r["ref_x_onset_cf_min"] / NU
    wu = {tag: wu_flow(dose, tag, gate.F) for tag in dose.TAGS}
    for tag, r in wu.items():
        print(tag, r)
    result = {
        "f_rise": gate.F,
        "dx": DX,
        "flows": flows,
        "split": split,
        "wu": wu,
    }
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=1)
        fh.write("\n")
    print(json.dumps(split, indent=1))


if __name__ == "__main__":
    main()
