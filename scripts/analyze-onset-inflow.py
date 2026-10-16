#!/usr/bin/env python
"""Does bypass onset at a given free-stream intensity depend on whether the
layer grew from a leading edge?

The leading-edge clipping closure transitions T3A and T3B early
(results/t3-rans.json), and neither its threshold nor the free-stream
forcing accumulated by onset (results/onset-dose.json) accounts for it.
The zero-gradient flows begin in three ways: from a Blasius inflow with
turbulence laid over it and no leading edge (Wu et al., Bienner et al.),
from a blunt super-ellipse leading edge (the JHTDB plate,
JHTDBDescription), and from a sharp one (the T3 plates). Nagarajan2007
found a blunter leading edge moves onset upstream. If onset at matched
intensity orders with how the layer begins, a closure whose constant comes
from one kind of start carries that start's onset to the others.

This is descriptive, not a test: the grouping was found after the onset
values of results/onset-dose.json had been looked at, and in a first look
the JHTDB plate seemed to lie between the other two groups. Each flow's
onset is the
C_f minimum ahead of the first station where C_f sqrt(Re_x) reaches twice
Blasius's, the rule of scripts/analyze-onset-dose.py, and its measure is
the closure's rule,

    S0 = Tu_ref^2 Re_x at onset, in percent squared,

with Tu_ref the intensity at Re_x = RE_REF, where Re_theta is 80 in
Blasius and where Wu et al.'s inflow starts, so that a decay between a
leading edge and that station is not counted against the leading-edge
flows; S0 with the leading-edge intensity is reported too. The groups:

  Blasius inflow  Wu et al.'s DNS (five intensities) and Bienner et al.'s
                  LES in air at Mach 0.1 (two integral length scales a
                  factor of seven apart, about 4 percent; Re_x from the
                  inflow at 10^4); Tu_ref is the first station's
  blunt edge      the JHTDB plate, Tu as in scripts/analyze-onset-dose.py,
                  with the documented leading-edge intensity at Re_x = 0
  sharp edge      the T3A and T3B LES (Zemmour2026, the solver of
                  Bienner2024), and as an out-of-sample check the T3A, T3B
                  and T3A- experiments (Roach1990), whose summary files
                  give C_f and the free-stream intensity by station; the
                  first T3A- station is at Re_x 1.2e5, so with the
                  intensity decaying its S0 is a lower bound

Each sharp- and blunt-edge flow's S0 is divided by the Blasius-inflow
flows' S0 interpolated in ln Tu_ref (held at the end values outside
their range), and each sharp-edge flow's ratio by the JHTDB plate's, the
offset a closure with the JHTDB plate's constant carries to a sharp edge
at matched intensity. Two checks on Bienner et al.'s LES, which differ from the
T3 LES in neither solver nor group, bound what else could separate them:
its Mach 0.9 air cases against its Mach 0.1 ones (the T3A and T3B LES are
at Mach 0.5), and its two length scales against each other. The T3C
plates are left out, their onset being moved by the pressure gradient
too, as are Bienner et al.'s organic-vapor cases.

Outputs
-------
results/onset-inflow.json
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np

BIENNER = "data/bienner-bypass-les/evol_streamwise/stats_vs_x_{}.dat"
BIENNER_SKIP = 19
BIENNER_M01 = ("low_Lf_Tu4_Air_M0p1", "high_Lf_Tu4_Air_M0p1")
BIENNER_M09 = ("low_Lf_Tu4_Air_M0p9", "high_Lf_Tu4_Air_M0p9")
EXP = "data/t3-ercoftac-exp/{}y.dat"
EXP_CASES = ("t3a", "t3b", "t3am")
OUT = "results/onset-inflow.json"
RE_REF = (80.0 / 0.664) ** 2


def dose_module():
    spec = importlib.util.spec_from_file_location(
        "onset_dose", "scripts/analyze-onset-dose.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def row(group, rex_on, tu_ref, tu_le):
    return {
        "group": group,
        "re_x_onset": float(rex_on),
        "tu_ref_percent": float(tu_ref),
        "tu_le_percent": None if tu_le is None else float(tu_le),
        "s0": float(tu_ref**2 * rex_on),
        "s0_le": None if tu_le is None else float(tu_le**2 * rex_on),
    }


def bienner(dose, tag):
    a = np.loadtxt(BIENNER.format(tag), skiprows=BIENNER_SKIP)
    rex, cf, tu = a[:, 0], a[:, 8], 100.0 * a[:, 11]
    return row("blasius", rex[dose.onset_index(rex, cf)], tu[0], None)


def experiment(dose, case):
    a = np.loadtxt(EXP.format(case), skiprows=18, usecols=(2, 4, 8))
    rex, cf, tu = a[:, 0], a[:, 1], a[:, 2]
    r = row("sharp", rex[dose.onset_index(rex, cf)], tu[0], None)
    r["kind"] = "experiment"
    r["first_station_re_x"] = float(rex[0])
    return r


def main():
    dose = dose_module()
    with open(dose.SETUP) as f:
        setup = json.load(f)["cases"]
    rows = {}
    for tag in dose.TAGS:
        rex_on, _, tu_le, _, _ = dose.wu(tag)
        rows[f"wu-{tag.lower()}"] = row("blasius", rex_on, tu_le, None)
    for tag in BIENNER_M01:
        rows[f"bienner-{tag.lower()}"] = bienner(dose, tag)
    rex_on, _, tu_le, rex, tu = dose.jhtdb()
    tu_ref = np.interp(RE_REF, np.r_[0.0, rex], np.r_[tu_le, tu])
    rows["jhtdb"] = row("blunt", rex_on, tu_ref, tu_le)
    for c in dose.T3:
        rex_on, _, tu_le, rex, tu = dose.t3(c, setup)
        rows[c] = row("sharp", rex_on, np.interp(RE_REF, rex, tu), tu_le)
    for c in EXP_CASES:
        rows[f"{c}-exp"] = experiment(dose, c)
    for r in rows.values():
        r.setdefault("kind", "simulation")

    base = sorted(
        (r["tu_ref_percent"], r["s0"])
        for r in rows.values()
        if r["group"] == "blasius"
    )
    tu_b, s0 = (np.array(v) for v in zip(*base))
    for r in rows.values():
        if r["group"] != "blasius":
            ref = np.interp(np.log(r["tu_ref_percent"]), np.log(tu_b), s0)
            r["s0_over_blasius"] = float(r["s0"] / ref)
    for name, r in rows.items():
        print(name, r)

    def over(names, key="s0_over_blasius"):
        return [rows[n][key] for n in names]

    m09 = {t: bienner(dose, t)["s0"] for t in BIENNER_M09}
    m01 = {t: rows[f"bienner-{t.lower()}"]["s0"] for t in BIENNER_M01}
    result = {
        "re_x_ref": RE_REF,
        "flows": rows,
        "blasius_s0_range": [float(s0.min()), float(s0.max())],
        "blunt_s0_over_blasius": over(["jhtdb"])[0],
        "sharp_les_s0_over_blasius": over(list(dose.T3)),
        "sharp_exp_s0_over_blasius": over([f"{c}-exp" for c in EXP_CASES]),
        "sharp_over_blunt": {
            n: r["s0_over_blasius"] / rows["jhtdb"]["s0_over_blasius"]
            for n, r in rows.items()
            if r["group"] == "sharp"
        },
        "bienner_mach_ratio": {
            t1.replace("_M0p1", ""): m09[t9] / m01[t1]
            for t1, t9 in zip(BIENNER_M01, BIENNER_M09)
        },
        "bienner_length_scale_ratio": m01[BIENNER_M01[1]] / m01[BIENNER_M01[0]],
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items() if k != "flows"})


if __name__ == "__main__":
    main()
