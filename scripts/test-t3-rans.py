#!/usr/bin/env python
"""Do SST, Langtry-Menter and the leading-edge clipping closure transition
the ERCOFTAC T3 plates where Zemmour et al.'s LES does?

Every T3 plate is held out: none of the models was fitted to it, and the
leading-edge closure's coefficients, its threshold's reference intensity
included, come from the JHTDB plate alone. The runs (scripts/run-t3.py)
are on the LES's own domains, under free streams fitted to each model's
own decay of the LES's. The LES is used for C_f, onset, the edge velocity
and the free stream only, the measures its check against the JHTDB DNS
supports (results/t3-les-validation.json).

Onset, for the LES and the runs alike, is the C_f minimum ahead of the
peak of C_f sqrt(Re_x) (Re_x on the local edge velocity), from X_SKIP to
the start of the LES's outflow sponge, as scripts/reduce-t3-les.py finds
the LES's. A run has transitioned only if that peak is at least
TRANS_FACTOR times the value at its minimum, or than Blasius's 0.664 where
the minimum is lower (the LES's own lines give 2.7 to 4.7); a run that has
not is scored at the end of that stretch, a lower bound on its error. The
onset ratio is the run's onset x over the LES's.

Amended after the first runs, which a start-up defect spoiled (see
scripts/run-t3.py), and before their reruns: a run already turbulent at
X_SKIP has no laminar minimum, its C_f falling all the way to the peak,
and the rule as first written scored it as never transitioning, at the
sponge. Such a run, with the minimum at the peak and the peak at least
TRANS_FACTOR times Blasius's, is now scored at X_SKIP, again a lower bound
on its error.

Set-up check: a run whose edge velocity, from its wall pressure and its
inlet's total pressure, departs from the LES's by more than UE_TOL anywhere
from X_SKIP to the LES's onset is left out of the onset tests, since it
then sees another pressure gradient. The runs are incompressible, the LES
at Mach 0.5 (T3A, T3B) and 0.2 (T3C).

Tests, fixed before any T3 run:

  transfer   the leading-edge closure's onset ratio within ONSET_TOL
             (as a factor either way) on T3A and T3B, the plates at zero
             pressure gradient, as the JHTDB plate it was built on
  ranking    its mean |ln onset ratio| over the six plates below both
             Langtry-Menter's and SST's
  pressure   which way the leading-edge closure, which has no
             pressure-gradient term, misses on T3C2 to T3C5; two
             predictions, each stated before the runs:
             - local: the sign of ln(onset ratio) is opposite to the sign
               of the LES's acceleration parameter K at its onset, late
               where the layer is decelerating there (T3C2, T3C3, T3C4)
               and early where accelerating (T3C5);
             - energy: it is opposite to the sign of ln(U_e at onset /
               U_e at the leading edge), which is positive on all four,
               so early on all four. This is Pete's reading through the
               mean energy budget: the pressure's work raises the free
               stream's energy between the leading edge and onset, and
               with it the energy the layer can store before it clips,
               which the closure, whose intensity is fixed at the leading
               edge, does not see.
             Each holds if it does on every T3C plate that passes the
             set-up check. Its magnitude is reported, for the derivation
             that follows, as ln(onset ratio) against ln(U_e ratio).

Also reported: each run's C_f error, the mean |C_f / C_f,LES - 1| from
X_SKIP to the sponge; its free stream's intensity at the leading edge
against the LES's; its convergence; and the leading-edge closure's
intensity next to the wall.

Outputs
-------
results/t3-rans.json
"""

from __future__ import annotations

import json
import os

import numpy as np

SETUP = "results/t3-rans-setup.json"
OUT = "results/t3-rans.json"
CASES = ("t3a", "t3b", "t3c2", "t3c3", "t3c4", "t3c5")
ZPG = ("t3a", "t3b")
PG = ("t3c2", "t3c3", "t3c4", "t3c5")
MODELS = ("sst", "lm", "clip-le")
LE_MODEL = "clip-le"
NU = 1.0 / 800.0
X_SKIP = 10.0
TRANS_FACTOR = 2.0
BLASIUS = 0.664
UE_TOL = 0.05
ONSET_TOL = 1.25


def onset(x, cf, ue, x_end):
    """Onset x, and whether the layer transitioned by x_end."""
    m = (x >= X_SKIP) & (x <= x_end)
    x, cf, ue = x[m], cf[m], ue[m]
    g = cf * np.sqrt(np.abs(ue) * x / NU)
    i_turb = int(np.argmax(g))
    i_min = int(np.argmin(cf[: i_turb + 1]))
    if i_min == i_turb and g[i_turb] >= TRANS_FACTOR * BLASIUS:
        # Turbulent from X_SKIP on
        return float(x[0]), True
    done = g[i_turb] >= TRANS_FACTOR * max(g[i_min], BLASIUS)
    return float(x[i_min]), bool(done)


def main():
    with open(SETUP) as f:
        setup = json.load(f)["cases"]
    rows = {}
    for case in CASES:
        s = setup[case]
        les = s["les"]
        lx, lue = np.array(les["x"]), np.array(les["ue"])
        px, pcf = np.array(les["plate_x"]), np.array(les["cf"])
        x_on, x_end = les["x_onset"], s["x_sponge"]
        i_le = int(np.argmin(np.abs(lx)))
        ue_le = float(np.interp(0.0, lx, lue))
        ue_on = float(np.interp(x_on, lx, lue))
        row = {
            "les": {
                "x_onset": x_on,
                "x_sponge": x_end,
                "K_onset": float(np.interp(x_on, lx, les["K"])),
                "ue_ratio": ue_on / ue_le,
                "tu_le": float(les["tu"][i_le]),
            },
            "models": {},
        }
        for key in MODELS:
            path = os.path.join(
                "sim", "cases", f"t3-{case}-{key}", "postProcessing", "t3.json"
            )
            with open(path) as f:
                r = json.load(f)
            x = np.array(r["plate"]["x"])
            cf = np.array(r["plate"]["cf"])
            ue = np.array(r["plate"]["ue"])
            xo, done = onset(x, cf, ue, x_end)
            ratio = (xo if done else x_end) / x_on
            m = (x >= X_SKIP) & (x <= x_on)
            ue_err = float(np.max(np.abs(ue[m] / np.interp(x[m], lx, lue) - 1)))
            m = (x >= X_SKIP) & (x <= x_end)
            cf_err = float(
                np.mean(np.abs(cf[m] / np.interp(x[m], px, pcf) - 1))
            )
            fs = r["free_stream"]
            fx, fk, fu = (np.array(fs[n]) for n in ("x", "k", "U"))
            j = int(np.argmin(np.abs(fx)))
            mrow = {
                "transitioned": done,
                "x_onset": xo if done else None,
                "onset_ratio": float(ratio),
                "ln_onset_ratio": float(np.log(ratio)),
                "ue_err": ue_err,
                "setup_ok": ue_err <= UE_TOL,
                "cf_err": cf_err,
                "tu_le": float(100 * np.sqrt(2 * fk[j] / 3) / fu[j]),
                "tu_inlet": r["tu_inlet_percent"],
                "decay_fit_log_rms": r["decay_fit_log_rms"],
                "iterations": r["iterations"],
                "converged": r["converged"],
                "settled": r["settled"],
            }
            if "tule_wall" in r:
                tx = np.array(r["plate"]["x"])
                tw = np.array(r["tule_wall"])
                mrow["tule_wall_at_skip"] = float(np.interp(X_SKIP, tx, tw))
            row["models"][key] = mrow
        rows[case] = row
        print(
            case,
            {k: round(v["onset_ratio"], 3) for k, v in row["models"].items()},
            {k: round(v["ue_err"], 3) for k, v in row["models"].items()},
        )

    def ok(case, key):
        return rows[case]["models"][key]["setup_ok"]

    def lr(case, key):
        return rows[case]["models"][key]["ln_onset_ratio"]

    transfer = all(
        ok(c, LE_MODEL) and abs(lr(c, LE_MODEL)) <= np.log(ONSET_TOL)
        for c in ZPG
    )
    mean_abs = {
        key: float(np.mean([abs(lr(c, key)) for c in CASES if ok(c, key)]))
        for key in MODELS
    }
    ranking = all(
        mean_abs[LE_MODEL] < mean_abs[k] for k in MODELS if k != LE_MODEL
    )
    tested = [c for c in PG if ok(c, LE_MODEL)]
    local = [
        np.sign(lr(c, LE_MODEL)) == -np.sign(rows[c]["les"]["K_onset"])
        for c in tested
    ]
    energy = [
        np.sign(lr(c, LE_MODEL)) == -np.sign(np.log(rows[c]["les"]["ue_ratio"]))
        for c in tested
    ]
    result = {
        "onset_tol": ONSET_TOL,
        "ue_tol": UE_TOL,
        "x_skip": X_SKIP,
        "trans_factor": TRANS_FACTOR,
        "cases": rows,
        "mean_abs_ln_onset_ratio": mean_abs,
        "pressure_tested": tested,
        "pressure_magnitude": {
            c: {
                "ln_onset_ratio": lr(c, LE_MODEL),
                "ln_ue_ratio": float(np.log(rows[c]["les"]["ue_ratio"])),
            }
            for c in PG
        },
        "transfer_passes": bool(transfer),
        "ranking_passes": bool(ranking),
        "local_passes": bool(tested) and all(bool(v) for v in local),
        "energy_passes": bool(tested) and all(bool(v) for v in energy),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items() if k != "cases"})


if __name__ == "__main__":
    main()
