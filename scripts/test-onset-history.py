#!/usr/bin/env python
"""Does the streak onset rule survive in a form a closure can transport,
by integrating the free stream's forcing rather than reading the inlet?

The streak rule, Re_x,onset = K^2 / Tu_in^2, predicts Wu et al.'s onsets
better than Langtry-Menter's correlation, but it needs the inlet intensity,
which a local closure does not have; at the local intensity it fails
(results/onset-laws.json). Between the two: lift-up makes the streaks'
energy grow with the forcing they have received, A^2 ~ Tu^2 Re_x at a
constant intensity, so with a decaying free stream the natural form is the
accumulated forcing

    I(Re_x) = integral from 0 to Re_x of Tu(s)^2 ds,

with onset where I first reaches K^2. At a constant intensity it is the
inlet rule exactly; where the free stream decays it puts onset later. A
closure can carry I as a transported scalar with source U Tu^2 / nu, with
no inlet to remember.

Upstream of each dataset's first station the intensity is taken as the
first station's, which is what the inlet rule assumes everywhere. Tu is
the free-stream intensity of results/onset-laws.json: on Wu et al.'s flows
u_rms at 500 to 800 inlet momentum thicknesses, on the plate the
streamwise intensity 1.5 to 2.5 delta_99 above the wall.

Test, fixed before any prediction was computed: with K set on the JHTDB
plate alone, the history rule predicts Wu et al.'s onset Re_theta with a
lower log-rms error than Langtry-Menter's correlation applied the same
way. Also reported: its error against the inlet rule's, K fitted to Wu et
al.'s flows leaving each out, and the onset Re_theta ratio it predicts
between Bienner et al.'s two length scales against the LES's. Tu_in
throughout is the free-stream intensity at each dataset's first station,
measured as above.

Added after that test failed, with its own test fixed before it was run:
the local form of results/onset-laws.json failed either because it read
the local intensity or because it swapped Re_x for Re_theta. Applying it
at the inlet intensity separates the two:

    Re_theta,onset = C / Tu_in,

with C set so the plate's onset is reproduced at its leading-edge
intensity (results/plate-leading-edge-intensity.json). It passes if it predicts
Wu et al.'s onsets with a lower log-rms error than Langtry-Menter's
correlation; its onset on Bienner et al.'s two runs is also reported. If
it does, a closure needs only the intensity at the
leading edge carried into the layer, since Re_theta is local.

Outputs
-------
results/onset-history.json
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np

OUT = "results/onset-history.json"
LAWS = "results/onset-laws.json"
ONSET = "results/bypass-onset.json"
MECHANICS = "results/transition-mechanics.json"
LENGTH_SCALE = "results/length-scale-onset.json"
PLATE_LE = "results/plate-leading-edge-intensity.json"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def forcing(rex, tu):
    """Accumulated forcing at each station, from the leading edge."""
    head = tu[0] ** 2 * rex[0]
    seg = 0.5 * (tu[1:] ** 2 + tu[:-1] ** 2) * np.diff(rex)
    return head + np.concatenate(([0.0], np.cumsum(seg)))


def predict(rex, th, tu, level):
    """Onset Re_theta where the accumulated forcing first reaches level."""
    acc = forcing(rex, tu)
    if acc[-1] < level:
        return None
    if acc[0] >= level:
        # Upstream of the first station, where the intensity is constant
        r = level / tu[0] ** 2
        return float(th[0] * np.sqrt(r / rex[0]))
    return float(np.interp(np.interp(level, acc, rex), rex, th))


def main():
    laws = load("laws", "scripts/compare-onset-laws.py")
    les = load("les", "scripts/analyze-length-scale-onset.py")
    from pypkg.dns_case import load_dns

    with open(ONSET) as f:
        onset = json.load(f)["cases"]
    with open(MECHANICS) as f:
        x_on = json.load(f)["x_transition_onset"]
    with open(LAWS) as f:
        prior = json.load(f)
    with open(LENGTH_SCALE) as f:
        ls = json.load(f)
    nu = load_dns()["nu"]
    rex_p, th_p, tu_p = laws.plate()
    k2 = float(np.interp(x_on / nu, rex_p, forcing(rex_p, tu_p)))
    # The plate's leading-edge intensity: its first station's, scaled by
    # the ratio of the leading edge's to the database inlet's that the
    # DNS's documentation gives. Until 2026-10-06 this was the first
    # station's alone
    with open(PLATE_LE) as f:
        le_over_inlet = json.load(f)["le_over_inlet"]
    c_inlet = float(np.interp(x_on / nu, rex_p, th_p) * tu_p[0] * le_over_inlet)
    rows = {}
    for tag, c in onset.items():
        rex, th, tu = laws.wu_flow(tag)
        acc = forcing(rex, tu)
        target = c_inlet / tu[0]
        rx_on = c["re_x_onset"]
        rows[tag] = {
            "tu_inlet_percent": c["tu_inlet_percent"],
            "dns": c["re_theta_onset"],
            "history": predict(rex, th, tu, k2),
            "theta_inlet": (
                float(target) if th[0] <= target <= th[-1] else None
            ),
            "streak": prior["cases"][tag]["streak"],
            "langtry_menter": prior["cases"][tag]["langtry_menter"],
            "forcing_at_dns_onset": (
                float(np.interp(rx_on, rex, acc)) if rx_on else None
            ),
            # What the decay costs: the forcing accumulated by the DNS onset
            # over what the inlet intensity alone would have given
            "forcing_over_inlet_at_onset": (
                float(np.interp(rx_on, rex, acc) / (tu[0] ** 2 * rx_on))
                if rx_on
                else None
            ),
        }
    # K^2 leaving each Wu flow out: geometric mean of the others' forcing
    # at their DNS onsets
    for tag, r in rows.items():
        ks = [
            rows[t]["forcing_at_dns_onset"]
            for t in rows
            if t != tag and rows[t]["forcing_at_dns_onset"]
        ]
        rex, th, tu = laws.wu_flow(tag)
        r["history_loo"] = predict(
            rex, th, tu, float(np.exp(np.mean(np.log(ks))))
        )
    errs = {}
    for law in (
        "langtry_menter",
        "streak",
        "history",
        "history_loo",
        "theta_inlet",
    ):
        ok = [
            (r[law], r["dns"])
            for r in rows.values()
            if r[law] is not None and r["dns"] is not None
        ]
        errs[law] = {
            "log_rms": laws.log_err(*zip(*ok)) if ok else None,
            "n": len(ok),
        }
    # Bienner et al.'s LES at two length scales: the history rule sees them
    # only through their different decay, as Langtry-Menter's correlation
    # does in results/length-scale-onset.json
    bienner = {}
    for name, run in les.RUNS.items():
        d = les.load(run)
        tu = 100.0 * d["tu"]
        bienner[name] = {
            "history": predict(d["re_x"], d["re_theta"], tu, k2),
            "theta_inlet": float(c_inlet / tu[0]),
            "les": ls["runs"][name]["re_theta_onset"],
            "forcing_at_les_onset": float(
                np.interp(
                    ls["runs"][name]["re_x_onset"],
                    d["re_x"],
                    forcing(d["re_x"], tu),
                )
            ),
        }
    sm, lg = bienner["small"], bienner["large"]
    result = {
        "k2_plate": k2,
        "c_inlet": c_inlet,
        "plate_forcing_over_inlet_at_onset": k2 / (tu_p[0] ** 2 * x_on / nu),
        "cases": rows,
        "errors": errs,
        "bienner": bienner,
        "bienner_history_ratio": (
            lg["history"] / sm["history"]
            if lg["history"] and sm["history"]
            else None
        ),
        "bienner_les_ratio": ls["les_onset_ratio"],
        "bienner_lm_ratio": ls["lm_onset_ratio"],
        "passes": bool(
            errs["history"]["log_rms"] is not None
            and errs["langtry_menter"]["log_rms"] is not None
            and errs["history"]["n"] == errs["langtry_menter"]["n"]
            and errs["history"]["log_rms"] < errs["langtry_menter"]["log_rms"]
        ),
        "theta_inlet_passes": bool(
            errs["theta_inlet"]["log_rms"] is not None
            and errs["theta_inlet"]["n"] == errs["langtry_menter"]["n"]
            and errs["theta_inlet"]["log_rms"]
            < errs["langtry_menter"]["log_rms"]
        ),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    for t, r in rows.items():
        print(
            t,
            {
                k: (round(v) if isinstance(v, float) and v > 10 else v)
                for k, v in r.items()
            },
        )
    print(
        json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=1)
    )


if __name__ == "__main__":
    main()
