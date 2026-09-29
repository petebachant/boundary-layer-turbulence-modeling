#!/usr/bin/env python
"""Inlet omega that makes each model's free stream follow the measured decay
from the inlet to onset.

The inlets fit each model's inlet omega to the measured free-stream decay
over a stretch that runs well past onset (x up to 400 on the JHTDB plate,
the first 40 percent of Wu et al.'s domains), and with SST's published
outer beta_2 = 0.0828 the decay exponent, about 1.09, cannot follow a
measured one of 1.7 and up over that long a stretch; on the plate
Langtry-Menter's free-stream k is already about a fifth low at onset. What
matters for transition is the free stream before onset, and over that
shorter stretch the published beta_2 can follow it with the right inlet
omega. So refit only the inlet omega, with beta_2 = 0.0828, from each
flow's inlet to its DNS onset (fit_inlet_omega of
scripts/make-inlet-profiles.py), leaving the model's constants as
published.

Outputs
-------
results/matched-decay.json
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

from pypkg.dns_case import load_dns

OUT = "results/matched-decay.json"
BETA = 0.0828
BETA_STAR = 0.09
PLATE_Y_FS = 20.0
ROOT = "data/wu-bypass-transition"


def decay(k0, w0, x, x0, ue):
    fac = 1.0 + BETA * w0 * (x - x0) / ue
    return k0 * fac ** (-BETA_STAR / BETA)


def fit(fit_inlet_omega, x, k, ue, x0, x1):
    k0, w0, _ = fit_inlet_omega(x, k, ue, x0, x1, BETA, BETA_STAR)
    m = (x >= x0) & (x <= x1)
    kf = decay(k0, w0, x[m], x[m][0], ue)
    rel = kf / k[m] - 1
    return {"omega_inlet": w0, "k_inlet_fit": k0,
            "fit_rel_err_mean": float(np.mean(np.abs(rel))),
            "fit_rel_err_at_onset": float(rel[-1])}


def main():
    spec = importlib.util.spec_from_file_location(
        "mip", "scripts/make-inlet-profiles.py")
    mip = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mip)
    out = {"beta2": BETA}
    d = load_dns()
    with open("results/inlet-profiles.json") as f:
        prof = json.load(f)
    with open("results/transition-mechanics.json") as f:
        x_on = json.load(f)["x_transition_onset"]
    j = int(np.argmin(np.abs(d["y"] - PLATE_Y_FS)))
    ue = float(np.mean(d["U"][j, :]))
    out["plate"] = fit(mip.fit_inlet_omega, d["x"], d["k"][j, :], ue,
                       prof["x_inlet"], x_on)
    out["plate"]["omega_inlet_before"] = prof["omega_inlet_by_beta"]["0.0828"]["omega_inlet"]
    with open("results/bypass-onset.json") as f:
        onset = json.load(f)["cases"]
    with open("results/wu-openfoam-inlets.json") as f:
        inlets = json.load(f)
    for tag, inl in inlets.items():
        dd = os.path.join(ROOT, f"stats_{tag}")
        urms = np.loadtxt(f"{dd}/Re_x_versus_urms_at_y_500_to_800_theta_0_{tag}.dat")
        x = urms[:, 0] / inl["re_theta_0"]
        k = 1.5 * urms[:, 1] ** 2
        x_end = min(onset[tag]["re_x_onset"] / inl["re_theta_0"],
                    inl["x_outlet"])
        out[tag] = fit(mip.fit_inlet_omega, x, k, 1.0, inl["x_inlet"], x_end)
        out[tag]["omega_inlet_before"] = inl["omega_inlet_by_beta"]["0.0828"]["omega_inlet"]
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    for k_, v in out.items():
        print(k_, v if not isinstance(v, dict)
              else {a: round(b, 5) for a, b in v.items()})


if __name__ == "__main__":
    main()
