#!/usr/bin/env python
"""Inlets and domains for running Wu et al.'s bypass-transition flows in
OpenFOAM, in the same form as results/inlet-profiles.json for the JHTDB
plate, so sim/run.py can set them up the same way.

Each flow is a zero-pressure-gradient plate in units of the inlet momentum
thickness theta_0 and the free-stream speed, so x = Re_x / Re_theta_0 and
nu = 1 / Re_theta_0, as in pypkg/cases/wu_bypass.py:

  inlet     a Blasius profile with theta = 1 at the first station where both
            C_f and the free stream are given
  outlet    the last station where C_f, Re_theta and the free stream are all
            given
  top       Y_TOP inlet momentum thicknesses above the plate, inside the band
            (500-800) where the dataset measures the free stream, so the
            free stream the model decays is the one measured
  k         the free-stream k = 1.5 u_rms^2 (isotropic, which the dataset's
            v_rms and w_rms bear out above the layer) at the inlet, times
            (U / U_inf)^2 inside the layer
  omega     fitted per destruction coefficient beta, exactly as for the
            plate (scripts/make-inlet-profiles.py), to the measured
            free-stream decay over FIT_SPAN of the domain after the inlet;
            kkLOmega's free stream uses beta = 0.92, betaStar = 1
  ReThetat  Langtry and Menter's zero-pressure-gradient correlation at the
            inlet intensity

Outputs
-------
results/wu-openfoam-inlets.json, keyed by flow
"""

from __future__ import annotations

import importlib.util
import json
import os

import numpy as np

from pypkg.cases.wu_bypass import CASES, blasius

ROOT = "data/wu-bypass-transition"
OUT = "results/wu-openfoam-inlets.json"
COEFFS = "results/clip-k-gamma-coeffs.json"
Y_TOP = 650.0
NY_PROFILE = 400
FIT_SPAN = 0.4
BETAS = (0.0828, 0.09, 0.075, 0.06, 0.05, 0.047, 0.04)


def _load_fitter():
    spec = importlib.util.spec_from_file_location(
        "make_inlet_profiles", "scripts/make-inlet-profiles.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.fit_inlet_omega, mod.BETA_STAR


def lm_re_theta_t(tu):
    if tu <= 1.3:
        return 1173.51 - 589.428 * tu + 0.2196 / tu ** 2
    return 331.50 * (tu - 0.5658) ** -0.671


def main():
    fit_inlet_omega, beta_star = _load_fitter()
    with open(COEFFS) as f:
        clip_beta = float(json.load(f)["openfoam_coeffs"]["beta"])
    betas = sorted(set(BETAS) | {round(clip_beta, 4)})
    fp = blasius()
    out = {}
    for tag, tu_in in CASES.items():
        d = os.path.join(ROOT, f"stats_{tag}")
        load = lambda n: np.loadtxt(f"{d}/{n}_{tag}.dat")  # noqa: E731
        re_theta0 = float(load("Re_theta_versus_delta")[0, 0])
        cf = load("Re_x_versus_cf")
        rth = load("Re_x_versus_Re_theta")
        urms = load("Re_x_versus_urms_at_y_500_to_800_theta_0")
        to_x = lambda rex: rex / re_theta0  # noqa: E731
        x_fs, k_fs = to_x(urms[:, 0]), 1.5 * urms[:, 1] ** 2
        x_in = float(max(to_x(cf[0, 0]), x_fs[0]))
        x_out = float(min(to_x(cf[-1, 0]), to_x(rth[-1, 0]), x_fs[-1]))
        k_in = float(np.interp(x_in, x_fs, k_fs))
        tu = float(100.0 * np.sqrt(2.0 * k_in / 3.0))
        x1 = x_in + FIT_SPAN * (x_out - x_in)
        omega_by_beta = {}
        for b in betas:
            _, w, c = fit_inlet_omega(x_fs, k_fs, 1.0, x_in, x1, b)
            omega_by_beta[f"{b:.4f}"] = {"omega_inlet": w, "fit_cost": c,
                                         "beta_star": beta_star}
        _, w, c = fit_inlet_omega(x_fs, k_fs, 1.0, x_in, x1, 0.92,
                                  betaStar=1.0)
        omega_by_beta["0.9200"] = {"omega_inlet": w, "fit_cost": c,
                                   "beta_star": 1.0}
        # Blasius with theta = 1: y/theta = eta/0.664
        y = np.linspace(0.0, Y_TOP, NY_PROFILE)
        U = fp(0.664 * y)
        U[0] = 0.0
        k = k_in * U ** 2
        out[tag] = {
            "tu_inlet_percent_nominal": tu_in,
            "re_theta_0": re_theta0, "nu": 1.0 / re_theta0,
            "ReThetat_inlet": float(lm_re_theta_t(tu)),
            "x_inlet": x_in, "x_outlet": x_out, "y_max": Y_TOP,
            "Ue_inlet": 1.0,
            "delta99_inlet": float(np.interp(0.99, U[:60], y[:60])),
            "Tu_inlet_percent": tu,
            "omega_inlet": omega_by_beta["0.0828"]["omega_inlet"],
            "omega_fit_beta": 0.0828,
            "omega_inlet_by_beta": omega_by_beta,
            "note": ("Blasius inlet with theta = 1 in inlet momentum "
                     "thicknesses; free stream from Wu et al.'s measured "
                     "u_rms 500-800 theta_0 above the plate."),
            "y": y.tolist(), "U": U.tolist(), "k": k.tolist(),
        }
        print(f"{tag}: x {x_in:.0f}-{x_out:.0f}, Tu_in {tu:.2f}%, "
              f"ReThetat {lm_re_theta_t(tu):.0f}, omega(0.0828) "
              f"{omega_by_beta['0.0828']['omega_inlet']:.4g}, "
              f"omega(kkl) {w:.4g}")
    with open(OUT, "w") as f:
        json.dump(out, f)
        f.write("\n")


if __name__ == "__main__":
    main()
