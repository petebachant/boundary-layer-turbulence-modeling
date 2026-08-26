#!/usr/bin/env python
"""Write OpenFOAM inlet boundary profiles taken directly from the DNS.

The DNS domain starts at x = 30.218 with the wall already present and a
laminar boundary layer already developed. Prescribing that station as the RANS
inlet means every model solves the same problem the DNS solved, instead of
developing its own boundary layer and its own free-stream decay history over a
run-up the DNS never had.

That matters most for the comparison against transition models. With the old
120-unit run-up, matching the measured free-stream level required an inlet
turbulence intensity of 20 %, which trips a gamma-Re_theta model at the
leading edge and makes it score like a fully turbulent model. Taking the inlet
from the DNS gives Tu = 2.6 % for every model, as measured.

Fields written
--------------
U        streamwise profile, interpolated onto the mesh inlet faces
k        turbulence energy profile
omega    uniform free-stream value fitted from the measured decay just
         downstream of the inlet, since the pointwise estimate
         omega = -U dk/dx / (betaStar k) is noisy and goes negative near the
         wall where k is still growing

Outputs
-------
results/inlet-profiles.json   the profiles, for the record and for plotting
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from py_package.dns_case import load_dns

BETA_STAR = 0.09


def fit_inlet_omega(x, k_fs, ue, x0, x1, beta, betaStar=BETA_STAR):
    """Inlet omega from the measured free-stream decay just after the inlet."""
    m = (x >= x0) & (x <= x1)
    xs, ks = x[m], k_fs[m]

    def resid(p):
        k0, logw = p
        w0 = np.exp(logw)
        fac = 1.0 + beta * w0 * (xs - xs[0]) / ue
        return np.log(np.maximum(k0 * fac ** (-betaStar / beta), 1e-30) / ks)

    best = None
    for w0 in (0.01, 0.05, 0.2, 1.0):
        r = least_squares(resid, [ks[0], np.log(w0)],
                          bounds=([1e-8, np.log(1e-4)], [1.0, np.log(1e3)]))
        if best is None or r.cost < best.cost:
            best = r
    return float(best.x[0]), float(np.exp(best.x[1])), float(best.cost)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.0828,
                    help="omega destruction coefficient used to infer the "
                         "inlet omega from the measured decay")
    ap.add_argument("--fit-to", type=float, default=400.0,
                    help="downstream limit of the decay fit")
    ap.add_argument("--y-freestream", type=float, default=20.0)
    ap.add_argument("--out", default="results/inlet-profiles.json")
    args = ap.parse_args()

    d = load_dns()
    x, y, U, k = d["x"], d["y"], d["U"], d["k"]
    x_in = float(x[0])
    jfs = int(np.argmin(np.abs(y - args.y_freestream)))
    ue = float(np.mean(U[jfs, :]))

    _, omega_inlet, cost = fit_inlet_omega(
        x, k[jfs, :], ue, x_in, args.fit_to, args.beta)

    # omega must be fitted per model, not shared. The inlet omega that
    # reproduces the measured decay depends on the model's own destruction
    # coefficient, since k falls as fac^(-betaStar/beta). Handing every model
    # the same omega would give the ones with a different beta the wrong
    # free-stream decay -- the mistake that made the old comparison
    # meaningless, just in a smaller form.
    omega_by_beta = {}
    for b in (0.0828, 0.09, 0.075, 0.06, 0.05, 0.047, 0.04):
        _, w, c = fit_inlet_omega(x, k[jfs, :], ue, x_in, args.fit_to, b)
        omega_by_beta[f"{b:.4f}"] = {"omega_inlet": w, "fit_cost": c}

    # Inlet profiles, with the wall point prepended so the no-slip condition
    # lands on the wall rather than in the first fluid cell.
    y_prof = np.concatenate(([0.0], y))
    U_prof = np.concatenate(([0.0], U[:, 0]))
    k_prof = np.concatenate(([k[0, 0]], k[:, 0]))

    ue_in = float(np.max(U[:, 0]))
    i = int(np.argmax(U[:, 0]))
    d99 = float(np.interp(0.99 * ue_in, U[:i + 1, 0], y[:i + 1]))
    tu = float(100.0 * np.sqrt(2.0 * k[jfs, 0] / 3.0) / ue_in)

    # Langtry-Menter's own empirical correlation for the transition-onset
    # momentum-thickness Reynolds number as a function of local turbulence
    # intensity (zero pressure gradient branch). Leaving ReThetat at an
    # invented placeholder trips the model at the leading edge, which would
    # make it lose the comparison for a reason that is our fault, not its.
    if tu <= 1.3:
        re_theta_t = 1173.51 - 589.428 * tu + 0.2196 / tu ** 2
    else:
        re_theta_t = 331.50 * (tu - 0.5658) ** -0.671

    payload = {
        "ReThetat_inlet": float(re_theta_t),
        "x_inlet": x_in,
        "x_outlet": float(x[-1]),
        "y_max": float(y[-1]),
        "Ue_inlet": ue_in,
        "delta99_inlet": d99,
        "Tu_inlet_percent": tu,
        "omega_inlet": omega_inlet,
        "omega_fit_beta": args.beta,
        "omega_fit_cost": cost,
        "omega_inlet_by_beta": omega_by_beta,
        "note": ("Inlet taken from the DNS station itself, so no run-up "
                 "development is required of the model. The DNS inlet already "
                 "carries a developed laminar boundary layer, which also "
                 "removes the leading-edge singularity a plate-at-the-inlet "
                 "mesh would otherwise have."),
        "y": y_prof.tolist(),
        "U": U_prof.tolist(),
        "k": k_prof.tolist(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f)

    print(f"DNS inlet x = {x_in:.3f}, outlet x = {x[-1]:.3f}, "
          f"y_max = {y[-1]:.3f}")
    print(f"  Ue = {ue_in:.4f}, delta99 = {d99:.3f}, Tu = {tu:.2f}%")
    print(f"  ReThetat_inlet = {re_theta_t:.1f} "
          f"(Langtry-Menter correlation at Tu = {tu:.2f}%)")
    print(f"  omega_inlet = {omega_inlet:.4f} "
          f"(fitted from decay over x = {x_in:.0f}-{args.fit_to:.0f}, "
          f"beta = {args.beta})")
    for b, v in omega_by_beta.items():
        print(f"    beta={b}: omega_inlet={v['omega_inlet']:.4f} "
              f"(decay cost {v['fit_cost']:.4f})")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
