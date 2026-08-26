#!/usr/bin/env python
"""Fit inlet k and omega for each baseline RANS model.

Bypass transition is driven by the free-stream turbulence, so a transition
model handed the wrong inlet turbulence intensity will predict transition in
the wrong place no matter how good it is. Comparing our closure -- whose inlet
conditions ARE fitted -- against baselines left at arbitrary defaults would be
a rigged comparison, so every model gets the same treatment here.

A k-omega model decays homogeneous turbulence analytically,

    omega(x) = omega_0 / (1 + beta*omega_0*(x - x_in)/U)
    k(x)     = k_0 * (1 + beta*omega_0*(x - x_in)/U)^(-betaStar/beta)

so for a given (beta, betaStar) the inlet pair (k_0, omega_0) is fully
determined by matching the measured DNS free-stream decay. We fit that pair
per model, using each model's own free-stream destruction coefficients.

A caveat worth stating: in the free stream the SST blending function goes to
zero, so k-omega SST and its Langtry-Menter transition variant both reduce to
the k-epsilon branch with beta = 0.0828. kkLOmega does not have a clean
(beta, betaStar) pair of this form, so its decay constants are approximated by
the same law; the fit residual reported below shows how well that holds.

Outputs
-------
results/baseline-freestream-bcs.json
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

# Free-stream destruction coefficients per model. In the free stream the SST
# blending function F1 -> 0, so the outer (k-epsilon branch) constants apply.
MODELS = {
    "kOmegaSST": {"beta": 0.0828, "betaStar": 0.09,
                  "note": "outer SST branch, F1 -> 0 in the free stream"},
    "kOmegaSSTLM": {"beta": 0.0828, "betaStar": 0.09,
                    "note": "Langtry-Menter transition SST, same free-stream "
                            "branch as kOmegaSST"},
    "kkLOmega": {"beta": 0.09, "betaStar": 0.09,
                 "note": "approximate: kkLOmega has no clean (beta, betaStar) "
                         "free-decay pair, so the generic law is used"},
    "kEpsilon": {"beta": 0.0828, "betaStar": 0.09,
                 "note": "converted to an equivalent epsilon at the inlet"},
}


def freestream_k(x, k0, omega0, beta, betaStar, ue, x_in):
    fac = 1.0 + beta * omega0 * (x - x_in) / ue
    fac = np.maximum(fac, 1e-12)
    return k0 * fac ** (-betaStar / beta)


def fit_model(xd, kd, ue, x_in, beta0, betaStar):
    """Fit the DNS free-stream decay as a power law about a virtual origin,
    then convert that to inlet values.

    Fitting (k0, omega0) directly at the inlet is ill-posed: the inlet sits 148
    length units upstream of the leading edge, no data constrains the decay
    over that stretch, and the fit happily buys a better downstream match with
    an inlet turbulence intensity of 80 percent. Parameterising by the decay
    itself removes that freedom.

    The analytic k-omega free decay is a power law,

        k(x) = A * (x - x_v)^(-n),   n = betaStar/beta

    with a virtual origin x_v set by omega0. Fit (A, x_v, n) to the measured
    DNS free stream, then invert:

        beta   = betaStar / n
        omega0 = ue / (beta * (x_in - x_v))
        k0     = A * (x_in - x_v)^(-n)

    beta has to come out of the fit rather than being held at the published
    value, and that is the fairness crux. The decay exponent is betaStar/beta,
    so the textbook beta = 0.0828 gives 1.09 while this DNS free stream falls
    by a factor of ~22 over the plate and needs about 1.9. Our own closure was
    allowed to fit beta -- it chose 0.047 -- so holding the baselines to the
    textbook value would hand them a free stream they cannot represent and let
    us win on a technicality.
    """
    # The virtual origin must lie upstream of the domain inlet, or the model
    # would need turbulence that has not been generated yet. Left free, the
    # DNS free stream fits best with an origin at x = -85, which is DOWNSTREAM
    # of the inlet at x = -118 -- the case geometry and the measured free
    # stream are not mutually consistent. Constraining the origin makes the
    # inlet condition realisable; the cost shows up as a larger decay residual,
    # which is reported rather than hidden.
    x_v_max = x_in - 10.0

    def resid(p):
        logA, x_v, n = p
        model = np.exp(logA) * np.maximum(xd - x_v, 1e-9) ** (-n)
        return np.log(np.maximum(model, 1e-30) / np.maximum(kd, 1e-30))

    best = None
    for x_v0 in (-2000.0, -800.0, -300.0, x_v_max - 1.0):
        for n0 in (1.1, 1.5, 2.0):
            A0 = kd[0] * max(xd[0] - x_v0, 1.0) ** n0
            try:
                r = least_squares(resid, [np.log(A0), x_v0, n0],
                                  bounds=([-50.0, -5000.0, 0.3],
                                          [50.0, x_v_max, 6.0]))
            except Exception:
                continue
            if best is None or r.cost < best.cost:
                best = r
    logA, x_v, n = best.x
    beta = betaStar / n
    # The inlet must sit downstream of the virtual origin for omega0 > 0.
    dx_in = x_in - x_v
    assert dx_in > 0, "virtual origin constraint should guarantee this"
    omega0 = ue / (beta * dx_in)
    k0 = float(np.exp(logA)) * dx_in ** (-n)
    return k0, omega0, float(beta), float(x_v), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x-inlet", type=float, default=-118.0)
    ap.add_argument("--y-freestream", type=float, default=20.0,
                    help="height at which the DNS free stream is sampled")
    ap.add_argument("--out", default="results/baseline-freestream-bcs.json")
    args = ap.parse_args()

    d = load_dns()
    x, y = d["x"], d["y"]
    j = int(np.argmin(np.abs(y - args.y_freestream)))
    k_inf = d["k"][j, :]
    ue = float(np.max(d["U"][j, :]))

    # Fit over the plate only; upstream of the leading edge the DNS free stream
    # is not yet in its self-similar decay.
    m = (x >= 30.0) & (x <= 990.0)
    xd, kd = x[m], k_inf[m]

    out = {"x_inlet": args.x_inlet, "Ue": ue,
           "y_freestream": float(y[j]),
           "note": ("Inlet k and omega fitted per model so every model sees "
                    "the same measured DNS free-stream decay. Without this "
                    "the transition baselines would be handed the wrong "
                    "turbulence intensity and lose for the wrong reason."),
           "models": {}}

    for name, cfg in MODELS.items():
        k0, w0, beta_fit, x_v, res = fit_model(xd, kd, ue, args.x_inlet,
                                               cfg["beta"], cfg["betaStar"])
        fit = freestream_k(xd, k0, w0, beta_fit, cfg["betaStar"], ue,
                           args.x_inlet)
        rel = np.abs(fit - kd) / np.maximum(kd, 1e-30)
        tu = float(100.0 * np.sqrt(2.0 * k0 / 3.0) / ue)
        entry = {
            "k_inlet": k0, "omega_inlet": w0,
            "epsilon_inlet": float(cfg["betaStar"] * k0 * w0),
            "Tu_inlet_percent": tu,
            "beta": beta_fit, "beta_published": cfg["beta"],
            "betaStar": cfg["betaStar"],
            "decay_exponent": float(cfg["betaStar"] / beta_fit),
            "virtual_origin_x": x_v,
            "fit_rel_err_mean": float(np.mean(rel)),
            "fit_rel_err_max": float(np.max(rel)),
            "note": cfg["note"],
        }
        out["models"][name] = entry
        print(f"{name:<12} k0={k0:.3e} omega0={w0:7.3f} Tu={tu:5.2f}%  "
              f"beta {beta_fit:.4f} (published {cfg['beta']:.4f}), "
              f"exponent {cfg['betaStar']/beta_fit:.2f}  "
              f"decay err {100*np.mean(rel):4.1f}% mean")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
