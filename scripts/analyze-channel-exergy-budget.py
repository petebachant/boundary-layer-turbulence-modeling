#!/usr/bin/env python
"""Where do the fluctuations' exergy come from and where does it go, in
Lee & Moser's channels?

The fluctuation exergy of scripts/test-fluctuation-exergy.py,
X = k - 3/2 (det R)^(1/3), changes with the Reynolds stresses as

    dX = 1/2 tr(W dR),   W = I - (det R)^(1/3) R^-1,

so each term T_ij of the Reynolds stress budget contributes 1/2 tr(W T)
to the budget of X. Two things follow without data. Mean-strain
production leaves det R unchanged, tr(R^-1 P) = -2 div U = 0, so it feeds
X with the whole of P_k: production is the reversible step. And the
pressure-strain term, which moves no energy (it is traceless), changes X
only through the second part of W, destroying it where it returns the
stresses toward isotropy: that is where fluctuation exergy turns into
energy that can no longer do work, and the term is inviscid. Viscosity
acts through the dissipation tensor, which near the wall is far from
isotropic.

Check: production's contribution equals P_k at every point, to
IDENTITY_TOL relative to the peak of P_k. If it does not, the
computation is wrong, not the physics. The tolerance was first 1e-6; the
first run missed it by up to 4e-5, at the first points off the wall,
where v'v' is 1e-13 and uu vv - uv^2 loses its digits to cancellation, so
it is set at 1e-4, which still catches any error in the algebra.

Test, fixed before any budget was computed: in every channel, the
pressure-strain term carries more than half of the fluctuation exergy
destroyed across the half channel, the destruction being minus the
integrals of the pressure-strain and dissipation contributions. Also
reported: each term's integral over the half channel and above y+ = 30,
the share of the wall's work that goes through the fluctuations rather
than straight to viscous dissipation of the mean flow, and the profiles.

Added after the first run, which found pressure transport destroying
nearly as much exergy as pressure-strain: the split of the
velocity-pressure-gradient term into those two is not unique, and near
the wall they cancel in the v'v' budget, where W is singular, so the
split matters there. Their sum, and its share of all destruction with
the viscous terms (dissipation and viscous transport) as the rest, is
reported as the robust measure of how much is destroyed inviscidly; the
test above stands as fixed.

All in wall units; integrals are over y+ from the first point off the
wall, where W is singular (it grows as y^-4/3 against terms that vanish
faster), to the centerline.

Outputs
-------
results/channel-exergy-budget.json
"""

from __future__ import annotations

import json
import os

import numpy as np

OUT = "results/channel-exergy-budget.json"
FLUC = "data/lee-moser-channel"
BUDGETS = "data/lee-moser-budgets"
RE = ("0180", "0550", "1000", "2000", "5200")
TERMS = (
    "production",
    "turbulent_transport",
    "viscous_transport",
    "pressure_strain",
    "pressure_transport",
    "dissipation",
)
IDENTITY_TOL = 1e-4
Y_OUTER = 30.0


def read(path):
    return np.loadtxt(
        [line for line in open(path) if line.strip() and line[0] != "%"]
    )


def main():
    out = {}
    for re_ in RE:
        f = read(os.path.join(FLUC, f"LM_Channel_{re_}_vel_fluc_prof.dat"))
        mean = read(os.path.join(FLUC, f"LM_Channel_{re_}_mean_prof.dat"))
        yp = f[:, 1]
        uu, vv, ww, uv = f[:, 2], f[:, 3], f[:, 4], f[:, 5]
        b = {}
        for c in ("uu", "vv", "ww", "uv"):
            a = read(
                os.path.join(BUDGETS, f"LM_Channel_{re_}_RSTE_{c}_prof.dat")
            )
            assert np.allclose(a[:, 1], yp), f"{c} grid differs at {re_}"
            b[c] = {t: a[:, 2 + i] for i, t in enumerate(TERMS)}
            # The files give the dissipation as a positive rate, subtracted
            b[c]["dissipation"] = -b[c]["dissipation"]
        # The wall row's y+ is 2e-14, not 0, at Re_tau = 180
        m = (yp > 1e-6) & (f[:, 0] < 1.0)
        det2 = uu * vv - uv**2
        lam = np.cbrt(np.maximum(det2 * ww, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            w11 = 1.0 - lam * vv / det2
            w22 = 1.0 - lam * uu / det2
            w12 = lam * uv / det2
            w33 = 1.0 - lam / ww
        e = {
            t: 0.5
            * (
                w11 * b["uu"][t]
                + w22 * b["vv"][t]
                + w33 * b["ww"][t]
                + 2.0 * w12 * b["uv"][t]
            )
            for t in TERMS
        }
        pk = 0.5 * (
            b["uu"]["production"]
            + b["vv"]["production"]
            + b["ww"]["production"]
        )
        ident = float(
            np.max(np.abs(e["production"][m] - pk[m])) / np.max(np.abs(pk))
        )
        y = yp[m]
        outer = y > Y_OUTER

        def integ(a, mask=None):
            mm = np.ones_like(y, bool) if mask is None else mask
            return float(np.trapezoid(a[m][mm], y[mm]))

        ints = {t: integ(e[t]) for t in TERMS}
        ints_outer = {t: integ(e[t], outer) for t in TERMS}
        destroyed = -(ints["pressure_strain"] + ints["dissipation"])
        dudy = mean[:, 3]
        direct = float(np.trapezoid((dudy[m]) ** 2, y))
        X = 0.5 * (uu + vv + ww) - 1.5 * lam
        step = 1
        eps_k = -0.5 * (
            b["uu"]["dissipation"]
            + b["vv"]["dissipation"]
            + b["ww"]["dissipation"]
        )
        out[f"re_tau_{int(re_)}"] = {
            "identity_error": ident,
            "integrals": ints,
            "integrals_outer": ints_outer,
            "production": integ(pk),
            "destroyed": destroyed,
            "pressure_strain_share": -ints["pressure_strain"] / destroyed,
            "pressure_strain_share_outer": -ints_outer["pressure_strain"]
            / -(ints_outer["pressure_strain"] + ints_outer["dissipation"]),
            "through_fluctuations": integ(pk) / (integ(pk) + direct),
            "inviscid_share": (
                ints["pressure_strain"] + ints["pressure_transport"]
            )
            / (
                ints["pressure_strain"]
                + ints["pressure_transport"]
                + ints["dissipation"]
                + ints["viscous_transport"]
            ),
            "profiles": {
                "y_plus": y[::step].tolist(),
                "exergy": X[m][::step].tolist(),
                "fraction": (X / (0.5 * (uu + vv + ww)))[m][::step].tolist(),
                "k": (0.5 * (uu + vv + ww))[m][::step].tolist(),
                "production_k": pk[m][::step].tolist(),
                "dissipation_k": eps_k[m][::step].tolist(),
                # lambda_g tr(R^-1), for the pressure-strain models'
                # implied destruction in scripts/fit-exergy-destruction.py
                "lam_tr_rinv": (lam * ((uu + vv) / det2 + 1.0 / ww))[m][
                    ::step
                ].tolist(),
                **{t: e[t][m][::step].tolist() for t in TERMS},
            },
        }
        r = out[f"re_tau_{int(re_)}"]
        print(
            re_,
            {
                k: v
                for k, v in r.items()
                if k not in ("profiles", "integrals", "integrals_outer")
            },
        )
        print("   ", {k: round(v, 3) for k, v in ints.items()})
    result = {
        "identity_tol": IDENTITY_TOL,
        "channels": out,
        "identity_error_max": max(r["identity_error"] for r in out.values()),
        "identity_holds": all(
            r["identity_error"] < IDENTITY_TOL for r in out.values()
        ),
        "passes": all(r["pressure_strain_share"] > 0.5 for r in out.values()),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print({k: v for k, v in result.items() if k != "channels"})


if __name__ == "__main__":
    main()
