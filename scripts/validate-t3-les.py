#!/usr/bin/env python
"""Does Zemmour et al.'s LES of the T3A plate match the JHTDB DNS where the
two describe the same flow?

Both are zero-pressure-gradient plates under about 3 percent free-stream
turbulence at the leading edge. Their transitions need not agree (the
free-stream length scales and decays differ), but once turbulent the
layers should, at the same Re_theta. As for Bienner et al.'s LES, the
first LES admitted by exception, the check is of the turbulent profiles in
wall units, at the largest Re_theta the T3A layer reaches before the
outflow sponge, and at the middle of its turbulent stretch.

Tests, fixed before the comparison was run:

  u_plus   U+ within U_TOL (RMS relative difference) of the DNS's over
           Y_RANGE in y+
  uv_peak  the peak of -u'v'+ within UV_TOL of the DNS's

at both stations. Also reported: the leading-edge intensity of each, in
the k-based measure the suite uses, and the Re_x of each one's C_f
minimum. The LES is compressible, at Mach 0.5 in T3A, and theta is taken
from the velocity alone.

Outputs
-------
results/t3-les-validation.json
"""

from __future__ import annotations

import json

import numpy as np

from pypkg import t3_les
from pypkg.dns_case import edge_velocity, load_dns

OUT = "results/t3-les-validation.json"
U_TOL = 0.05
UV_TOL = 0.10
Y_RANGE = (5.0, 100.0)
#: The LES's edge: where |omega| d / U first falls below this
EDGE_C = 0.02
SPONGE = 50
PLATE_LE = "results/plate-leading-edge-intensity.json"


def theta(y, U, ue):
    f = np.clip(U / ue, 0, None)
    return np.trapezoid(f * (1 - f), y)


def les_layer():
    """T3A's plate: per station, y, U, -u'v', the edge, theta, u_tau."""
    x, y = t3_les.read_grid("t3a")
    s = t3_les.read_stats(
        "t3a", ["rho", "uu", "vv", "uv", "mu", "rho*duy", "rho*dvx"]
    )
    rho, U = s["rho"], s["uu"]
    omega = (s["rho*dvx"] - s["rho*duy"]) / rho
    d = y - y[:, :1]
    rows = []
    for i in range(x.shape[0] - SPONGE):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.abs(omega[i]) * d[i] / np.abs(U[i])
        below = np.nonzero(r[2:] < EDGE_C)[0]
        je = below[0] + 2 if below.size else U.shape[1] - 1
        ue = U[i, je]
        nu_e = s["mu"][i, je] / rho[i, je]
        nu_w = s["mu"][i, 0] / rho[i, 0]
        dudy = s["rho*duy"][i, 0] / rho[i, 0]
        th = theta(d[i, : je + 1], U[i, : je + 1], ue)
        rows.append(
            {
                "x": x[i, 0],
                "y": d[i],
                "U": U[i],
                "uv": -(s["uv"][i] - U[i] * s["vv"][i]),
                "ue": ue,
                "re_theta": ue * th / nu_e,
                "rex": ue * x[i, 0] / nu_e,
                "nu_w": nu_w,
                "u_tau": np.sqrt(nu_w * abs(dudy)),
                "cf": 2 * nu_w * dudy * rho[i, 0] / (rho[i, je] * ue**2),
            }
        )
    return rows


def dns_layer():
    """The JHTDB plate: per station, as les_layer."""
    d = load_dns()
    x, y, nu = d["x"], d["y"], d["nu"]
    ue = [edge_velocity(d["U"][:, i]) for i in range(len(x))]
    rows = []
    y1, y2 = y[0], y[1]
    for i in range(len(x)):
        U = d["U"][:, i]
        # Slope at the wall of the quadratic through it and the first two
        # nodes, which sit above it
        dudy = (U[0] * y2**2 - U[1] * y1**2) / (y1 * y2 * (y2 - y1))
        yy = np.concatenate([[0.0], y])
        UU = np.concatenate([[0.0], U])
        top = np.searchsorted(y, 0.5 * y[-1])
        rows.append(
            {
                "x": x[i],
                "y": y,
                "U": U,
                "uv": -d["uv"][:, i],
                "ue": ue[i],
                "re_theta": ue[i] * theta(yy[:top], UU[:top], ue[i]) / nu,
                "rex": x[i] / nu,
                "nu_w": nu,
                "u_tau": np.sqrt(nu * abs(dudy)),
                "cf": 2 * nu * dudy / ue[i] ** 2,
            }
        )
    return rows


def compare(les, dns):
    yp = np.geomspace(*Y_RANGE, 60)
    lp = les["y"] * les["u_tau"] / les["nu_w"]
    dp = dns["y"] * dns["u_tau"] / dns["nu_w"]
    ul = np.interp(yp, lp, les["U"] / les["u_tau"])
    ud = np.interp(yp, dp, dns["U"] / dns["u_tau"])
    uv_l = np.max(les["uv"]) / les["u_tau"] ** 2
    uv_d = np.max(dns["uv"]) / dns["u_tau"] ** 2
    u_err = float(np.sqrt(np.mean(((ul - ud) / ud) ** 2)))
    uv_err = float(abs(uv_l / uv_d - 1))
    return {
        "re_theta_les": float(les["re_theta"]),
        "re_theta_dns": float(dns["re_theta"]),
        "x_les": float(les["x"]),
        "x_dns": float(dns["x"]),
        "cf_les": float(les["cf"]),
        "cf_dns": float(dns["cf"]),
        "u_plus_rms_rel": u_err,
        "uv_peak_les": float(uv_l),
        "uv_peak_dns": float(uv_d),
        "uv_peak_rel": uv_err,
        "u_plus_ok": u_err <= U_TOL,
        "uv_ok": uv_err <= UV_TOL,
    }


def main():
    les, dns = les_layer(), dns_layer()
    cf_l = np.array([r["cf"] for r in les])
    i_min = int(np.argmin(cf_l))
    i_max = int(np.argmax(cf_l[i_min:])) + i_min
    re_l = np.array([r["re_theta"] for r in les])
    re_d = np.array([r["re_theta"] for r in dns])
    stations = {}
    for name, i in (
        ("end", len(les) - 1),
        ("middle", (i_max + len(les) - 1) // 2),
    ):
        j = int(np.argmin(np.abs(re_d - re_l[i])))
        stations[name] = compare(les[i], dns[j])
    cf_d = np.array([r["cf"] for r in dns])
    with open(PLATE_LE) as f:
        tu_le_dns = json.load(f)["tu_le_percent"]
    with open("results/t3-les.json") as f:
        tu_le_les = json.load(f)["cases"]["t3a"]["tu_le"]
    result = {
        "u_tol": U_TOL,
        "uv_tol": UV_TOL,
        "y_plus_range": list(Y_RANGE),
        "stations": stations,
        "rex_cf_min_les": float(les[i_min]["rex"]),
        "rex_cf_max_les": float(les[i_max]["rex"]),
        "rex_cf_min_dns": float(dns[int(np.argmin(cf_d))]["rex"]),
        "tu_le_les": tu_le_les,
        "tu_le_dns": tu_le_dns,
        "passes": all(s["u_plus_ok"] and s["uv_ok"] for s in stations.values()),
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
