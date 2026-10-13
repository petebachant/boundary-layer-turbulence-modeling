#!/usr/bin/env python
"""Reduce Zemmour et al.'s LES of the ERCOFTAC T3 plates to the lines and
profiles the suite compares RANS against.

For each case (data/t3-bypass-les, read by pypkg/t3_les.py), along the
plate and the slip wall ahead of it:

- the edge velocity U_e, from the wall pressure and the free stream's
  total pressure and temperature at the inlet, isentropically (the runs
  are at Mach 0.2 and 0.5), as a static tap and the inlet's total
  pressure would give it, with the solver's gas constant, gamma and
  power-law viscosity (feos_air.ini); the layer's edge, for its height
  only, where |omega| d / U first falls below EDGE_C going out from the
  wall (d the wall distance), as the clipping closure's free-stream gate
  does. The edge velocity first taken there jumped between grid rows near
  the leading edge, where the layer is a few rows thick;
- the acceleration parameter K = nu dU_e/dx / U_e^2;
- the free-stream intensity, sqrt((u'^2 + v'^2 + w'^2)/3) / U_e averaged
  over the wall-normal index band the authors read it over (config.json),
  the measure used for the JHTDB plate;
- C_f = mu du/dy / (rho_e U_e^2 / 2) at the wall, with the solver's own
  wall gradient;
- the opposite wall's height, which sets the pressure gradient;

and, at PROFILE_FRACTIONS of the plate, the wall-normal profiles of the
mean velocity and the Reynolds stresses. The last SPONGE points of the
plate, the outflow sponge, are dropped.

As a check of the reading, K is compared with the authors' own
(acceleration_factor.dat, in the T3C cases), on their normalized Re_x.

Outputs
-------
results/t3-les.json (lines, onsets and the check)
results/t3-les-profiles.npz (the profiles)
"""

from __future__ import annotations

import json
import os

import numpy as np

from pypkg import t3_les

OUT = "results/t3-les.json"
PROFILES = "results/t3-les-profiles.npz"
EDGE_C = 0.02
SPONGE = 50
PROFILE_FRACTIONS = np.linspace(0.05, 0.9, 12)
#: Points kept along each line
N_LINE = 400
FIELDS = ["rho", "uu", "vv", "u2", "v2", "w2", "uv", "mu", "rho*duy"]
FIELDS += ["rho*dvx", "prs", "T"]
#: Air as the solver models it (feos_air.ini)
GAMMA = 1.4
R_GAS = 287.06
MU_EXP = 0.7


def edge_index(U, omega, d):
    """For each station, the first wall-normal index past the wall where
    |omega| d / U falls below EDGE_C."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(omega) * d / np.abs(U)
    out = np.empty(U.shape[0], dtype=int)
    for i in range(U.shape[0]):
        below = np.nonzero(r[i, 2:] < EDGE_C)[0]
        out[i] = below[0] + 2 if below.size else U.shape[1] - 1
    return out


def inlet_total(case):
    """Total pressure and temperature of the free stream just inside the
    inlet, at mid-height."""
    s = t3_les.read_stats(case, ["uu", "prs", "T"], block=1)
    i, j = 5, s["uu"].shape[1] // 2
    m2 = s["uu"][i, j] ** 2 / (GAMMA * R_GAS * s["T"][i, j])
    f = 1 + (GAMMA - 1) / 2 * m2
    return s["prs"][i, j] * f ** (GAMMA / (GAMMA - 1)), s["T"][i, j] * f


def block_lines(case, block, cfg, total, info):
    x, y = t3_les.read_grid(case, block)
    s = t3_les.read_stats(case, FIELDS, block)
    rho = s["rho"]
    U = s["uu"]
    omega = (s["rho*dvx"] - s["rho*duy"]) / rho
    d = y - y[:, :1]
    je = edge_index(U, omega, d)
    i = np.arange(U.shape[0])
    p0, t0 = total
    pw = s["prs"][:, 0]
    m2 = 2 / (GAMMA - 1) * ((p0 / pw) ** ((GAMMA - 1) / GAMMA) - 1)
    t_e = t0 / (1 + (GAMMA - 1) / 2 * np.clip(m2, 0, None))
    ue = np.sqrt(np.clip(m2, 0, None) * GAMMA * R_GAS * t_e)
    rho_e = pw / (R_GAS * t_e)
    nu_e = info["Muref"] * (t_e / info["Tref"]) ** MU_EXP / rho_e
    var = (s["u2"] - U**2) + (s["v2"] - s["vv"] ** 2) + s["w2"]
    j0, j1 = cfg["jmin_tu"], cfg["jmax_tu"]
    tu = 100 * np.sqrt(np.mean(var[:, j0:j1], axis=1) / 3) / ue
    lines = {
        "x": x[:, 0],
        "ue": ue,
        "nu_e": nu_e,
        "rho_e": rho_e,
        "tu": tu,
        "y_top": y[:, -1],
        "delta_edge": d[i, je],
    }
    if block == t3_les.PLATE:
        dudy = s["rho*duy"][:, 0] / rho[:, 0]
        lines["cf"] = s["mu"][:, 0] * dudy / (0.5 * rho_e * ue**2)
        lines = {k: v[:-SPONGE] for k, v in lines.items()}
    return lines, (x, y, s)


def thin(a, n=N_LINE):
    idx = np.unique(np.linspace(0, len(a) - 1, n).round().astype(int))
    return a[idx]


def main():
    result = {"edge_c": EDGE_C, "sponge_points": SPONGE, "cases": {}}
    profiles = {}
    for case in t3_les.CASES:
        info = t3_les.read_info(case)
        cfg = t3_les.config(case)
        total = inlet_total(case)
        up, _ = block_lines(case, 1, cfg, total, info)
        pl, (x, y, s) = block_lines(case, t3_les.PLATE, cfg, total, info)
        # K within each block: the edge jumps where the slip wall ends
        for b in (up, pl):
            b["K"] = b["nu_e"] * np.gradient(b["ue"], b["x"]) / b["ue"] ** 2
        line = {k: np.concatenate([up[k], pl[k]]) for k in up}
        rex = pl["ue"] * pl["x"] / pl["nu_e"]
        cf = pl["cf"]
        # Onset at the C_f minimum, where the laminar decline turns into
        # the rise, completion at the maximum after it, as the authors
        # normalize by; a plate that ends while C_f still rises has its
        # completion at the end
        i_min = int(np.argmin(cf))
        i_max = int(np.argmax(cf[i_min:])) + i_min
        row = {
            "u_ref": info["Uref"],
            "l_ref": info["Lref"],
            "mach_ref": info["Mref"],
            "re_ref": info["Reref"],
            "u_exp": cfg["Uin_exp"],
            "rex_cf_min_authors": cfg["rex_cf_min"],
            "rex_cf_max_authors": cfg["rex_cf_max"],
            "rex_cf_min": float(rex[i_min]),
            "rex_cf_max": float(rex[i_max]),
            "x_cf_min": float(pl["x"][i_min]),
            "x_cf_max": float(pl["x"][i_max]),
            # Just ahead of the plate, where the slip wall ends
            "tu_le": float(up["tu"][-1]),
            "tu_inlet": float(line["tu"][1]),
            "line": {
                k: thin(v).tolist() for k, v in line.items() if k != "nu_e"
            },
            "plate": {
                "x": thin(pl["x"]).tolist(),
                "rex": thin(rex).tolist(),
                "cf": thin(cf).tolist(),
            },
            "nu_ref": float(np.median(pl["nu_e"])),
        }
        # The authors' K, on their normalized Re_x, along the plate
        path = os.path.join(t3_les.case_dir(case), "acceleration_factor.dat")
        if os.path.isfile(path):
            a = np.loadtxt(path, comments="#")
            a = a[np.isfinite(a).all(axis=1)]
            r0, r1 = cfg["rex_cf_min"], cfg["rex_cf_max"]
            norm = (rex - r0) / (r1 - r0)
            a = a[(a[:, 0] >= norm[0]) & (a[:, 0] <= norm[-1])]
            mine = np.interp(a[:, 0], norm, pl["K"])
            scale = np.max(np.abs(a[:, 1]))
            row["k_check"] = {
                "rms_over_max": float(
                    np.sqrt(np.mean((mine - a[:, 1]) ** 2)) / scale
                ),
                "correlation": float(np.corrcoef(mine, a[:, 1])[0, 1]),
                "max_abs_authors": float(scale),
            }
        result["cases"][case] = row
        # Profiles at fractions of the plate
        n = x.shape[0] - SPONGE
        for k, f in enumerate(PROFILE_FRACTIONS):
            i = int(round(f * (n - 1)))
            key = f"{case}/{k}"
            profiles[f"{key}/x"] = np.array(x[i, 0])
            profiles[f"{key}/y"] = y[i] - y[i, 0]
            profiles[f"{key}/U"] = s["uu"][i]
            profiles[f"{key}/uu"] = s["u2"][i] - s["uu"][i] ** 2
            profiles[f"{key}/vv"] = s["v2"][i] - s["vv"][i] ** 2
            profiles[f"{key}/ww"] = s["w2"][i]
            profiles[f"{key}/uv"] = s["uv"][i] - s["uu"][i] * s["vv"][i]
            profiles[f"{key}/nu_w"] = np.array(s["mu"][i, 0] / s["rho"][i, 0])
            profiles[f"{key}/cf"] = np.array(
                np.interp(x[i, 0], pl["x"], pl["cf"])
            )
            profiles[f"{key}/ue"] = np.array(
                np.interp(x[i, 0], pl["x"], pl["ue"])
            )
        print(
            case,
            {k: v for k, v in row.items() if not isinstance(v, (dict, list))},
            row.get("k_check"),
        )
    with open(OUT, "w") as f:
        json.dump(result, f, indent=1)
        f.write("\n")
    np.savez_compressed(PROFILES, **profiles)


if __name__ == "__main__":
    main()
