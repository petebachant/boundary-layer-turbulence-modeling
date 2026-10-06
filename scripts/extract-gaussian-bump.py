#!/usr/bin/env python
"""Extract the Gaussian bump DNS's inflow profile and wall-normal stations.

The field statistics at Re_L = 2 million (Uzun2022) are one ordered,
block-packed Tecplot zone, 15146 by 384 points on a curvilinear grid whose
second index runs away from the wall. This keeps the grid line at the
inflow, x/L = -0.8, which a RANS case takes as its inlet profile, and the
lines nearest STATIONS, against which its profiles are scored.

Outputs
-------
results/gaussian-bump-profiles.json
"""

from __future__ import annotations

import json
import zipfile

import numpy as np

SRC = "data/gaussian-bump-dns/SpeedBump-ReL-2M-Statistics.dat.zip"
OUT = "results/gaussian-bump-profiles.json"
NAMES = ("x", "y", "rho", "U", "V", "p", "uu", "vv", "ww", "uv")
STATIONS = (
    -0.6,
    -0.4,
    -0.2,
    -0.1,
    0.0,
    0.05,
    0.1,
    0.15,
    0.2,
    0.25,
    0.3,
    0.35,
    0.4,
    0.5,
    0.6,
    0.8,
)


def main():
    with zipfile.ZipFile(SRC) as z:
        name = z.namelist()[0]
        text = z.read(name).decode()
    head, _, body = text.partition("DT=(")
    ni = int(head.split("I=")[1].split(",")[0])
    nj = int(head.split("J=")[1].split(",")[0])
    body = body.split(")", 1)[1]
    vals = np.fromstring(body, sep=" ")
    n = ni * nj
    assert vals.size == n * len(NAMES), (vals.size, n)
    f = {
        k: vals[i * n : (i + 1) * n].reshape(nj, ni)
        for i, k in enumerate(NAMES)
    }
    del vals, text, body
    x_wall = f["x"][0]

    def line(i):
        return {k: f[k][:, i].tolist() for k in NAMES}

    out = {
        "re_l": 2_000_000,
        "mach": 0.2,
        "ni": ni,
        "nj": nj,
        "y_top": float(f["y"][-1].mean()),
        "inflow": line(0),
        "stations": {
            f"{s:g}": line(int(np.argmin(np.abs(x_wall - s)))) for s in STATIONS
        },
    }
    with open(OUT, "w") as fo:
        json.dump(out, fo)
    print(
        f"{ni} x {nj}, top y/L {out['y_top']:.3f}, inflow U_e "
        f"{max(out['inflow']['U']):.4f}"
    )


if __name__ == "__main__":
    main()
