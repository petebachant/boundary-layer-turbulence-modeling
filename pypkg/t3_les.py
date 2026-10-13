"""Read Zemmour, Gloerfelt and Cinnella's LES of the ERCOFTAC T3 plates.

Each case in data/t3-bypass-les (scripts/fetch-t3-les.py) is a MUSICAA run
in two blocks: block 1 upstream of the plate, over a slip wall, and block 2
the plate, a no-slip wall below and a slip wall above, shaped in the T3C
cases to set the pressure gradient. The format follows the authors' reader
(github.com/LouenasZm/ppMusicaa):

- info.ini: block sizes and reference quantities.
- grid_bl<n>[_ngh5].bin: x and y over the block as float64 in Fortran
  order, with five ghost points a side in the newer grids, then the
  spanwise coordinates. The older grids (T3A, T3B) are Fortran sequential
  records. The byte order is the case's (config.json). In the release,
  T3A's and T3B's grid_bl1.bin are copies of grid_bl2.bin, so block 1 is
  read from the ASCII Plot3D grid/grid_mod_bl1.x instead: the block as the
  solver ran it, its last points half a cell short of the plate as in the
  newer grids, and its wall-normal points block 2's.
- stats1_bl<n>.bin, stats2_bl<n>.bin: 23 and 144 time- and spanwise-
  averaged fields, each nx by ny float64, little-endian, Fortran order.
  "uu", "vv", "ww" are the mean velocities and "u2" the mean of u^2.

Everything is dimensional, in SI units, on the solver's rescaled geometry.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

ROOT = "data/t3-bypass-les"
CASES = ("t3a", "t3b", "t3c2", "t3c3", "t3c4", "t3c5")
STATS1 = (
    "rho uu vv ww prs T rhou rhov rhow rhoe rho^2 u2 v2 w2 uv uw vw vT p2 "
    "T2 mu div div^2"
).split()
STATS2 = (
    "e h c s M kt g la cp cv pr eck rho*dux rho*duy rho*duz rho*dvx rho*dvy "
    "rho*dvz rho*dwx rho*dwy rho*dwz p*div rho*div b1 b2 b3 rho*T u*T v*T "
    "e^2 h^2 c^2 s^2 Mt^2 g^2 mu^2 la^2 cv^2 cp^2 pr^2 eck^2 p*u p*v s*u "
    "s*v p*rho h*rho T*p p*s T*s rho*s g*rho g*p g*s g*T g*u g*v p*dux "
    "p*dvy p*dwz p*duy p*dvx rho*div^2 dux^2 duy^2 duz^2 dvx^2 dvy^2 dvz^2 "
    "dwx^2 dwy^2 dwz^2 b1^2 b2^2 b3^2 rho*b1 rho*b2 rho*b3 rho*u^2 rho*v^2 "
    "rho*w^2 rho*T^2 rho*b1^2 rho*b2^2 rho*b3^2 rho*u*v rho*u*w rho*v*w "
    "rho*v*T rho*u^2*v rho*v^2*v rho*w^2*v rho*v^2*u rho*dux^2 rho*dvy^2 "
    "rho*dwz^2 rho*duy*dvx rho*duz*dwx rho*dvz*dwy u^3 p^3 u^4 p^4 Frhou "
    "Frhov Frhow Grhov Grhow Hrhow Frhov*u Frhou*u Frhov*v Frhow*w Grhov*u "
    "Grhov*v Grhow*w Frhou*dux Frhou*dvx Frhov*dux Frhov*duy Frhov*dvx "
    "Frhov*dvy Frhow*duz Frhow*dvz Frhow*dwx Grhov*duy Grhov*dvy Grhow*duz "
    "Grhow*dvz Grhow*dwy Hrhow*dwz la*dTx la*dTy la*dTz"
).split()
#: Fields in each file, at least: the authors' reader names only the first
#: 134 of stats2's, which holds 144 or, in T3C2, run by a later version of
#: the solver, 154
NVAR = {1: 23, 2: 144}
#: The plate's block
PLATE = 2


def case_dir(case, root=ROOT):
    return os.path.join(root, case)


def config(case, root=ROOT):
    """The authors' set-up for the case: byte order, grid version, the
    wall-normal index band they read the free-stream intensity over, and
    the Re_x of minimum and maximum C_f they normalize by."""
    with open(os.path.join(root, "config.json")) as f:
        return json.load(f)[case]


def read_info(case, root=ROOT):
    """info.ini as a dict: nbloc, is_curv, each block's (nx, ny, nz), and
    the reference quantities."""
    info = {"blocks": {}}
    with open(os.path.join(case_dir(case, root), "info.ini")) as f:
        for line in f:
            if "=" not in line:
                continue
            keys, values = (s.split() for s in line.split("=", 1))
            m = re.match(r"bl0*(\d+)", keys[-1])
            if keys[:3] == ["NX", "NY", "NZ"] and m:
                info["blocks"][int(m.group(1))] = tuple(int(v) for v in values)
                continue
            for k, v in zip(keys, values):
                if v in ("T", "F"):
                    info[k] = v == "T"
                else:
                    try:
                        info[k] = float(v.replace("D", "E"))
                    except ValueError:
                        info[k] = v
    info["nbloc"] = int(info["nbloc"])
    return info


def read_grid(case, block=PLATE, root=ROOT):
    """x and y over the block, each (nx, ny), without ghost points."""
    info = read_info(case, root)
    cfg = config(case, root)["ppMusicaa"]["grid"]
    order = "<" if cfg["endianess"] == "little" else ">"
    nx, ny, nz = info["blocks"][block]
    d = case_dir(case, root)
    if cfg["new_grid"]:
        ngh = int(info.get("ngh", 5))
        gx, gy = nx + 2 * ngh, ny + 2 * ngh
        path = os.path.join(d, f"grid_bl{block}_ngh{ngh}.bin")
        a = np.fromfile(path, dtype=order + "f8", count=2 * gx * gy)
        x = a[: gx * gy].reshape((gx, gy), order="F")[ngh:-ngh, ngh:-ngh]
        y = a[gx * gy :].reshape((gx, gy), order="F")[ngh:-ngh, ngh:-ngh]
        return x.copy(), y.copy()
    if block == 1:
        return _plot3d(os.path.join(d, "grid", "grid_mod_bl1.x"), nx, ny)
    # Fortran sequential records: each record is its length, the data and
    # the length again; the first three hold nx, ny and nz
    with open(os.path.join(d, f"grid_bl{block}.bin"), "rb") as f:
        raw = f.read()
    records, pos = [], 0
    while pos < len(raw):
        n = int(np.frombuffer(raw, order + "i4", 1, pos)[0])
        records.append(raw[pos + 4 : pos + 4 + n])
        pos += n + 8
    dims = [int(np.frombuffer(r, order + "i4")[0]) for r in records[:3]]
    assert dims[:2] == [nx, ny], f"{case} grid is {dims}, not {nx, ny}"
    x = np.frombuffer(records[3], order + "f8").reshape((nx, ny), order="F")
    y = np.frombuffer(records[4], order + "f8").reshape((nx, ny), order="F")
    return x.copy(), y.copy()


def _plot3d(path, nx, ny):
    """x and y of a single-block ASCII Plot3D grid."""
    with open(path) as f:
        a = f.read().split()
    dims = [int(v) for v in a[1:3]]
    assert dims == [nx, ny], f"{path} is {dims}, not {nx, ny}"
    v = np.array(a[4 : 4 + 2 * nx * ny], dtype=float)
    x = v[: nx * ny].reshape((nx, ny), order="F")
    y = v[nx * ny :].reshape((nx, ny), order="F")
    return x, y


def read_stats(case, names, block=PLATE, root=ROOT):
    """The named averaged fields over the block, each (nx, ny)."""
    info = read_info(case, root)
    nx, ny, _ = info["blocks"][block]
    out = {}
    for which, table in ((1, STATS1), (2, STATS2)):
        wanted = [n for n in names if n in table]
        if not wanted:
            continue
        path = os.path.join(case_dir(case, root), f"stats{which}_bl{block}.bin")
        size = os.path.getsize(path)
        nvar, rest = divmod(size, nx * ny * 8)
        assert not rest and nvar >= NVAR[which], f"{path} has {size} bytes"
        a = np.memmap(path, dtype="<f8", mode="r")
        for n in wanted:
            i = table.index(n)
            out[n] = np.array(
                a[i * nx * ny : (i + 1) * nx * ny].reshape((nx, ny), order="F")
            )
    missing = set(names) - set(out)
    assert not missing, f"unknown fields {sorted(missing)}"
    return out
