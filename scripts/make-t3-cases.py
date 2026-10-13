#!/usr/bin/env python
"""Write OpenFOAM cases for the ERCOFTAC T3 plates on Zemmour et al.'s LES
domains, and the LES's lines in the cases' units.

Each case is the LES's two blocks as one two-dimensional domain: a slip
wall (here a symmetry plane) from the inlet to the plate's sharp leading
edge at x = 0, the no-slip plate from there to the outlet, and above both
the LES's slip top wall, flat in T3A and T3B and contoured in T3C2 to T3C5
to set the pressure gradient, taken from the top row of the LES's grid.
The outflow sponge's stretch is kept in the domain and left out of every
comparison.

Units are the plate runs': the inlet's bulk speed is 1 and the viscosity
1/800, so the length unit is 800 nu / U, with nu the LES's kinematic
viscosity in the free stream at its inlet and U the bulk speed over its
inlet, the mean of u across it, which sets the mass flux and so, through
the top wall's shape, the free stream downstream. The LES is compressible,
at Mach 0.5 in T3A and T3B and 0.2 in T3C; the runs are not.

The LES's lines (results/t3-les.json) are written in these units: the edge
velocity, the free-stream k (from its intensity and edge velocity), C_f
along the plate, and its onset and completion.

Outputs
-------
sim/t3/<case>/setup/ (system and constant; the mesh-t3 stage copies each
to sim/t3/<case>/template and meshes it there)
results/t3-rans-setup.json
"""

from __future__ import annotations

import json
import os
import shutil

import numpy as np

from pypkg import t3_les

OUT = "sim/t3"
SETUP = "results/t3-rans-setup.json"
LES = "results/t3-les.json"
NU = 1.0 / 800.0
#: First cell height, as on the nosed plate: y+ well under one
Y1 = 0.01
N_NORMAL = 160
#: Cells upstream of the plate and, per unit length, along it, with the
#: plate's first cell DX0 long at the sharp leading edge, where the layer
#: is a few tenths thick
N_UP = 80
PLATE_CELLS_PER_UNIT = 1.2
N_PLATE_MIN = 500
DX0 = 0.1
#: Wall-normal rows of the LES's inlet left out of its bulk speed: in T3A
#: the stored statistics in the rows next to its slip wall are far from
#: the free stream's
J_SKIP = 5
#: Points kept along the top contour's spline
N_TOP = 300


def grading(length, n, first):
    """Last-over-first cell ratio of n cells over length, the first first
    long."""
    lo, hi = 1.0 + 1e-12, 2.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        if length * (r - 1) / (r**n - 1) > first:
            lo = r
        else:
            hi = r
    return r ** (n - 1)


def reference(case):
    """The LES inlet's bulk speed and free-stream kinematic viscosity."""
    x, y = t3_les.read_grid(case, 1)
    s = t3_les.read_stats(case, ["rho", "uu", "mu"], block=1)
    i = 3
    yy, uu = y[i, J_SKIP:], s["uu"][i, J_SKIP:]
    u_ref = float(np.trapezoid(uu, yy) / (yy[-1] - yy[0]))
    j = y.shape[1] // 2
    nu_ref = float(s["mu"][i, j] / s["rho"][i, j])
    return u_ref, nu_ref


def top_contour(case, l_unit):
    """The LES's top wall over both blocks, in the cases' units."""
    pts = []
    for b in (1, t3_les.PLATE):
        x, y = t3_les.read_grid(case, b)
        pts.append(np.c_[x[:, -1], y[:, -1]])
    p = np.concatenate(pts) / l_unit
    p = p[np.argsort(p[:, 0])]
    _, keep = np.unique(p[:, 0].round(9), return_index=True)
    return p[keep]


def block_mesh_dict(top, x_in, x_end):
    y_in = float(np.interp(x_in, top[:, 0], top[:, 1]))
    y_le = float(np.interp(0.0, top[:, 0], top[:, 1]))
    y_end = float(np.interp(x_end, top[:, 0], top[:, 1]))
    v = [
        (x_in, 0.0),  # 0 inlet, bottom
        (0.0, 0.0),  # 1 leading edge
        (x_end, 0.0),  # 2 outlet, bottom
        (x_end, y_end),  # 3 outlet, top
        (0.0, y_le),  # 4 top above the leading edge
        (x_in, y_in),  # 5 inlet, top
    ]
    o = len(v)
    verts = [f"    ({x:.10g} {y:.10g} {z})" for z in (0, 0.1) for x, y in v]
    gy = grading(min(y_in, y_le, y_end), N_NORMAL, Y1)
    n_plate = max(N_PLATE_MIN, int(round(PLATE_CELLS_PER_UNIT * x_end)))
    gp = grading(x_end, n_plate, DX0)
    gu = 1.0 / grading(-x_in, N_UP, DX0)
    blocks = [
        f"    hex (0 1 4 5 {o} {o + 1} {o + 4} {o + 5}) "
        f"({N_UP} {N_NORMAL} 1) simpleGrading ({gu:.8g} {gy:.8g} 1)",
        f"    hex (1 2 3 4 {o + 1} {o + 2} {o + 3} {o + 4}) "
        f"({n_plate} {N_NORMAL} 1) simpleGrading ({gp:.8g} {gy:.8g} 1)",
    ]
    edges = []
    for a, b, lo, hi in ((5, 4, x_in, 0.0), (4, 3, 0.0, x_end)):
        xs = np.linspace(lo, hi, N_TOP)[1:-1]
        ys = np.interp(xs, top[:, 0], top[:, 1])
        for z, off in ((0, 0), (0.1, o)):
            p = " ".join(f"({x:.10g} {y:.10g} {z})" for x, y in zip(xs, ys))
            edges.append(f"    spline {a + off} {b + off} ({p})")

    def quad(a, b):
        return f"            ({a} {b} {b + o} {a + o})"

    text = BMD.format(
        verts="\n".join(verts),
        blocks="\n".join(blocks),
        edges="\n".join(edges),
        inlet=quad(0, 5),
        outlet=quad(2, 3),
        top="\n".join([quad(5, 4), quad(4, 3)]),
        sym=quad(0, 1),
        plate=quad(1, 2),
    )
    return text, {"n_plate": n_plate, "grading_plate": gp, "grading_y": gy}


BMD = """FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object blockMeshDict;
}}

convertToMeters 1;

vertices
(
{verts}
);

blocks
(
{blocks}
);

edges
(
{edges}
);

boundary
(
    inlet
    {{
        type patch;
        faces
        (
{inlet}
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
{outlet}
        );
    }}
    upperWall
    {{
        type patch;
        faces
        (
{top}
        );
    }}
    lowerWall
    {{
        type symmetryPlane;
        faces
        (
{sym}
        );
    }}
    plate
    {{
        type wall;
        faces
        (
{plate}
        );
    }}
);

defaultPatch
{{
    name frontAndBack;
    type empty;
}}
"""


def main():
    with open(LES) as f:
        les = json.load(f)
    setup = {"nu": NU, "y1": Y1, "cases": {}}
    for case in t3_les.CASES:
        u_ref, nu_ref = reference(case)
        l_unit = nu_ref / (NU * u_ref)
        top = top_contour(case, l_unit)
        x1, _ = t3_les.read_grid(case, 1)
        x2, _ = t3_les.read_grid(case, t3_les.PLATE)
        x_in = float(x1[0, 0] / l_unit)
        x_end = float(x2[-1, 0] / l_unit)
        x_sponge = float(x2[-1 - les["sponge_points"], 0] / l_unit)
        bmd, mesh = block_mesh_dict(top, x_in, x_end)
        d = os.path.join(OUT, case, "setup")
        if os.path.isdir(d):
            shutil.rmtree(d)
        for sub in ("system", "constant"):
            os.makedirs(os.path.join(d, sub))
        with open(os.path.join(d, "system", "blockMeshDict"), "w") as f:
            f.write(bmd)
        for name in ("controlDict", "fvSchemes", "fvSolution"):
            shutil.copy(
                os.path.join("sim", "system", name),
                os.path.join(d, "system", name),
            )
        shutil.copy(
            os.path.join("sim", "constant", "transportProperties"),
            os.path.join(d, "constant", "transportProperties"),
        )
        r = les["cases"][case]
        line = {k: np.array(v) for k, v in r["line"].items()}
        plate = {k: np.array(v) for k, v in r["plate"].items()}
        ue = line["ue"] / u_ref
        setup["cases"][case] = {
            "u_ref": u_ref,
            "nu_ref": nu_ref,
            "l_unit": l_unit,
            "x_inlet": x_in,
            "x_end": x_end,
            "x_sponge": x_sponge,
            "mesh": mesh,
            "top": {"x": top[:, 0].tolist(), "y": top[:, 1].tolist()},
            "les": {
                "x": (line["x"] / l_unit).tolist(),
                "ue": ue.tolist(),
                "k": (1.5 * (line["tu"] / 100 * ue) ** 2).tolist(),
                "tu": line["tu"].tolist(),
                "K": line["K"].tolist(),
                "plate_x": (plate["x"] / l_unit).tolist(),
                "cf": plate["cf"].tolist(),
                "x_onset": r["x_cf_min"] / l_unit,
                "x_completion": r["x_cf_max"] / l_unit,
                "tu_le": r["tu_le"],
            },
        }
        print(
            f"{case}: U {u_ref:.4g}, nu {nu_ref:.4g}, unit {l_unit:.4g} m, "
            f"x {x_in:.1f} to {x_end:.1f}, onset "
            f"{r['x_cf_min'] / l_unit:.1f}, {mesh}"
        )
    with open(SETUP, "w") as f:
        json.dump(setup, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
