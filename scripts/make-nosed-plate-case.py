#!/usr/bin/env python
"""Write an OpenFOAM case for the JHTDB plate with its leading edge.

The JHTDB transitional boundary layer (Zaki 2013) was computed on a plate
of half-thickness L with a super-elliptic nose of aspect ratio 20,

    (1 - x / 20)^4 + (y + 1)^2 = 1,

stagnation point at (0, -1), the flat plate from x = 20 at y = 0, and
Re_L = 800, under free-stream turbulence of about 3 percent at the leading
edge. The database stores only x >= 30, which is where every other plate
run in this project starts. This case puts the nose back: the upper half of
the flow, with a symmetry line ahead of the stagnation point, a C-shaped
block around the nose whose outer edge, a quarter ellipse, is the inlet,
and a block over the flat plate to x = 1000 under a slip top at y = TOP.

The patch names are the plate runs' (inlet, outlet, upperWall, lowerWall,
plate), so their field templates carry over; lowerWall, the symmetry line
here, is a symmetry plane.

Outputs
-------
sim/nosed-plate/setup/ (system and constant; the mesh-nosed-plate stage
copies it to sim/nosed-plate/template and meshes it there)
"""

from __future__ import annotations

import os
import shutil

import numpy as np

OUT = "sim/nosed-plate/setup"
AR = 20.0
NU = 1.0 / 800.0
TOP = 40.0
#: Distance of the inlet ahead of the stagnation point along the symmetry
#: line
UPSTREAM = 40.0
X_END = 1000.0
#: Cells along the nose, along the plate, and normal to the wall
N_NOSE, N_PLATE, N_NORMAL = 140, 700, 160
#: First cell height: y+ well under one at the DNS's turbulent C_f
Y1 = 0.01
#: Last-over-first ratios along the nose (fine at the stagnation point) and
#: along the plate (fine at its start)
G_NOSE, G_PLATE = 6.0, 8.0


def nose(n):
    """Points on the nose from the stagnation point to the flat plate."""
    t = np.linspace(0.0, np.pi / 2, n)
    # Parametrized by angle so points cluster at the stagnation point:
    # y + 1 = sin t, 1 - x/20 = cos(t)^(1/2)
    y = np.sin(t) - 1.0
    x = AR * (1.0 - np.sqrt(np.cos(t)))
    x[-1], y[-1] = AR, 0.0
    return np.c_[x, y]


def wall_grading(length, n, first):
    lo, hi = 1.0 + 1e-9, 2.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        if length * (r - 1) / (r**n - 1) > first:
            lo = r
        else:
            hi = r
    return r ** (n - 1)


def inlet_curve(n):
    """The quarter ellipse about (AR, -1) from the symmetry line to the top
    above the nose's end."""
    t = np.linspace(0.0, np.pi / 2, n)
    ax, ay = AR + UPSTREAM, TOP + 1.0
    return np.c_[AR - ax * np.cos(t), -1.0 + ay * np.sin(t)]


def block_mesh_dict():
    v = [
        (0.0, -1.0),  # 0 stagnation point
        (AR, 0.0),  # 1 end of the nose
        (AR, TOP),  # 2 top above the end of the nose
        (-UPSTREAM, -1.0),  # 3 inlet on the symmetry line
        (X_END, 0.0),  # 4 end of the plate
        (X_END, TOP),  # 5 outlet top
    ]
    o = len(v)
    verts = [f"    ({x:.10g} {y:.10g} {z})" for z in (0, 0.1) for x, y in v]
    g = wall_grading(UPSTREAM, N_NORMAL, Y1)
    blocks = [
        # Nose: x along the surface from the stagnation point, y outward;
        # its left edge is the symmetry line, its outer edge the inlet
        f"    hex (0 1 2 3 {o} {o + 1} {o + 2} {o + 3}) "
        f"({N_NOSE} {N_NORMAL} 1) simpleGrading ({G_NOSE} {g:.8g} 1)",
        # Plate
        f"    hex (1 4 5 2 {o + 1} {o + 4} {o + 5} {o + 2}) "
        f"({N_PLATE} {N_NORMAL} 1) simpleGrading ({G_PLATE} {g:.8g} 1)",
    ]
    nz = nose(200)[1:-1]
    ic = inlet_curve(200)[1:-1][::-1]
    edges = []
    for z, off in ((0, 0), (0.1, o)):
        p = " ".join(f"({x:.10g} {y:.10g} {z})" for x, y in nz)
        edges.append(f"    spline {0 + off} {1 + off} ({p})")
        p = " ".join(f"({x:.10g} {y:.10g} {z})" for x, y in ic)
        edges.append(f"    spline {2 + off} {3 + off} ({p})")

    def quad(a, b):
        return f"            ({a} {b} {b + o} {a + o})"

    return BMD.format(
        verts="\n".join(verts),
        blocks="\n".join(blocks),
        edges="\n".join(edges),
        inlet=quad(2, 3),
        outlet=quad(4, 5),
        top=quad(2, 5),
        sym=quad(0, 3),
        plate="\n".join([quad(0, 1), quad(1, 4)]),
    )


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
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for sub in ("system", "constant"):
        os.makedirs(os.path.join(OUT, sub))
    with open(os.path.join(OUT, "system", "blockMeshDict"), "w") as f:
        f.write(block_mesh_dict())
    # The plate runs' solver settings and sampling, and their transport
    # properties, which carry Re_L = 800
    for name in ("controlDict", "fvSchemes", "fvSolution", "sample"):
        shutil.copy(
            os.path.join("sim", "system", name),
            os.path.join(OUT, "system", name),
        )
    shutil.copy(
        os.path.join("sim", "constant", "transportProperties"),
        os.path.join(OUT, "constant", "transportProperties"),
    )
    print(
        f"Wrote {OUT}: wall grading {wall_grading(UPSTREAM, N_NORMAL, Y1):.4g}"
    )


if __name__ == "__main__":
    main()
