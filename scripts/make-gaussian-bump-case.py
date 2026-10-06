#!/usr/bin/env python
"""Write an OpenFOAM case for the Gaussian bump at Re_L = 2 million, the
flow of Uzun and Malik's DNS (Uzun2022).

The wall is y = 0.085 exp(-(x / 0.195)^2) in units of L from the DNS's
inflow, x = -0.8, to an outlet at X_OUT, past the DNS's physical domain,
which ends at x = 1. The top is a free stream at y = 1, as in the DNS. The
mesh is three blocks along x, with the middle one, over the bump and its
separation, finely and uniformly spaced; the first cell is sized for y+
about one at the DNS's largest skin friction.

The inlet takes the DNS's own mean profile at x = -0.8 (from
results/gaussian-bump-profiles.json) as fixedProfile tables in y: U and V,
k from the normal stresses, and omega from k over the eddy viscosity
-uv / (dU/dy), with the wall's 6 nu / (beta y^2) below the layer's first
points, capped at its value at the first cell's centre, as the fast tier
seeds its turbulent inlets. The DNS has no
free-stream turbulence, so above the layer k and omega take the NASA
Turbulence Modeling Resource's quiet free-stream values.

Outputs
-------
sim/gaussian-bump/setup/ (system, constant and 0; the mesh-gaussian-bump
stage copies it to sim/gaussian-bump/template and meshes it there)
"""

from __future__ import annotations

import json
import os
import re
import shutil

import numpy as np

OUT = "sim/gaussian-bump/setup"
PROFILES = "results/gaussian-bump-profiles.json"
RE_L = 2_000_000
NU = 1.0 / RE_L
H, X0 = 0.085, 0.195
X_IN, X_OUT = -0.8, 1.5
#: Block boundaries along x and cells in each block
X_BREAKS = (X_IN, -0.3, 0.7, X_OUT)
NX = (120, 500, 120)
NY = 200
#: First cell height: y+ about one at the DNS's largest C_f, 8.7e-3
Y1 = 5e-6
#: TMR quiet free stream at Mach 0.2
A2 = (1 / 0.2) ** 2
K_INF = 9e-9 * A2
OMEGA_INF = 1e-6 * A2 / NU
BETA = 0.075


def wall(x):
    return H * np.exp(-((x / X0) ** 2))


def wall_grading(length, n, first):
    lo, hi = 1.0 + 1e-9, 2.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        if length * (r - 1) / (r**n - 1) > first:
            lo = r
        else:
            hi = r
    return r ** (n - 1)


def block_mesh_dict():
    xs = X_BREAKS
    nb = len(xs)
    # Bottom vertices 0..nb-1 on the wall, top nb..2nb-1 at y = 1
    v = [(x, wall(x)) for x in xs] + [(x, 1.0) for x in xs]
    o = len(v)
    verts = [f"    ({x:.10g} {y:.10g} {z})" for z in (0, 0.01) for x, y in v]
    gy = wall_grading(1.0 - H, NY, Y1)
    # Outer blocks graded toward the middle one, whose spacing they meet
    dx_mid = (xs[2] - xs[1]) / NX[1]
    gx0 = 1.0 / wall_grading(xs[1] - xs[0], NX[0], dx_mid)
    gx2 = wall_grading(xs[3] - xs[2], NX[2], dx_mid)
    blocks = []
    for b, (nx, gx) in enumerate(zip(NX, (gx0, 1.0, gx2))):
        a, c = b, b + 1
        blocks.append(
            f"    hex ({a} {c} {c + nb} {a + nb} {a + o} {c + o} "
            f"{c + nb + o} {a + nb + o}) ({nx} {NY} 1) "
            f"simpleGrading ({gx:.8g} {gy:.8g} 1)"
        )
    edges = []
    for z, off in ((0, 0), (0.01, o)):
        for b in range(nb - 1):
            xx = np.linspace(xs[b], xs[b + 1], 60)[1:-1]
            pts = " ".join(f"({x:.10g} {wall(x):.10g} {z})" for x in xx)
            edges.append(f"    spline {b + off} {b + 1 + off} ({pts})")

    def quad(a, b):
        return f"            ({a} {b} {b + o} {a + o})"

    return BMD.format(
        verts="\n".join(verts),
        blocks="\n".join(blocks),
        edges="\n".join(edges),
        inlet=quad(0, nb),
        outlet=quad(nb - 1, 2 * nb - 1),
        top="\n".join(quad(nb + b, nb + b + 1) for b in range(nb - 1)),
        wall="\n".join(quad(b, b + 1) for b in range(nb - 1)),
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
    top
    {{
        type patch;
        faces
        (
{top}
        );
    }}
    bump
    {{
        type wall;
        faces
        (
{wall}
        );
    }}
);

defaultPatch
{{
    name frontAndBack;
    type empty;
}}
"""

HEADER = """FoamFile
{{
    version 2.0;
    format ascii;
    class {cls};
    object {obj};
}}

"""


def inflow():
    """The DNS's inflow line as RANS inlet profiles in height above the
    wall: U, V, k and omega."""
    with open(PROFILES) as f:
        p = json.load(f)["inflow"]
    y = np.array(p["y"]) - wall(X_IN)
    U, V = np.array(p["U"]), np.array(p["V"])
    k = 0.5 * (np.array(p["uu"]) + np.array(p["vv"]) + np.array(p["ww"]))
    k = np.maximum(k, K_INF)
    dudy = np.gradient(U, y)
    nut = -np.array(p["uv"]) / np.where(np.abs(dudy) > 1e-9, dudy, np.nan)
    good = np.isfinite(nut) & (nut > 1e-3 * NU)
    nut_i = np.interp(y, y[good], nut[good])
    w = np.maximum(k / np.maximum(nut_i, 1e-12), OMEGA_INF)
    with np.errstate(divide="ignore"):
        w_wall = 6 * NU / (BETA * y**2)
    w = np.where(y < y[good][0], np.maximum(w, w_wall), w)
    w[0] = w_wall[1] if y[0] == 0 else w_wall[0]
    # No higher than the wall value at the first cell's centre: the DNS's
    # points nearest the wall are far closer than any RANS cell, and the
    # wall formula there gave omega of order 1e33, which every model but SST
    # and kkL-omega failed to survive
    w = np.minimum(w, 6 * NU / (BETA * (Y1 / 2) ** 2))
    # Above the layer the DNS has no turbulence
    edge = y > y[np.argmax(U >= 0.99 * U.max())] * 1.5
    k[edge] = K_INF
    w[edge] = OMEGA_INF
    return y, U, V, k, w


def table(y, vals):
    if np.ndim(vals[0]) == 0:
        rows = " ".join(f"({yy:.8g} {v:.8g})" for yy, v in zip(y, vals))
    else:
        rows = " ".join(
            f"({yy:.8g} ({a:.8g} {b:.8g} 0))" for yy, (a, b) in zip(y, vals)
        )
    return rows


def profile_bc(rows):
    return (
        "        type fixedProfile;\n"
        "        profile table\n        (\n"
        f"            {rows}\n        );\n"
        "        direction (0 1 0);\n"
        f"        origin {wall(X_IN):.10g};\n"
    )


def field(name, cls, dims, internal, bcs):
    body = "".join(
        f"    {patch}\n    {{\n{bc}    }}\n" for patch, bc in bcs.items()
    )
    return (
        HEADER.format(cls=cls, obj=name)
        + f"dimensions {dims};\n\ninternalField {internal};\n\n"
        + "boundaryField\n{\n"
        + body
        + "    frontAndBack\n    {\n        type empty;\n    }\n}\n"
    )


def fields():
    y, U, V, k, w = inflow()
    zg = "        type zeroGradient;\n"
    io = (
        "        type inletOutlet;\n        inletValue $internalField;\n"
        "        value $internalField;\n"
    )
    u_inf = "(1 0 0)"
    return {
        "U": field(
            "U",
            "volVectorField",
            "[0 1 -1 0 0 0 0]",
            f"uniform {u_inf}",
            {
                "inlet": profile_bc(table(y, list(zip(U, V)))),
                "outlet": "        type inletOutlet;\n"
                f"        inletValue uniform {u_inf};\n"
                f"        value uniform {u_inf};\n",
                "top": "        type freestreamVelocity;\n"
                f"        freestreamValue uniform {u_inf};\n"
                f"        value uniform {u_inf};\n",
                "bump": "        type noSlip;\n",
            },
        ),
        "p": field(
            "p",
            "volScalarField",
            "[0 2 -2 0 0 0 0]",
            "uniform 0",
            {
                "inlet": zg,
                "outlet": "        type fixedValue;\n        value uniform 0;\n",
                "top": "        type freestreamPressure;\n"
                "        freestreamValue uniform 0;\n        value uniform 0;\n",
                "bump": zg,
            },
        ),
        "k": field(
            "k",
            "volScalarField",
            "[0 2 -2 0 0 0 0]",
            f"uniform {K_INF:.8g}",
            {
                "inlet": profile_bc(table(y, k)),
                "outlet": io,
                "top": io,
                "bump": "        type fixedValue;\n        value uniform 1e-12;\n",
            },
        ),
        "omega": field(
            "omega",
            "volScalarField",
            "[0 0 -1 0 0 0 0]",
            f"uniform {OMEGA_INF:.8g}",
            {
                "inlet": profile_bc(table(y, w)),
                "outlet": io,
                "top": io,
                "bump": "        type omegaWallFunction;\n"
                "        value $internalField;\n",
            },
        ),
        "nut": field(
            "nut",
            "volScalarField",
            "[0 2 -1 0 0 0 0]",
            "uniform 0",
            {
                "inlet": "        type calculated;\n        value uniform 0;\n",
                "outlet": "        type calculated;\n        value uniform 0;\n",
                "top": "        type calculated;\n        value uniform 0;\n",
                "bump": "        type nutLowReWallFunction;\n"
                "        value uniform 0;\n",
            },
        ),
    }


CONTROL = (
    HEADER.format(cls="dictionary", obj="controlDict")
    + """\
application     simpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         20000;
deltaT          1;
writeControl    timeStep;
writeInterval   1000;
purgeWrite      1;
writeFormat     ascii;
writePrecision  8;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable true;
"""
)

TRANSPORT = HEADER.format(cls="dictionary", obj="transportProperties") + (
    f"transportModel  Newtonian;\n\nnu              {NU:.10g};\n"
)


def main():
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    for sub in ("system", "constant", "0"):
        os.makedirs(os.path.join(OUT, sub))
    with open(os.path.join(OUT, "system", "blockMeshDict"), "w") as f:
        f.write(block_mesh_dict())
    with open(os.path.join(OUT, "system", "controlDict"), "w") as f:
        f.write(CONTROL)
    # The plate cases' schemes and solver settings, with residual targets
    # tight enough for a separated flow
    shutil.copy(
        os.path.join("sim", "system", "fvSchemes"),
        os.path.join(OUT, "system", "fvSchemes"),
    )
    with open(os.path.join("sim", "system", "fvSolution")) as f:
        text = f.read()
    text, n = re.subn(
        r"residualControl\s*\{.*?\}",
        "residualControl\n    {\n        p               1e-6;\n"
        "        U               1e-6;\n"
        '        "(k|omega|gamma|gammaInt|ReThetat|kt|kl|TuLE)" 1e-6;\n'
        "    }",
        text,
        count=1,
        flags=re.S,
    )
    assert n == 1, "fvSolution has no residualControl block"
    with open(os.path.join(OUT, "system", "fvSolution"), "w") as f:
        f.write(text)
    with open(os.path.join(OUT, "constant", "transportProperties"), "w") as f:
        f.write(TRANSPORT)
    for name, text in fields().items():
        with open(os.path.join(OUT, "0", name), "w") as f:
            f.write(text)
    print(
        f"Wrote {OUT}: wall grading "
        f"{wall_grading(1.0 - H, NY, Y1):.4g}, k_inf {K_INF:.3g}"
    )


if __name__ == "__main__":
    main()
