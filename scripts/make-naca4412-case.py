#!/usr/bin/env python
"""Write an OpenFOAM case for the NACA 4412 at 5 degrees and Re_c = 400,000,
the flow of the KTH well-resolved LES (Vinuesa2018).

A two-dimensional structured C-mesh of six blocks: each surface is split at
SPLIT chords into a front and a rear block, since blockMesh allows one
curved edge between two vertices, and the wake behind the trailing edge is
split at the cut. The far field is an arc of FAR chords about the trailing
edge, closed by a straight outlet FAR chords downstream. The first cell is
sized for y+ about one at the LES's largest skin friction.

The free stream follows the NASA Turbulence Modeling Resource's guidance
for a quiet free stream at Mach 0.15 (k = 9e-9 a^2, omega = 1e-6 a^2/nu),
since the LES has none. The LES's boundary layers are tripped at
x/c = 0.1; the case is scored from x/c = 0.15, where its statistics begin.

Outputs
-------
sim/naca4412/setup/ (system, constant and 0; the mesh-naca4412 stage
copies it to sim/naca4412/template and meshes it there)
"""

from __future__ import annotations

import os
import re
import shutil

import numpy as np

OUT = "sim/naca4412/setup"
ALPHA = 5.0
RE_C = 400_000
NU = 1.0 / RE_C
FAR = 20.0
SPLIT = 0.3
#: Cells: along the front and rear of each surface, normal to it, and
#: along the wake
N_FRONT, N_REAR, N_NORMAL, N_WAKE = 80, 160, 120, 100
#: First cell height in chords: y+ about 1 at the LES's largest C_f, 6e-3
Y1 = 4e-5
#: Last-over-first cell ratio along the surfaces and the wake
G_FRONT, G_REAR, G_WAKE = 8.0, 0.25, 150.0
#: TMR quiet free stream at Mach 0.15, in units of U_inf and c
A2 = (1 / 0.15) ** 2
K_INF = 9e-9 * A2
OMEGA_INF = 1e-6 * A2 / NU


def surface(n, x):
    """NACA 4412 upper and lower surfaces at chordwise stations x, closed
    trailing edge."""
    m, p, t = 0.04, 0.4, 0.12
    yt = (
        5
        * t
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1036 * x**4
        )
    )
    yc = np.where(
        x < p,
        m / p**2 * (2 * p * x - x**2),
        m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2),
    )
    dyc = np.where(
        x < p, 2 * m / p**2 * (p - x), 2 * m / (1 - p) ** 2 * (p - x)
    )
    th = np.arctan(dyc)
    upper = np.c_[x - yt * np.sin(th), yc + yt * np.cos(th)]
    lower = np.c_[x + yt * np.sin(th), yc - yt * np.cos(th)]
    return upper, lower


def wall_grading(length, n, first):
    """simpleGrading's last-over-first ratio for n geometric cells over
    length starting at first."""
    lo, hi = 1.0 + 1e-9, 2.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        if length * (r - 1) / (r**n - 1) > first:
            lo = r
        else:
            hi = r
    return r ** (n - 1)


def far(phi):
    """A point on the far-field arc about the trailing edge."""
    a = np.radians(phi)
    return (1 + FAR * np.cos(a), FAR * np.sin(a))


def block_mesh_dict():
    # Points on each surface, clustered at the leading edge, split at SPLIT
    beta = np.linspace(0, np.pi, 401)
    xs = 0.5 * (1 - np.cos(beta))
    up, lo = surface(401, xs)
    i_up = int(np.argmin(np.abs(up[:, 0] - SPLIT)))
    i_lo = int(np.argmin(np.abs(lo[:, 0] - SPLIT)))
    v = [
        (0.0, 0.0),  # 0 leading edge
        (1.0, 0.0),  # 1 trailing edge
        tuple(up[i_up]),  # 2 suction side, split
        tuple(lo[i_lo]),  # 3 pressure side, split
        far(180),  # 4 far field ahead
        far(135),  # 5 far field above the split
        far(90),  # 6 far field above the trailing edge
        far(225),  # 7 far field below the split
        far(270),  # 8 far field below the trailing edge
        (1 + FAR, 0.0),  # 9 outlet on the cut
        (1 + FAR, FAR),  # 10 outlet top
        (1 + FAR, -FAR),  # 11 outlet bottom
    ]
    o = len(v)
    verts = [f"    ({x:.10g} {y:.10g} {z})" for z in (0, 0.01) for x, y in v]
    g = wall_grading(FAR, N_NORMAL, Y1)

    def hexb(a, b, c, d, nx, gx, gy):
        return (
            f"    hex ({a} {b} {c} {d} {a + o} {b + o} {c + o} {d + o}) "
            f"({nx} {N_NORMAL} 1) simpleGrading ({gx:.8g} {gy:.8g} 1)"
        )

    # Upper blocks: x along the surface toward the trailing edge, y from
    # the wall out. Lower blocks keep x toward the trailing edge, so y runs
    # from the far field in, and their wall-normal grading is inverted
    blocks = [
        hexb(0, 2, 5, 4, N_FRONT, G_FRONT, g),
        hexb(2, 1, 6, 5, N_REAR, G_REAR, g),
        hexb(1, 9, 10, 6, N_WAKE, G_WAKE, g),
        hexb(4, 7, 3, 0, N_FRONT, G_FRONT, 1 / g),
        hexb(7, 8, 1, 3, N_REAR, G_REAR, 1 / g),
        hexb(8, 11, 9, 1, N_WAKE, G_WAKE, 1 / g),
    ]

    def spline(a, b, arr, z, off):
        p = " ".join(f"({x:.10g} {y:.10g} {z})" for x, y in arr)
        return f"    spline {a + off} {b + off} ({p})"

    def arc(a, b, phi, z, off):
        x, y = far(phi)
        return f"    arc {a + off} {b + off} ({x:.10g} {y:.10g} {z})"

    edges = []
    for z, off in ((0, 0), (0.01, o)):
        edges += [
            spline(0, 2, up[1:i_up], z, off),
            spline(2, 1, up[i_up + 1 : -1], z, off),
            spline(0, 3, lo[1:i_lo], z, off),
            spline(3, 1, lo[i_lo + 1 : -1], z, off),
            arc(4, 5, 157.5, z, off),
            arc(5, 6, 112.5, z, off),
            arc(4, 7, 202.5, z, off),
            arc(7, 8, 247.5, z, off),
        ]

    def quad(a, b):
        return f"            ({a} {b} {b + o} {a + o})"

    far_faces = [
        quad(4, 5),
        quad(5, 6),
        quad(6, 10),
        quad(4, 7),
        quad(7, 8),
        quad(8, 11),
    ]
    out_faces = [quad(9, 10), quad(11, 9)]
    wing = [quad(0, 2), quad(2, 1), quad(0, 3), quad(3, 1)]
    return BMD.format(
        verts="\n".join(verts),
        blocks="\n".join(blocks),
        edges="\n".join(edges),
        far="\n".join(far_faces),
        out="\n".join(out_faces),
        wing="\n".join(wing),
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
    farfield
    {{
        type patch;
        faces
        (
{far}
        );
    }}
    outlet
    {{
        type patch;
        faces
        (
{out}
        );
    }}
    wing
    {{
        type wall;
        faces
        (
{wing}
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


def field(name, cls, dims, internal, far_bc, out_bc, wing_bc):
    return (
        HEADER.format(cls=cls, obj=name)
        + f"dimensions {dims};\n\ninternalField {internal};\n\n"
        + "boundaryField\n{\n"
        + f"    farfield\n    {{\n{far_bc}    }}\n"
        + f"    outlet\n    {{\n{out_bc}    }}\n"
        + f"    wing\n    {{\n{wing_bc}    }}\n"
        + "    frontAndBack\n    {\n        type empty;\n    }\n}\n"
    )


def fields():
    a = np.radians(ALPHA)
    u = f"({np.cos(a):.10g} {np.sin(a):.10g} 0)"
    io = (
        "        type inletOutlet;\n        inletValue $internalField;\n"
        "        value $internalField;\n"
    )
    return {
        "U": field(
            "U",
            "volVectorField",
            "[0 1 -1 0 0 0 0]",
            f"uniform {u}",
            f"        type freestreamVelocity;\n        freestreamValue "
            f"uniform {u};\n        value uniform {u};\n",
            f"        type inletOutlet;\n        inletValue uniform {u};\n"
            f"        value uniform {u};\n",
            "        type noSlip;\n",
        ),
        "p": field(
            "p",
            "volScalarField",
            "[0 2 -2 0 0 0 0]",
            "uniform 0",
            "        type freestreamPressure;\n        freestreamValue "
            "uniform 0;\n        value uniform 0;\n",
            "        type fixedValue;\n        value uniform 0;\n",
            "        type zeroGradient;\n",
        ),
        "k": field(
            "k",
            "volScalarField",
            "[0 2 -2 0 0 0 0]",
            f"uniform {K_INF:.8g}",
            io,
            io,
            "        type fixedValue;\n        value uniform 1e-12;\n",
        ),
        "omega": field(
            "omega",
            "volScalarField",
            "[0 0 -1 0 0 0 0]",
            f"uniform {OMEGA_INF:.8g}",
            io,
            io,
            "        type omegaWallFunction;\n        value $internalField;\n",
        ),
        "nut": field(
            "nut",
            "volScalarField",
            "[0 2 -1 0 0 0 0]",
            "uniform 0",
            "        type calculated;\n        value uniform 0;\n",
            "        type calculated;\n        value uniform 0;\n",
            "        type nutLowReWallFunction;\n        value uniform 0;\n",
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
    # The plate cases' schemes and solver settings, including omega's
    # tight, never-skipped solve, with residual targets tight enough for an
    # airfoil's pressure distribution: the plate's let a trial stop after a
    # few hundred iterations
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
        f"Wrote {OUT}: k_inf {K_INF:.3g}, omega_inf {OMEGA_INF:.3g}, "
        f"wall grading {wall_grading(FAR, N_NORMAL, Y1):.4g}"
    )


if __name__ == "__main__":
    main()
