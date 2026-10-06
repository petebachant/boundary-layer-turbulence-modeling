"""The Gaussian bump at Re_L = 2 million in OpenFOAM, as a benchmark plugin.

Uzun and Malik's DNS (Uzun2022) of a turbulent boundary layer accelerated
over the Boeing bump's front and separated from its smooth lee, from about
x/L = 0.10 to 0.42: separation from a smooth surface, set by the pressure
gradient alone, which is what RANS closures most often get wrong. The case
is the two-dimensional domain scripts/make-gaussian-bump-case.py writes and
the mesh-gaussian-bump stage meshes, with the DNS's own inflow profile and
a free stream on top at y = L, as in the DNS.

Scored on the wall C_f, the separation and reattachment points, the wall
C_p, and velocity profiles at the DNS's stations.

The inflow is a fully turbulent layer, so a model's transported activation
or intermittency enters at one; the harness derives those fields from k's
inlet, which here is a measured profile, so they are rewritten for the
inlet after it runs. Registered by naming this module in
RANS_BENCH_PLUGINS, after any plugin that wraps the OpenFOAM case set-up.
"""

from __future__ import annotations

import json
import os
import re
import shutil

import numpy as np

from .cases import openfoam as _of
from .cases.base import BenchmarkCase
from .foam_body import patch_values, prepare_case, run_info, solve
from .registry import TIER_OPENFOAM, register_case

TEMPLATE = "sim/gaussian-bump/template"
DATA = "data/gaussian-bump-dns"
PROFILES = "results/gaussian-bump-profiles.json"
RE_L = 2_000_000
H, X0 = 0.085, 0.195
#: Where C_f and C_p are scored, and the upstream stretch that scales C_f
X_SCORE = (-0.6, 0.9)
X_REF = (-0.6, -0.2)
#: Velocity profiles are scored up to this height above the wall, in L
Y_PROFILE = 0.1


def _wall(x):
    return H * np.exp(-((x / X0) ** 2))


def _read_dat(path):
    return np.loadtxt(path, skiprows=2)


def _set_inlet(case_dir, name, bc):
    """Replace the inlet block of one field's boundary conditions."""
    path = os.path.join(case_dir, "0", name)
    if not os.path.isfile(path):
        return
    with open(path) as f:
        text = f.read()
    head, _, body = text.partition("boundaryField")
    body, n = re.subn(
        r"(\n\s*inlet\s*\n\s*)\{.*?\n\s*\}",
        lambda m: m.group(1) + "{\n" + bc + "    }",
        body,
        count=1,
        flags=re.S,
    )
    assert n == 1, f"{path} has no inlet block"
    with open(path, "w") as f:
        f.write(head + "boundaryField" + body)


def _profile_table(case_dir, name):
    """The y and value columns of a field's inlet fixedProfile table."""
    with open(os.path.join(case_dir, "0", name)) as f:
        text = f.read()
    block = text[text.index("table") :]
    rows = re.findall(r"\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)", block)
    origin = float(re.search(r"origin\s+([-\d.eE+]+);", block).group(1))
    a = np.array(rows, dtype=float)
    return a[:, 0], a[:, 1], origin


def _fix_inlets(case_dir):
    """Inlet conditions for the fields the harness derives from k's."""
    one = "        type fixedValue;\n        value uniform 1;\n"
    zero = "        type fixedValue;\n        value uniform 0;\n"
    _set_inlet(case_dir, "gamma", one)
    _set_inlet(case_dir, "gammaInt", one)
    _set_inlet(case_dir, "ReThetat", "        type zeroGradient;\n")
    _set_inlet(case_dir, "kl", zero)
    if os.path.isfile(os.path.join(case_dir, "0", "epsilon")):
        y, k, origin = _profile_table(case_dir, "k")
        _, w, _ = _profile_table(case_dir, "omega")
        rows = " ".join(
            f"({a:.8g} {0.09 * b * c:.8g})" for a, b, c in zip(y, k, w)
        )
        _set_inlet(
            case_dir,
            "epsilon",
            "        type fixedProfile;\n        profile table\n        (\n"
            f"            {rows}\n        );\n"
            f"        direction (0 1 0);\n        origin {origin:.10g};\n",
        )
    if os.path.isfile(os.path.join(case_dir, "0", "TuLE")):
        with open(os.path.join(case_dir, "0", "TuLE")) as f:
            tu = re.search(r"internalField\s+uniform\s+(\S+);", f.read())
        _set_inlet(
            case_dir,
            "TuLE",
            f"        type fixedValue;\n        value uniform {tu.group(1)};\n",
        )


def _crossings(x, cf):
    """Separation and reattachment: the first downward zero crossing of C_f
    past the bump's apex and the next upward one. With no separation, both
    are put at C_f's minimum there, a bubble of zero length."""
    m = x > 0
    xs, c = x[m], cf[m]
    down = np.flatnonzero((c[:-1] >= 0) & (c[1:] < 0))
    if len(down) == 0:
        x_min = float(xs[int(np.argmin(c))])
        return x_min, x_min

    def cross(i):
        return float(xs[i] - c[i] * (xs[i + 1] - xs[i]) / (c[i + 1] - c[i]))

    i = int(down[0])
    up = np.flatnonzero((c[i:-1] < 0) & (c[i + 1 :] >= 0))
    x_reat = cross(i + int(up[0])) if len(up) else float(xs[-1])
    return cross(i), x_reat


class GaussianBumpOpenFoam(BenchmarkCase):
    family = "smooth-separation"
    reference = "Uzun2022"
    fidelity = "dns"
    TARGETS = {
        "cf_abs_rms": 0.05,
        "x_sep_err": 0.02,
        "x_reat_err": 0.02,
        "cp_rms": 0.02,
        "U_rms": 0.01,
    }

    def __init__(self, root="."):
        self.root = root
        self.name = f"gaussian-bump-rel-{RE_L}-openfoam"
        self.template = os.path.join(root, TEMPLATE)
        if not os.path.isdir(
            os.path.join(self.template, "constant", "polyMesh")
        ):
            raise FileNotFoundError(f"{self.template} has no mesh")
        cf = _read_dat(os.path.join(root, DATA, "SpeedBump-ReL-2M-Cf.dat"))
        cp = _read_dat(os.path.join(root, DATA, "SpeedBump-ReL-2M-Cp.dat"))
        self.x_cf, self.cf = cf[:, 0], cf[:, 1]
        self.x_cp, self.cp = cp[:, 0], cp[:, 1]
        with open(os.path.join(root, PROFILES)) as f:
            self.stations = json.load(f)["stations"]
        self.x_sep, self.x_reat = _crossings(self.x_cf, self.cf)
        self.last_case_dir = None

    def case_dir(self, closure_name):
        return os.path.join(
            self.root, _of.CASES_DIR, f"{self.name}-{closure_name}"
        )

    def run(self, closure):
        model = getattr(closure, "openfoam_model", None)
        name = getattr(closure, "bench_name", None) or "closure"
        if not model:
            raise NotImplementedError(f"{name} has no OpenFOAM model")
        case_dir = self.case_dir(name)
        if os.path.isdir(case_dir):
            shutil.rmtree(case_dir)
        shutil.copytree(self.template, case_dir)
        _of.write_turbulence_properties(case_dir, model)
        _of.write_model_coeffs(case_dir, model, self.root)
        _of.ensure_libs(case_dir, model)
        _of.ensure_model_fields(case_dir, model)
        _fix_inlets(case_dir)
        _of.prepare_fv_solution(case_dir, self.family)
        prepare_case(case_dir)
        self.last_case_dir = case_dir
        return solve(self, case_dir, self.root)

    def read_solution(self, case_dir):
        t = _of.latest_time_dir(case_dir)
        if float(t) == 0.0:
            raise RuntimeError(f"no solution written in {case_dir}")
        d = os.path.join(case_dir, t)
        sol = {
            "C": _of.read_field(os.path.join(d, "C"))[:, :2],
            "U": _of.read_field(os.path.join(d, "U"))[:, :2],
            "p": _of.read_field(os.path.join(d, "p")),
            "Cw": patch_values(os.path.join(d, "C"), "bump")[:, :2],
            "tau": patch_values(os.path.join(d, "wallShearStress"), "bump")[
                :, :2
            ],
            "time": t,
        }
        with open(
            os.path.join(case_dir, "log.simpleFoam"), errors="replace"
        ) as f:
            sol["converged"] = "SIMPLE solution converged" in f.read()
        return sol

    def wall_curves(self, sol):
        """C_f and C_p along the wall, ordered in x."""
        from scipy.spatial import cKDTree

        cw, tau = sol["Cw"], sol["tau"]
        order = np.argsort(cw[:, 0])
        cw, tau = cw[order], tau[order]
        x = cw[:, 0]
        # Tangent along the wall, downstream; wallShearStress is the force
        # the fluid feels, so C_f is minus twice its tangential component
        slope = np.gradient(_wall(x), x)
        t = np.c_[np.ones_like(x), slope] / np.hypot(1, slope)[:, None]
        cf = -2.0 * np.sum(tau * t, axis=1)
        _, j = cKDTree(sol["C"]).query(cw)
        cp = 2.0 * sol["p"][j]
        return x, cf, cp

    def errors(self, sol):
        from scipy.interpolate import LinearNDInterpolator

        if not np.all(np.isfinite(sol["U"])):
            return {"cf_abs_rms": np.inf}
        x, cf, cp = self.wall_curves(sol)
        xs = self.x_cf[(self.x_cf >= X_SCORE[0]) & (self.x_cf <= X_SCORE[1])]
        cf_d = np.interp(xs, self.x_cf, self.cf)
        ref = np.mean(
            np.abs(self.cf[(self.x_cf >= X_REF[0]) & (self.x_cf <= X_REF[1])])
        )
        cf_m = np.interp(xs, x, cf)
        cp_d = np.interp(xs, self.x_cp, self.cp)
        cp_m = np.interp(xs, x, cp)
        x_sep, x_reat = _crossings(x, cf)
        interp = LinearNDInterpolator(sol["C"], sol["U"][:, 0])
        u_err = []
        for st in self.stations.values():
            sx, sy = np.array(st["x"]), np.array(st["y"])
            m = sy - sy[0] <= Y_PROFILE
            um = interp(np.c_[sx[m], sy[m]])
            ok = np.isfinite(um)
            u_err.append(
                np.sqrt(np.mean((um[ok] - np.array(st["U"])[m][ok]) ** 2))
            )
        return {
            "cf_abs_rms": float(np.sqrt(np.mean((cf_m - cf_d) ** 2)) / ref),
            "x_sep_err": abs(x_sep - self.x_sep),
            "x_reat_err": abs(x_reat - self.x_reat),
            "cp_rms": float(np.sqrt(np.mean((cp_m - cp_d) ** 2))),
            "U_rms": float(np.mean(u_err)),
            "x_sep": x_sep,
            "x_reat": x_reat,
            **run_info(sol),
        }

    def describe(self):
        d = super().describe()
        d.update(
            {
                "Re_L": RE_L,
                "x_sep_dns": self.x_sep,
                "x_reat_dns": self.x_reat,
                "x_score": list(X_SCORE),
                "template": os.path.relpath(self.template, self.root),
            }
        )
        return d


register_case(
    f"gaussian-bump-rel-{RE_L}-openfoam",
    lambda root=".": GaussianBumpOpenFoam(root=root),
    family="smooth-separation",
    tier=TIER_OPENFOAM,
    reference="Uzun2022",
    description=(
        "Boeing Gaussian bump at Re_L = 2 million, the whole flow solved in "
        "OpenFOAM with the DNS's inflow, scored on wall C_f, separation and "
        "reattachment, C_p and velocity profiles against the DNS."
    ),
)
