"""The NACA 4412 at Re_c = 400,000 in OpenFOAM, as a benchmark plugin.

The fast tier marches the suction-side boundary layer with the LES's own
edge velocity handed to it (pypkg/cases/naca4412.py). This case solves the
whole flow around the wing, at 5 degrees in a quiet free stream, on the
C-mesh scripts/make-naca4412-case.py writes and the mesh-naca4412 stage
meshes, so the pressure distribution is the model's own. It is scored on
the suction side against the KTH well-resolved LES (Vinuesa2018) at the
LES's stations, with the fast tier's metrics plus the edge velocity.

The LES's layers are tripped at x/c = 0.1 and its statistics start at
x/c = 0.15; a RANS model is not tripped, and the harness starts the
clipping closures' activation fraction, and the free stream's, at one, as
on the Closure Challenge cases. Scoring starts at x/c = 0.2.

Registered by naming this module in RANS_BENCH_PLUGINS, so adding it leaves
pypkg/cases, and every stage that reads it, unchanged. Load it after any
plugin that wraps the OpenFOAM case set-up, so those wrappers apply here.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess

import numpy as np

from .cases import openfoam as _of
from .cases.base import BenchmarkCase, rel_rms
from .cases.naca4412 import load_wing
from .registry import TIER_OPENFOAM, register_case

TEMPLATE = "sim/naca4412/template"
RE_C = 400_000
X_MIN = 0.20
AFT_XC = 0.90
#: Wall-normal sampling reaches this many LES delta99 above the surface
Y_SPAN = 2.5


def _patch_values(path, patch):
    """The nonuniform values of one boundary patch of an ASCII field."""
    with open(path) as f:
        text = f.read()
    body = text[text.index("boundaryField") :]
    m = re.search(rf"\n\s*{patch}\s*\n\s*\{{", body)
    if m is None:
        raise ValueError(f"{path} has no patch {patch}")
    block = body[m.end() :]
    v = re.search(
        r"value\s+nonuniform\s+List<(scalar|vector)>\s*(\d+)\s*\(", block
    )
    n = int(v.group(2))
    data = block[v.end() :]
    if v.group(1) == "scalar":
        return np.array(data.split(")")[0].split(), dtype=float)[:n]
    rows = re.findall(r"\(([^()]*)\)", data[: data.index("\n)")])[:n]
    return np.array([r.split() for r in rows], dtype=float)


def _edge_metrics(y, u):
    """theta and H of a tangential profile, with the edge at its maximum,
    as pypkg/dns_case.bl_metrics measures it."""
    j = int(np.argmax(u)) + 1
    ue = float(u[j - 1])
    f = np.clip(u[:j] / ue, 0.0, 1.0)
    theta = np.trapezoid(f * (1 - f), y[:j])
    dstar = np.trapezoid(1 - f, y[:j])
    return theta, dstar / max(theta, 1e-12)


class Naca4412OpenFoam(BenchmarkCase):
    family = "wing-apg"
    reference = "Vinuesa2018"
    fidelity = "les"
    # The fast tier's targets, so a closure's two NACA 4412 scores mean the
    # same thing, plus the edge velocity, which an elliptic solver predicts
    # and the fast tier is handed. The edge velocity is scored rather than
    # the wall pressure because the LES's pressure has a different
    # reference: a trial SST run matched its edge velocity to within one
    # percent at every station while its wall C_p sat 0.13 to 0.16 apart
    TARGETS = {
        "cf_rel_rms": 0.02,
        "U_rms": 0.01,
        "theta_rel_rms": 0.05,
        "H_rel_rms": 0.02,
        "cf_aft_abs": 0.10,
        "H_aft_rel_rms": 0.05,
        "ue_rel_rms": 0.01,
    }

    def __init__(self, root="."):
        self.root = root
        self.name = f"naca4412-rec-{RE_C}-openfoam"
        self.template = os.path.join(root, TEMPLATE)
        if not os.path.isdir(
            os.path.join(self.template, "constant", "polyMesh")
        ):
            raise FileNotFoundError(f"{self.template} has no mesh")
        st = load_wing(RE_C, root=root)
        st = [s for s in st if s["xa"] >= X_MIN]
        self.stations = st
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
        # Looked up at call time, so a plugin's wrappers apply
        _of.write_turbulence_properties(case_dir, model)
        _of.write_model_coeffs(case_dir, model, self.root)
        _of.ensure_libs(case_dir, model)
        _of.ensure_model_fields(case_dir, model)
        _of.prepare_fv_solution(case_dir, self.family)
        rel = os.path.relpath(case_dir, self.root)
        cmd = [
            "calkit",
            "xenv",
            "-n",
            "blsim",
            "--no-check",
            "--",
            "bash",
            "-c",
            f"source sim/foam-env.sh && cd {rel} && "
            "FOAM_SIGFPE=false simpleFoam > log.simpleFoam 2>&1 && "
            "postProcess -func writeCellCentres -latestTime "
            "> log.cellCentres 2>&1 && "
            "simpleFoam -postProcess -func wallShearStress -latestTime "
            "> log.wallShearStress 2>&1",
        ]
        subprocess.run(cmd, cwd=self.root, check=True)
        self.last_case_dir = case_dir
        return self.read_solution(case_dir)

    def read_solution(self, case_dir):
        t = _of.latest_time_dir(case_dir)
        if float(t) == 0.0:
            raise RuntimeError(f"no solution written in {case_dir}")
        d = os.path.join(case_dir, t)
        sol = {
            "C": _of.read_field(os.path.join(d, "C"))[:, :2],
            "U": _of.read_field(os.path.join(d, "U"))[:, :2],
            "p": _of.read_field(os.path.join(d, "p")),
            "Cw": _patch_values(os.path.join(d, "C"), "wing")[:, :2],
            "tau": _patch_values(os.path.join(d, "wallShearStress"), "wing")[
                :, :2
            ],
            "time": t,
        }
        with open(
            os.path.join(case_dir, "log.simpleFoam"), errors="replace"
        ) as f:
            sol["converged"] = "SIMPLE solution converged" in f.read()
        return sol

    def profiles(self, sol):
        """C_f, C_p, theta, H and the tangential profile at each station,
        from the model's solution."""
        from scipy.interpolate import LinearNDInterpolator
        from scipy.spatial import cKDTree

        C, U, Cw = sol["C"], sol["U"], sol["Cw"]
        # Suction side: the wall faces above the chord line's camber
        x = Cw[:, 0]
        upper = Cw[:, 1] > _camber(x)
        cw_u = Cw[upper]
        order = np.argsort(cw_u[:, 0])
        cw_u = cw_u[order]
        tau_u = sol["tau"][upper][order]
        tree = cKDTree(C)
        out = []
        for s in self.stations:
            xa = s["xa"]
            # Tangent from the neighboring wall faces, pointing downstream
            i = int(np.clip(np.searchsorted(cw_u[:, 0], xa), 1, len(cw_u) - 1))
            t = cw_u[i] - cw_u[i - 1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            w = (xa - cw_u[i - 1, 0]) / (cw_u[i, 0] - cw_u[i - 1, 0])
            p0 = cw_u[i - 1] + w * (cw_u[i] - cw_u[i - 1])
            tau = tau_u[i - 1] + w * (tau_u[i] - tau_u[i - 1])
            cf = 2.0 * abs(float(tau @ t))
            yn = s["y"][s["y"] <= Y_SPAN * s["d99"]]
            pts = p0 + yn[:, None] * n
            near = tree.query_ball_point(p0, 1.2 * Y_SPAN * s["d99"] + 0.01)
            interp = LinearNDInterpolator(C[near], U[near] @ t)
            ut = interp(pts)
            ut[0] = 0.0
            _, j0 = tree.query(p0 + 1e-9 * n)
            cp = 2.0 * float(sol["p"][j0])
            th, H = _edge_metrics(yn, np.nan_to_num(ut))
            out.append(
                {
                    "xa": xa,
                    "cf": cf,
                    "cp": cp,
                    "ue": float(np.nanmax(ut)),
                    "theta": th,
                    "H": H,
                    "y": yn,
                    "U": ut,
                }
            )
        return out

    def errors(self, sol):
        prof = self.profiles(sol)
        st = self.stations
        if not all(np.all(np.isfinite(p["U"][1:])) for p in prof):
            return {"cf_rel_rms": np.inf}
        fwd = [i for i, s in enumerate(st) if s["xa"] < AFT_XC]
        aft = [i for i, s in enumerate(st) if s["xa"] >= AFT_XC]
        cf = np.array([p["cf"] for p in prof])
        cf_d = np.array([s["cf"] for s in st])
        les = [
            _edge_metrics(
                s["y"][s["y"] <= Y_SPAN * s["d99"]],
                s["U"][s["y"] <= Y_SPAN * s["d99"]],
            )
            for s in st
        ]
        th = np.array([p["theta"] for p in prof])
        H = np.array([p["H"] for p in prof])
        th_d = np.array([v[0] for v in les])
        H_d = np.array([v[1] for v in les])
        u_err = []
        for i in fwd:
            s, p = st[i], prof[i]
            m = p["y"] <= s["d99"]
            u_err.append(
                np.sqrt(
                    np.mean(
                        ((p["U"][m] - s["U"][: len(p["y"])][m]) / s["Ue"]) ** 2
                    )
                )
            )
        ue = np.array([p["ue"] for p in prof])
        ue_d = np.array([s["Ue"] for s in st])
        return {
            "cf_rel_rms": rel_rms(cf[fwd], cf_d[fwd]),
            "U_rms": float(np.mean(u_err)),
            "theta_rel_rms": rel_rms(th[fwd], th_d[fwd]),
            "H_rel_rms": rel_rms(H[fwd], H_d[fwd]),
            "cf_aft_abs": float(
                np.sqrt(np.mean((cf[aft] - cf_d[aft]) ** 2))
                / np.mean(cf_d[fwd])
            ),
            "H_aft_rel_rms": rel_rms(H[aft], H_d[aft]),
            "ue_rel_rms": rel_rms(ue, ue_d),
        }

    def describe(self):
        d = super().describe()
        d.update(
            {
                "Re_c": RE_C,
                "x_c_min": X_MIN,
                "aft_x_c": AFT_XC,
                "n_stations": len(self.stations),
                "template": os.path.relpath(self.template, self.root),
            }
        )
        return d


def _camber(x):
    m, p = 0.04, 0.4
    x = np.clip(x, 0, 1)
    return np.where(
        x < p,
        m / p**2 * (2 * p * x - x**2),
        m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * x - x**2),
    )


register_case(
    f"naca4412-rec-{RE_C}-openfoam",
    lambda root=".": Naca4412OpenFoam(root=root),
    family="wing-apg",
    tier=TIER_OPENFOAM,
    reference="Vinuesa2018",
    description=(
        "NACA 4412 at 5 degrees and Re_c = 400,000, the whole flow solved "
        "in OpenFOAM, scored on the suction side against the KTH LES."
    ),
)
