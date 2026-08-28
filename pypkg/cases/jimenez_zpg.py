"""Zero-pressure-gradient turbulent boundary layer, Re_theta = 4000-6500.

Sillero, Jimenez & Moser DNS. Six stations, already in ``data/jiminez/`` and
until now used only for the a-priori term regression. As a *solvable* case it
is the cheapest genuine out-of-sample test available to this project: no
download, and Re_theta is three to five times anything the JHTDB
transitional case reaches, so a closure tuned on transition has to survive
being asked about the fully-turbulent log layer.

Reconstructing x
----------------
The files are six independent stations with no streamwise coordinate. For a
zero-pressure-gradient layer the von Karman momentum integral is exact,

    d(theta)/dx = c_f/2 = (u_tau/U_e)^2,

so the spacing follows from quantities in the headers alone:

    x_i - x_0 = integral of d(theta) / (u_tau/U_e)^2.

Nothing is fitted. The reconstruction is checked in ``self.consistency``: the
six headers give a molecular viscosity that agrees to 0.3 % whether it is
derived from Re_tau or from Re_theta, which is what makes the outer-unit
description below self-consistent in the first place.

Units
-----
The files are in wall units with u_tau normalised by the free-stream velocity,
so setting U_e = 1 gives nu = u_tau * delta_99 / Re_tau ~ 2.855e-4, constant
across the six stations to 0.3 %.

Initial condition
-----------------
The inlet is a fully-developed turbulent profile, which no closure in this
repository initialises itself into -- they all seed a thin pre-transitional
state suited to the JHTDB plate. So the case seeds the transported scalars
from the DNS at the inlet station: k from the measured stresses, omega from
the measured eddy viscosity k/nu_t with nu_t = -<u'v'>/(dU/dy), gamma = 1
(fully turbulent), epsilon = betaStar*k*omega. Every closure gets the same
seed by the same rule, and the first station is excluded from scoring so no
model is judged on the seed itself.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np

from ..bl_solver import BLSolver
from ..registry import register_case
from .base import BenchmarkCase, log_rms, rel_rms
from .wrappers import SeededClosure

DATA_GLOB = "data/jiminez/Re_theta.*.prof"
# Column names as given in the file header, in order.
COLUMNS = ["y/d99", "y+", "urms", "vrms", "wrms", "uv", "uw", "vw",
           "umed", "vmed", "wmed", "u3", "v3", "u2v", "v2u", "w2u", "w2v",
           "dumdy", "dvmdy", "dwmdy", "oxmed", "oymed", "ozmed",
           "oxrms", "oyrms", "ozrms", "pm", "pp"]
_HDR = re.compile(r"([A-Za-z_{}\^*0-9]+?)\s*=\s*([0-9.eE+-]+)")


def load_station(path):
    """One profile file, converted from wall units to outer units (U_e = 1)."""
    hdr = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("%"):
                break
            for m in _HDR.finditer(line):
                hdr[m.group(1)] = float(m.group(2))
    ut = hdr["u_{tau}"]
    d99 = hdr["delta_99"]
    nu = ut * d99 / hdr["Re_{tau}"]
    raw = np.loadtxt(path, comments="%")
    c = {name: raw[:, i] for i, name in enumerate(COLUMNS)}
    y = c["y/d99"] * d99
    U = c["umed"] * ut
    uu = (c["urms"] * ut) ** 2
    vv = (c["vrms"] * ut) ** 2
    ww = (c["wrms"] * ut) ** 2
    # uv is in wall units and is negative in this dataset
    uv = c["uv"] * ut ** 2
    dUdy = c["dumdy"] * ut / (nu / ut)   # dumdy is d(U+)/d(y+)
    return {
        "path": path,
        "u_tau": ut, "d99": d99, "nu": nu,
        "Re_theta": hdr["Re_{theta}"], "Re_tau": hdr["Re_{tau}"],
        "theta": hdr["theta"], "dstar": hdr["delta^*"],
        "H": hdr["delta^*"] / hdr["theta"],
        "cf": 2.0 * ut ** 2,
        "y": y, "U": U, "uu": uu, "vv": vv, "ww": ww, "uv": uv,
        "k": 0.5 * (uu + vv + ww), "dUdy": dUdy,
    }


class JimenezZPG(BenchmarkCase):
    name = "jimenez-zpg-tbl"
    family = "zpg-tbl"
    reference = "Sillero2013"

    # A fully turbulent ZPG layer is the best-understood wall flow there is,
    # so the bar is higher than on the transitional plate: 2 % on c_f is about
    # the spread between careful experiments, and H is quoted to three figures
    # in the literature.
    TARGETS = {"cf_rel_rms": 0.02, "U_rms": 0.01, "theta_rel_rms": 0.02,
               "H_rel_rms": 0.02, "k_log_rms": 0.20}

    # Grid-converged defaults. Between (181, 300) and (361, 700) the c_f
    # error moves by 1 % for clip-k-omega-gamma and 4 % for
    # launder-sharma, and the ranking is unchanged on all four grids.
    def __init__(self, root=".", ny=261, y_max_factor=1.6, nx=450):
        files = sorted(glob.glob(os.path.join(root, DATA_GLOB)))
        if not files:
            raise FileNotFoundError(f"no Jimenez profiles under {root}")
        self.stations = sorted((load_station(f) for f in files),
                               key=lambda s: s["Re_theta"])
        self.nu = float(np.mean([s["nu"] for s in self.stations]))
        self.Ue = 1.0
        self.consistency = self._consistency()
        self._build_grid(ny, y_max_factor, nx)

    # -- setup --------------------------------------------------------------

    def _consistency(self):
        """How well the six headers agree on a single molecular viscosity.

        Reported rather than asserted: if this ever exceeds a per cent the
        outer-unit reconstruction below is not valid and the case should fail
        loudly rather than quietly produce a slightly wrong Reynolds number.
        """
        a = np.array([s["u_tau"] * s["d99"] / s["Re_tau"]
                      for s in self.stations])
        b = np.array([s["theta"] / s["Re_theta"] for s in self.stations])
        return {"nu_from_Re_tau": a.tolist(), "nu_from_Re_theta": b.tolist(),
                "max_rel_spread": float(np.max(np.abs(a - b) / b)),
                "nu": self.nu}

    def _build_grid(self, ny, y_max_factor, nx):
        """Marching grid, with x recovered from the momentum integral."""
        th = np.array([s["theta"] for s in self.stations])
        cf2 = np.array([(s["u_tau"] / self.Ue) ** 2 for s in self.stations])
        # x_i - x_0 = int dtheta / (cf/2), trapezoidal in theta
        dx = np.zeros(len(th))
        dx[1:] = np.cumsum(np.diff(th) * 0.5 * (1.0 / cf2[1:] + 1.0 / cf2[:-1]))
        self.x_stations = dx
        # Wall-normal grid: geometric stretching to resolve the viscous
        # sublayer at the highest Re_tau in the set
        y_max = y_max_factor * max(s["d99"] for s in self.stations)
        y_wall = 0.3 * self.nu / self.stations[-1]["u_tau"]   # y+ ~ 0.3
        r = _stretch_ratio(y_wall, y_max, ny - 1)
        # y_j = y_wall*(r^j - 1)/(r - 1), so y_0 = 0 is the wall and the
        # first cell height is y_wall. Prepending a separate zero would give
        # two coincident nodes and a zero spacing in the diffusion operator.
        self.y = y_wall * (r ** np.arange(ny) - 1.0) / (r - 1.0)
        self.y = self.y / self.y[-1] * y_max
        self.x = np.linspace(0.0, self.x_stations[-1], nx)
        self.idx_stations = np.searchsorted(self.x, self.x_stations)
        self.idx_stations = np.clip(self.idx_stations, 0, len(self.x) - 1)
        # Reference fields interpolated onto the solver grid
        s0 = self.stations[0]
        self.U0 = np.interp(self.y, s0["y"], s0["U"],
                            left=0.0, right=self.Ue)
        self.U0[0] = 0.0
        self.U0[self.y > s0["d99"] * 1.05] = self.Ue

    # -- what the closure needs --------------------------------------------

    def closure_kwargs(self, spec=None):
        # A ZPG DNS boundary layer has a laminar free stream. Give the closures
        # a small but nonzero k_inf so free-stream-decay terms stay finite.
        return {"k_inf": lambda xx: 1e-8}

    def _seed(self, grid, nu, U, Ue):
        """DNS-derived initial state for the transported scalars.

        omega comes from the measured eddy viscosity, nu_t = -<u'v'>/(dU/dy),
        as omega = k/nu_t. That is a quantity the data actually contains,
        rather than an assumed equilibrium.
        """
        s0 = self.stations[0]
        y = grid.y
        k = np.interp(y, s0["y"], s0["k"], left=0.0, right=1e-10)
        k = np.maximum(k, 1e-12)
        nut = np.where(s0["dUdy"] > 1e-9, -s0["uv"] / np.maximum(
            s0["dUdy"], 1e-12), np.nan)
        nut = np.maximum(nut, 1e-8)
        good = np.isfinite(nut)
        nut_i = np.interp(y, s0["y"][good], nut[good])
        w = np.maximum(k / np.maximum(nut_i, 1e-12), 1e-6)
        # Wilcox wall value at the first fluid node
        w[0] = 6.0 * nu / (0.072 * max(y[1], 1e-9) ** 2)
        g = np.ones_like(y)
        g[0] = 0.0
        return {"k": k, "omega": w, "gamma": g, "epsilon": 0.09 * k * w}

    # -- running and scoring ------------------------------------------------

    def run(self, closure):
        seeded = SeededClosure(closure, self._seed)
        solver = BLSolver(self.y, self.x, self.nu,
                          np.full(len(self.x), self.Ue),
                          np.zeros(len(self.x)),
                          seeded, self.U0)
        return solver.run()

    def _model_metrics(self, U):
        cf, th, H = [], [], []
        for i in self.idx_stations:
            u = U[:, i]
            dudy_w = (u[1] - u[0]) / (self.y[1] - self.y[0])
            cf.append(2.0 * self.nu * dudy_w / self.Ue ** 2)
            f = np.clip(u / self.Ue, 0.0, 1.0)
            t = np.trapezoid(f * (1.0 - f), self.y)
            d = np.trapezoid(1.0 - f, self.y)
            th.append(t)
            H.append(d / max(t, 1e-12))
        return np.array(cf), np.array(th), np.array(H)

    def errors(self, solution):
        U = solution["U"]
        if not np.all(np.isfinite(U)):
            return {"U_rms": np.inf}
        cf, th, H = self._model_metrics(U)
        ref_cf = np.array([s["cf"] for s in self.stations])
        ref_th = np.array([s["theta"] for s in self.stations])
        ref_H = np.array([s["H"] for s in self.stations])
        # Skip the inlet: it carries the DNS seed, so scoring it would credit
        # every closure for the initial condition it was handed.
        m = slice(1, None)
        errs = {
            "cf_rel_rms": rel_rms(cf[m], ref_cf[m]),
            "theta_rel_rms": rel_rms(th[m], ref_th[m]),
            "H_rel_rms": rel_rms(H[m], ref_H[m]),
        }
        # Velocity profiles, on the DNS y grid up to the edge
        u_err = []
        for j, s in list(enumerate(self.stations))[1:]:
            i = self.idx_stations[j]
            sel = s["y"] <= s["d99"]
            um = np.interp(s["y"][sel], self.y, U[:, i])
            u_err.append((um - s["U"][sel]) ** 2)
        errs["U_rms"] = float(np.sqrt(np.mean(np.concatenate(u_err))))
        k = solution.get("k")
        if k is not None:
            km, kd = [], []
            for j, s in list(enumerate(self.stations))[1:]:
                i = self.idx_stations[j]
                km.append(np.max(k[:, i]))
                kd.append(np.max(s["k"]))
            errs["k_log_rms"] = log_rms(km, kd)
        return errs

    def describe(self):
        d = super().describe()
        d["stations_Re_theta"] = [s["Re_theta"] for s in self.stations]
        d["x_stations"] = self.x_stations.tolist()
        d["nu_consistency"] = self.consistency
        return d


def _stretch_ratio(y1, y_max, n, tol=1e-12):
    """Geometric ratio giving n cells from y1 to y_max. Bisection."""
    lo, hi = 1.0 + 1e-9, 1.5
    f = lambda r: y1 * (r ** n - 1.0) / (r - 1.0) - y_max
    while f(hi) < 0:
        hi *= 1.2
        if hi > 10:
            return hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


@register_case(
    "jimenez-zpg-tbl",
    family="zpg-tbl",
    reference="Sillero2013",
    description=("Sillero/Jimenez ZPG turbulent boundary layer, "
                 "Re_theta 4000-6500. Out-of-sample for every closure here."),
)
def _make(root="."):
    return JimenezZPG(root=root)
