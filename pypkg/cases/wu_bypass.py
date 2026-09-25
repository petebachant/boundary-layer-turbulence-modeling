"""Bypass transition at five free-stream intensities (Wu et al. 2026).

The only flows in the suite that vary the free-stream turbulence, and so the
only test of whether a transition closure calibrated on the JHTDB plate, at
one intensity, knows what the intensity does. Every closure here is out of
sample on them.

Each case is a zero-pressure-gradient plate from a Blasius inlet at
Re_theta = Re_theta_0 (about 81), in units of the inlet momentum thickness
theta_0 and the free-stream speed, so x = Re_x / Re_theta_0 and
nu = 1 / Re_theta_0. The free stream is the measured one: k_inf(x) =
1.5 u_rms(x)^2 from the u_rms the dataset gives 500-800 theta_0 above the
plate, assuming isotropy there, and a closure that integrates its own
free-stream decay is started from the virtual origin of a power law fitted
to that decay.

Scored against the DNS on C_f, theta and H along the plate and on the mean
velocity profiles at the stations the dataset gives them, with the JHTDB
plate's targets, since these are the same kind of flow.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from ..bl_solver import BLSolver
from ..registry import register_case
from .base import BenchmarkCase, rel_rms

DATA = "data/wu-bypass-transition"
CASES = {"WM075": 0.75, "WM150": 1.5, "WM225": 2.25, "WM300": 3.0,
         "WM600": 6.0}


def blasius(eta_max=12.0):
    """f'(eta) of the Blasius solution, with f''(0) = 0.33206."""
    sol = solve_ivp(lambda e, f: [f[1], f[2], -0.5 * f[0] * f[2]],
                    (0.0, eta_max), [0.0, 0.0, 0.33206],
                    dense_output=True, rtol=1e-10, atol=1e-12)
    return lambda eta: np.clip(sol.sol(np.minimum(eta, eta_max))[1], 0, 1)


def _load(path):
    return np.loadtxt(path)


class WuBypassTransition(BenchmarkCase):
    family = "bypass-transition"
    reference = "Wu2026"
    TARGETS = {"cf_rel_rms": 0.02, "U_rms": 0.01, "theta_rel_rms": 0.05,
               "H_rel_rms": 0.05}

    def __init__(self, tag, root=".", ny=241, nx=700, y_max_factor=1.6):
        self.tag = tag
        self.tu_inlet_percent = CASES[tag]
        d = os.path.join(root, DATA, f"stats_{tag}")
        rth_delta = _load(f"{d}/Re_theta_versus_delta_{tag}.dat")
        self.re_theta0 = float(rth_delta[0, 0])
        self.nu = 1.0 / self.re_theta0
        self.Ue = 1.0
        cf = _load(f"{d}/Re_x_versus_cf_{tag}.dat")
        rth = _load(f"{d}/Re_x_versus_Re_theta_{tag}.dat")
        H = _load(f"{d}/Re_x_versus_shape_factor_{tag}.dat")
        urms = _load(f"{d}/Re_x_versus_urms_at_y_500_to_800_theta_0_{tag}.dat")
        to_x = lambda rex: rex / self.re_theta0  # noqa: E731
        self.x_cf, self.cf_ref = to_x(cf[:, 0]), cf[:, 1]
        self.x_th = to_x(rth[:, 0])
        self.theta_ref = rth[:, 1] / self.re_theta0
        self.x_H, self.H_ref = to_x(H[:, 0]), H[:, 1]
        self.x_fs = to_x(urms[:, 0])
        self.k_fs = 1.5 * urms[:, 1] ** 2
        x0 = float(max(self.x_cf[0], self.x_fs[0]))
        x1 = float(min(self.x_cf[-1], self.x_th[-1], self.x_fs[-1]))
        self.x = np.linspace(x0, x1, nx)
        self.x_virtual = self._fit_virtual_origin(x0)
        # Profiles of U/U_inf against y/delta at fixed Re_theta
        self.profiles = []
        for f in sorted(glob.glob(f"{d}/y_over_delta_versus_U_at_Re_theta_*"
                                  f"_{tag}.dat")):
            r = float(re.search(r"Re_theta_(\d+)", f).group(1))
            xs = float(np.interp(r, rth[:, 1], rth[:, 0])) / self.re_theta0
            if not x0 < xs <= x1:
                continue
            delta = float(np.interp(r, rth_delta[:, 0], rth_delta[:, 1]))
            p = _load(f)
            keep = p[:, 0] <= 1.0
            self.profiles.append({"x": xs, "y": p[keep, 0] * delta,
                                  "U": p[keep, 1]})
        delta_end = float(np.interp(self.theta_ref[-1] * self.re_theta0,
                                    rth_delta[:, 0], rth_delta[:, 1]))
        y_max = y_max_factor * delta_end
        y_wall = 0.05
        r_ = 1.02
        n = ny - 1
        while y_wall * (r_ ** n - 1) / (r_ - 1) < y_max:
            r_ += 0.0005
        self.y = y_wall * (r_ ** np.arange(ny) - 1) / (r_ - 1)
        self.y *= y_max / self.y[-1]
        # Blasius inlet with theta = 1: y/theta = eta / 0.664
        fp = blasius()
        self.U0 = fp(0.664 * self.y)
        self.U0[0] = 0.0

    def _fit_virtual_origin(self, x0):
        """Virtual origin of k_fs = A (x - x_v)^(-n), fitted to the measured
        decay with n free, so each closure's own exponent then acts on the
        flow's own origin, as on the JHTDB plate."""
        x, k = self.x_fs, self.k_fs
        m = (x >= x0) & (k > 0)
        x, k = x[m], k[m]

        def resid(p):
            logA, xv, n = p
            return logA - n * np.log(np.maximum(x - xv, 1e-9)) - np.log(k)

        r = least_squares(resid, [np.log(k[0]) + 1.3 * np.log(x[0]),
                                  0.0, 1.3],
                          bounds=([-50, -1e6, 0.1], [50, x[0] - 1.0, 5.0]))
        return float(r.x[1])

    def closure_kwargs(self, spec=None):
        x_fs, k_fs = self.x_fs, self.k_fs
        return {"k_inf": lambda xx: float(np.interp(xx, x_fs, k_fs)),
                "x0": float(self.x[0]), "x_virtual": self.x_virtual}

    def run(self, closure):
        solver = BLSolver(self.y, self.x, self.nu,
                          np.full(len(self.x), self.Ue),
                          np.zeros(len(self.x)), closure, self.U0)
        return solver.run()

    def _metrics(self, U):
        du = (U[1, :] - U[0, :]) / (self.y[1] - self.y[0])
        cf = 2.0 * self.nu * du / self.Ue ** 2
        f = np.clip(U / self.Ue, 0.0, 1.0)
        th = np.trapezoid(f * (1.0 - f), self.y, axis=0)
        ds = np.trapezoid(1.0 - f, self.y, axis=0)
        return cf, th, ds / np.maximum(th, 1e-12)

    def errors(self, solution):
        U = solution["U"]
        if not np.all(np.isfinite(U)):
            return {"U_rms": np.inf}
        cf, th, H = self._metrics(U)
        # The first tenth of the plate carries the inlet the closure was
        # handed, so it is not scored
        m = self.x >= self.x[0] + 0.1 * (self.x[-1] - self.x[0])
        xs = self.x[m]
        errs = {
            "cf_rel_rms": rel_rms(cf[m], np.interp(xs, self.x_cf,
                                                   self.cf_ref)),
            "theta_rel_rms": rel_rms(th[m], np.interp(xs, self.x_th,
                                                      self.theta_ref)),
            "H_rel_rms": rel_rms(H[m], np.interp(xs, self.x_H, self.H_ref)),
        }
        u_err = []
        for p in self.profiles:
            i = int(np.argmin(np.abs(self.x - p["x"])))
            um = np.interp(p["y"], self.y, U[:, i])
            u_err.append((um - p["U"]) ** 2)
        if u_err:
            errs["U_rms"] = float(np.sqrt(np.mean(np.concatenate(u_err))))
        return errs

    def describe(self):
        d = super().describe()
        d.update({"tu_inlet_percent": self.tu_inlet_percent,
                  "re_theta_0": self.re_theta0,
                  "x_range": [float(self.x[0]), float(self.x[-1])],
                  "x_virtual": self.x_virtual,
                  "n_profiles": len(self.profiles)})
        return d


def _register(tag, tu):
    name = f"wu-bypass-tu{tag[2:]}"

    @register_case(
        name,
        family="bypass-transition",
        reference="Wu2026",
        description=(f"Bypass transition on a flat plate at {tu:g} percent "
                     "inlet free-stream intensity (Wu et al. 2026). "
                     "Out of sample for every closure here."),
    )
    def _make(root="."):
        case = WuBypassTransition(tag, root=root)
        case.name = name
        return case

    return _make


for _tag, _tu in CASES.items():
    _register(_tag, _tu)
