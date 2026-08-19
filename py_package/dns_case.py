"""Set up the JHTDB transitional-BL test case and score closures against it."""

from __future__ import annotations

import os

import h5py
import numpy as np

from .bl_solver import BLSolver

NU = 1.25e-3
PROFILES = "data/jhtdb-transitional-bl/time-ave-profiles.h5"


def load_dns(path=None, root="."):
    path = path or os.path.join(root, PROFILES)
    d = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            d[key] = f[key][()]
    x, y = d["x_coor"], d["y_coor"]
    U, V, P = d["um"], d["vm"], d["pm"]
    uu = d["uum"] - d["um"] ** 2
    vv = d["vvm"] - d["vm"] ** 2
    ww = d["wwm"] - d["wm"] ** 2
    uv = d["uvm"] - d["um"] * d["vm"]
    return {
        "x": x, "y": y, "U": U, "V": V, "P": P,
        "uu": uu, "vv": vv, "ww": ww, "uv": uv,
        "k": 0.5 * (uu + vv + ww), "nu": NU,
    }


def bl_metrics(y, U, Ue, nu):
    """Skin friction, momentum thickness and shape factor for one profile."""
    ue = Ue
    dudy_w = (U[1] - U[0]) / (y[1] - y[0])
    cf = 2.0 * nu * dudy_w / ue ** 2
    f = np.clip(U / ue, 0.0, 1.5)
    theta = np.trapz(f * (1.0 - f), y)
    dstar = np.trapz(1.0 - f, y)
    return cf, theta, dstar / max(theta, 1e-12)


class Case:
    """The DNS case, subsampled onto a marching grid."""

    def __init__(self, root=".", x_stride=4, x0=None, y_max=None):
        self.dns = load_dns(root=root)
        d = self.dns
        y = d["y"]
        if y_max is not None:
            y = y[y <= y_max]
        self.ny = len(y)
        # The DNS grid has no wall point; prepend y=0 so the no-slip
        # condition lands on the wall rather than in the first fluid cell.
        self.y = np.concatenate(([0.0], y))
        xs = d["x"][::x_stride]
        if x0 is not None:
            xs = xs[xs >= x0]
        self.x = xs
        self.idx = np.searchsorted(d["x"], xs)
        self.idx = np.clip(self.idx, 0, len(d["x"]) - 1)
        self.nu = d["nu"]
        # Freestream from the top of the DNS domain
        self.Ue = d["U"][-1, self.idx]
        self.dUedx = np.gradient(self.Ue, self.x)
        self.k_inf = d["k"][-1, self.idx]
        # DNS fields restricted to the marching grid, with a wall row of zeros
        def wall_pad(a):
            return np.vstack((np.zeros((1, a.shape[1])), a))

        self.U_dns = wall_pad(d["U"][: self.ny, self.idx])
        self.V_dns = wall_pad(d["V"][: self.ny, self.idx])
        self.uv_dns = wall_pad(d["uv"][: self.ny, self.idx])
        self.k_dns = wall_pad(d["k"][: self.ny, self.idx])
        self.ny += 1
        self.U0 = self.U_dns[:, 0].copy()
        self.V0 = self.V_dns[:, 0].copy()

    def kinf_fn(self):
        return lambda xx: float(np.interp(xx, self.x, self.k_inf))

    def epsinf_fn(self, L=2.0):
        eps = 0.09 * self.k_inf ** 1.5 / L
        return lambda xx: float(np.interp(xx, self.x, eps))

    def solve(self, closure, n_inner=2):
        s = BLSolver(self.y, self.x, self.nu, self.Ue, self.dUedx,
                     closure, self.U0, self.V0)
        return s.run(n_inner=n_inner)

    def metrics(self, U):
        cf = np.zeros(len(self.x))
        th = np.zeros(len(self.x))
        H = np.zeros(len(self.x))
        for i in range(len(self.x)):
            cf[i], th[i], H[i] = bl_metrics(self.y, U[:, i], self.Ue[i], self.nu)
        return cf, th, H

    def dns_metrics(self):
        return self.metrics(self.U_dns)

    def score(self, U, x_lo=60.0, x_hi=990.0):
        """Combined error in cf and in the velocity field over the plate."""
        cf, th, H = self.metrics(U)
        cfd, thd, Hd = self.dns_metrics()
        m = (self.x >= x_lo) & (self.x <= x_hi)
        cf_err = np.sqrt(np.mean(((cf[m] - cfd[m]) / cfd[m]) ** 2))
        u_err = np.sqrt(np.mean((U[:, m] - self.U_dns[:, m]) ** 2))
        th_err = np.sqrt(np.mean(((th[m] - thd[m]) / thd[m]) ** 2))
        return {"cf_rel_rms": cf_err, "U_rms": u_err, "theta_rel_rms": th_err,
                "total": cf_err + 10.0 * u_err + th_err}
