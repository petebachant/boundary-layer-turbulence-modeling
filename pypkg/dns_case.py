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


def edge_velocity(U):
    """Boundary-layer edge velocity.

    NOT the value at the top of the domain. In this DNS the streamwise
    velocity peaks INSIDE the domain and falls slightly towards the upper
    boundary, so roughly a third of the nodes exceed the top-node value. Using
    the top node makes the integrand of the momentum thickness go negative and
    theta comes out negative -- which it did, for both the DNS and every model,
    silently corrupting theta, the shape factor, and any objective built on
    them.
    """
    return float(np.max(U))


def bl_metrics(y, U, Ue=None, nu=NU):
    """Skin friction, momentum thickness and shape factor for one profile."""
    ue = edge_velocity(U) if Ue is None else Ue
    dudy_w = (U[1] - U[0]) / (y[1] - y[0])
    cf = 2.0 * nu * dudy_w / ue ** 2
    # Integrate only up to the edge; above it the integrand is meaningless
    f = np.clip(U / ue, 0.0, 1.0)
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
            # Edge velocity is taken from the profile itself, so model and DNS
            # are each measured against their own edge rather than a shared
            # top-of-domain value
            cf[i], th[i], H[i] = bl_metrics(self.y, U[:, i], None, self.nu)
        return cf, th, H

    def dns_metrics(self):
        return self.metrics(self.U_dns)

    # Targets defining "matches the DNS by inspection". Each error term is
    # divided by its target, so a total of 1 per term means that quantity has
    # arrived. Skin friction is weighted equally with velocity because it is
    # the engineering quantity - it is the drag - and because the two can
    # diverge badly: a 2 percent velocity error can hide a 30 percent c_f
    # error, since c_f depends on the wall gradient rather than the profile.
    # k_log_rms is the RMS of log(k_model/k_dns) over the plate, measured on
    # the peak of each profile. It is here because without it the objective is
    # blind to the turbulence energy: a model can carry 5-10x too little k
    # through the pre-transitional region, lose the streak reservoir the
    # closure is built on, and still score well on c_f by transitioning early
    # enough to cancel the error. That is exactly what happened.
    TARGETS = {"cf_rel_rms": 0.02, "U_rms": 0.01,
               "theta_rel_rms": 0.05, "freestream_rel_rms": 0.05,
               "k_log_rms": 0.20}

    def k_peak_dns(self):
        """Peak fluctuation energy in each DNS profile, on the marching grid."""
        if getattr(self, "_kpk", None) is None:
            self._kpk = np.max(self.k_dns, axis=0)
        return self._kpk

    def score(self, U, x_lo=60.0, x_hi=990.0, k=None):
        """Combined error in cf and in the velocity field over the plate."""
        cf, th, H = self.metrics(U)
        cfd, thd, Hd = self.dns_metrics()
        m = (self.x >= x_lo) & (self.x <= x_hi)
        cf_err = np.sqrt(np.mean(((cf[m] - cfd[m]) / cfd[m]) ** 2))
        u_err = np.sqrt(np.mean((U[:, m] - self.U_dns[:, m]) ** 2))
        th_err = np.sqrt(np.mean(((th[m] - thd[m]) / thd[m]) ** 2))
        # Worst single station, so a model cannot pass by being good on
        # average while missing the transition region entirely
        cf_max = np.max(np.abs((cf[m] - cfd[m]) / cfd[m]))
        out = {"cf_rel_rms": cf_err, "U_rms": u_err, "theta_rel_rms": th_err,
               "cf_rel_max": cf_max}
        out["total"] = (cf_err / self.TARGETS["cf_rel_rms"]
                        + u_err / self.TARGETS["U_rms"]
                        + th_err / self.TARGETS["theta_rel_rms"])
        if k is not None:
            kd = self.k_peak_dns()[m]
            km = np.max(k[:, m], axis=0)
            lk = np.log(np.maximum(km, 1e-16) / np.maximum(kd, 1e-16))
            k_err = float(np.sqrt(np.mean(lk ** 2)))
            # Pre-transitional energy is reported separately because it is the
            # part the closure gets structurally wrong, and an average over the
            # whole plate hides it behind the turbulent region.
            pre = self.x[m] <= 205.0
            out["k_log_rms"] = k_err
            out["k_log_rms_pre"] = float(
                np.sqrt(np.mean(lk[pre] ** 2))) if pre.any() else 0.0
            out["total"] = out["total"] + k_err / self.TARGETS["k_log_rms"]
        return out
