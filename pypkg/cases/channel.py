"""Fully developed turbulent channel flow, Re_tau = 180 to 5200.

Lee & Moser DNS. A different geometry from everything else in the suite: no
free stream, no boundary-layer edge, no streamwise development. That is the
point. A closure tuned on a flat-plate boundary layer has nowhere to hide here
-- the case cannot be passed by getting a transition location right, only by
getting the near-wall balance and the log layer right, and the five Reynolds
numbers give a Re-trend for free.

Formulation
-----------
With the half-width and the friction velocity as the units (delta = 1,
u_tau = 1), the molecular viscosity is 1/Re_tau and the total stress is exactly
linear:

    (nu + nu_t) dU/dy = 1 - y

so U follows from a single integration once the closure supplies nu_t. No
momentum BVP is needed, and the mean-momentum balance is satisfied exactly
rather than approximately -- which means any error in the leaderboard is an
error in the closure, not in the case's own discretization of the momentum
equation.

The transported scalars are relaxed with the same ``advance`` the marching
solver uses, stepping in a pseudo-streamwise coordinate until the state stops
changing. Because ``march_scalar`` solves ``U dphi/dx = d/dy[Gamma dphi/dy] +
S`` implicitly in y, its fixed point is exactly the fully developed balance
``d/dy[Gamma dphi/dy] + S = 0``. The converged answer does not depend on the
pseudo-step; only how many iterations it takes to get there does.

Two adaptations, applied identically to every closure and both properties of
the geometry rather than of any model (see ``pypkg.cases.wrappers``):

* the centerline is a symmetry plane, so ``SymmetryTop`` converts the
  closures' Dirichlet free-stream top boundary into zero-gradient;
* free-stream decay is switched off, because a duct has no free stream whose
  turbulence could decay.

The transported scalars are also seeded from the DNS, as in the Jimenez and
NACA 4412 cases. Without that this case silently asked a different question.
Every closure here initializes k at the free-stream level, which in a duct is
essentially zero, and k = 0 is a fixed point of every one of these models:
production is proportional to nu_t, and nu_t vanishes with k. Launder-Sharma
duly returned **exactly zero** eddy viscosity at Re_tau = 180 -- fully laminar,
Ub+ = 60 against the DNS 15.69 -- while working normally at Re_tau = 550 and
1000. That is a statement about whether a low-Re model can self-start from
nothing, which is a real weakness but not the one this case exists to measure.
Seeded from the DNS, the case asks whether the model *sustains* the correct
turbulence, and every closure is handed the same initial state by the same
rule.
"""

from __future__ import annotations

import os
import re

import numpy as np

from ..bl_solver import BLGrid
from ..registry import register_case
from .base import BenchmarkCase, log_rms
from .wrappers import SeededClosure, SymmetryTop


class NotConverged(RuntimeError):
    """The pseudo-time iteration never reached a steady state."""

DATA_DIR = "data/lee-moser-channel"
RE_TAUS = (180, 550, 1000, 2000, 5200)
#: Registered by default. The full set is available through ``make_channel``;
#: three points spanning 180 to 5200 give the Reynolds-number trend without
#: letting one geometry dominate the leaderboard by weight of numbers.
DEFAULT_RE_TAUS = (180, 1000, 5200)


def _read_dat(path):
    """A Lee & Moser profile file: header scalars plus the numeric block."""
    hdr = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("%"):
                break
            for m in re.finditer(r"([A-Za-z_^+/'\\ ]*?)\s*=\s*([0-9.eE+-]+)",
                                 line):
                hdr[m.group(1).strip()] = float(m.group(2))
    return hdr, np.loadtxt(path, comments="%")


def load_channel(re_tau, root="."):
    """Mean and fluctuation profiles for one Reynolds number, in wall units."""
    tag = f"{int(re_tau):04d}"
    base = os.path.join(root, DATA_DIR)
    hdr, mean = _read_dat(os.path.join(base, f"LM_Channel_{tag}_mean_prof.dat"))
    _, fluc = _read_dat(os.path.join(base,
                                     f"LM_Channel_{tag}_vel_fluc_prof.dat"))
    # mean:  y/delta, y+, U+, dU+/dy+, W, P
    # fluc:  y/delta, y+, uu, vv, ww, uv, uw, vw, k   (all in wall units)
    return {
        "Re_tau": hdr.get("Re_tau", float(re_tau)),
        "nu": hdr.get("nu"),
        "y": mean[:, 0], "yplus": mean[:, 1],
        "U": mean[:, 2], "dUdy": mean[:, 3],
        "uv": fluc[:, 5], "k": fluc[:, 8],
    }


def _wall_grid(re_tau, ny, y1_plus=0.2):
    """Half-channel grid, geometrically stretched from the wall to y = 1."""
    y1 = y1_plus / re_tau
    lo, hi = 1.0 + 1e-12, 1.5
    f = lambda r: y1 * (r ** (ny - 1) - 1.0) / (r - 1.0) - 1.0
    while f(hi) < 0:
        hi *= 1.2
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    r = 0.5 * (lo + hi)
    y = y1 * (r ** np.arange(ny) - 1.0) / (r - 1.0)
    return y / y[-1]


def _reichardt(yplus, kappa=0.41):
    """Smooth wall law, used only as the starting profile for the iteration."""
    return ((1.0 / kappa) * np.log1p(0.4 * yplus)
            + 7.8 * (1.0 - np.exp(-yplus / 11.0)
                     - (yplus / 11.0) * np.exp(-0.33 * yplus)))


def _integrate_U(y, nut, nu):
    """U from the exact linear total stress, (nu + nu_t) dU/dy = 1 - y."""
    dUdy = (1.0 - y) / np.maximum(nu + nut, 1e-30)
    U = np.zeros_like(y)
    U[1:] = np.cumsum(0.5 * (dUdy[1:] + dUdy[:-1]) * np.diff(y))
    return U


class ChannelCase(BenchmarkCase):
    family = "channel"
    reference = "LeeMoser2015"

    # U is in wall units here, where the log layer spans roughly 10 to 25, so
    # an RMS of 0.5 is about the width of the spread between published
    # log-law constants -- i.e. "indistinguishable from the data by eye".
    # The bulk and centerline velocities are the integral quantities a user
    # would actually quote, and both are pure predictions: u_tau is imposed,
    # so the model has to produce the right flow rate for the right forcing.
    TARGETS = {"U_plus_rms": 0.5, "Ub_rel_err": 0.02,
               "Uc_rel_err": 0.02, "k_log_rms": 0.20}

    def __init__(self, re_tau, root=".", ny=257, n_iter=4000, dx=0.25,
                 tol=1e-9):
        self.re_tau = float(re_tau)
        self.name = f"channel-retau-{int(re_tau)}"
        self.dns = load_channel(re_tau, root=root)
        self.nu = 1.0 / self.re_tau
        self.y = _wall_grid(self.re_tau, ny)
        self.n_iter, self.dx, self.tol = n_iter, dx, tol
        self.iterations = None
        self.residual = None
        # DNS interpolated onto the solver grid, half channel only
        d = self.dns
        m = d["y"] <= 1.0
        self.U_dns = np.interp(self.y, d["y"][m], d["U"][m])
        self.k_dns = np.interp(self.y, d["y"][m], d["k"][m])

    def closure_kwargs(self, spec=None):
        # A duct has no free stream, so there is no free-stream turbulence to
        # decay and no measured k_inf to supply. The tiny floor keeps
        # closures that divide by k finite.
        return {"k_inf": lambda xx: 1e-10, "freestream_decay": False}

    def _seed(self, grid, nu, U, Ue):
        """DNS initial state, in the case's units (u_tau = delta = 1).

        omega comes from the measured eddy viscosity nu_t = -<u'v'>/(dU/dy),
        which the data contains directly, rather than from an assumed
        equilibrium.
        """
        d = self.dns
        m = (d["y"] <= 1.0) & (d["yplus"] > 0.3)
        y = grid.y
        k = np.maximum(np.interp(y, d["y"][m], d["k"][m]), 1e-10)
        # In wall units nu_t/nu = -uv+ / (dU+/dy+); here nu = 1/Re_tau.
        nut_plus = -d["uv"][m] / np.maximum(d["dUdy"][m], 1e-9)
        nut = nu * np.maximum(np.interp(y, d["y"][m], nut_plus), 1e-6)
        w = np.maximum(k / nut, 1e-6)
        w[0] = 6.0 * nu / (0.072 * max(y[1], 1e-9) ** 2)
        g = np.ones_like(y)
        g[0] = 0.0
        return {"k": k, "omega": w, "gamma": g, "epsilon": 0.09 * k * w}

    def run(self, closure):
        grid = BLGrid(self.y)
        wrapped = SymmetryTop(SeededClosure(closure, self._seed))
        U = _reichardt(self.y * self.re_tau)
        wrapped.initialize(grid, self.nu, U, U[-1])
        x = 0.0
        for it in range(self.n_iter):
            nut = wrapped.eddy_viscosity(U, self.nu, grid)
            U_new = _integrate_U(self.y, nut, self.nu)
            if not np.all(np.isfinite(U_new)):
                raise ValueError("channel iteration diverged")
            res = float(np.max(np.abs(U_new - U)) / max(U_new[-1], 1e-12))
            U = U_new
            x += self.dx
            wrapped.advance(grid, U, np.zeros_like(U), self.nu, self.dx,
                            U[-1], x)
            if res < self.tol and it > 20:
                break
        self.iterations, self.residual = it + 1, res
        if res >= self.tol:
            # A number produced from a state that is still moving is not a
            # result. Refusing it here is the whole point of the harness: a
            # closure that cannot reach steady state in the simplest
            # equilibrium wall flow should say so, not be given a score that
            # depends on where the iteration happened to be stopped. The
            # dx-independence of the fixed point was verified with
            # launder-sharma, which lands on the same answer for pseudo-steps
            # spanning a factor of 800.
            raise NotConverged(
                f"residual {res:.2e} after {self.iterations} iterations "
                f"(tolerance {self.tol:.0e})")
        state = {n: np.asarray(closure.state[n]).copy()
                 for n in closure.state_names if n in closure.state}
        return {"U": U, "nut": wrapped.eddy_viscosity(U, self.nu, grid), **state}

    def _bulk(self, U):
        return float(np.trapezoid(U, self.y))

    def errors(self, solution):
        U = solution["U"]
        if not np.all(np.isfinite(U)):
            return {"U_plus_rms": np.inf}
        errs = {
            "U_plus_rms": float(np.sqrt(np.mean((U - self.U_dns) ** 2))),
            "Ub_rel_err": abs(self._bulk(U) - self._bulk(self.U_dns))
            / self._bulk(self.U_dns),
            "Uc_rel_err": abs(U[-1] - self.U_dns[-1]) / self.U_dns[-1],
        }
        k = solution.get("k")
        if k is not None:
            # Scored away from the wall, where k+ falls through several decades
            # and a log ratio is dominated by the exact position of the last
            # node rather than by anything the model got right or wrong.
            m = (self.y * self.re_tau >= 10.0) & (self.k_dns > 1e-6)
            if m.any():
                errs["k_log_rms"] = log_rms(np.maximum(k[m], 1e-12),
                                            self.k_dns[m])
        return errs

    def describe(self):
        d = super().describe()
        d.update({"Re_tau": self.re_tau, "ny": len(self.y),
                  "iterations": self.iterations, "residual": self.residual,
                  "Ub_plus_dns": self._bulk(self.U_dns),
                  "Uc_plus_dns": float(self.U_dns[-1])})
        return d


def make_channel(re_tau, root=".", **kw):
    return ChannelCase(re_tau, root=root, **kw)


for _re in DEFAULT_RE_TAUS:
    register_case(
        f"channel-retau-{_re}",
        (lambda r: (lambda root=".": ChannelCase(r, root=root)))(_re),
        family="channel",
        reference="LeeMoser2015",
        description=(f"Lee & Moser turbulent channel at Re_tau = {_re}. "
                     "Different geometry from the boundary-layer cases: no "
                     "free stream, no edge, no streamwise development."),
    )
