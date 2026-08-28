"""Turbulent boundary layer on the suction side of a NACA 4412 wing section.

Well-resolved LES from KTH \\citep{Vinuesa2018}: 50 stations from x/c = 0.153 to
0.983 at Re_c = 100k, 200k, 400k and 1M, each carrying U, V, the edge velocity,
u_tau, c_f, delta99, theta and the Reynolds stresses.

Why this case
-------------
It is the only case in the suite that is both **external** -- flow over an
aerofoil, which is what these models are actually deployed on -- and under a
severe adverse pressure gradient. The Clauser parameter reaches **beta = 112**
at Re_c = 400k and **1720** at Re_c = 100k, against order 1-4 in the standard
flat-plate APG databases.

Most importantly, **c_f falls to 1.8e-4 without ever going negative**: the layer
is on the verge of separating and never does. That is what makes it usable in
the fast tier at all. The parabolic boundary-layer equations carry the
Goldstein singularity at separation, so a marching solver cannot pass a station
where c_f = 0 -- genuinely separated flow is Tier 2 by mathematics, not by
preference. A near-separation case that stays attached is the hardest thing the
fast screen can be asked, and near-separation is what decides whether a model
calls stall correctly.

The outer boundary, and why it is Ue rather than the measured profile
--------------------------------------------------------------------
The outer flow here is genuinely not uniform: away from a curved surface the
potential-flow speed varies, so the measured U keeps changing above the layer
-- falling by 17 % from the boundary-layer edge to the top of the LES domain
near the leading edge, and *rising* near the trailing edge.

Imposing that measured profile at a fixed height was tried and rejected. Across
domain heights from 1.1 to 4.7 delta99, the resulting streamwise pressure
gradient differs from the true edge gradient by 18-38 % RMS, and the ratio of
imposed to true edge velocity drifts from 0.83 to 1.04 along the surface. The
case would have been measuring the choice of domain height as much as the
closure.

So this case does what a boundary-layer method is supposed to do: it imposes
the LES's own edge velocity Ue(s) at the top of a domain tall enough for the
thickest station, and accepts the thin-layer idealisation that U -> Ue outside
the layer. The mismatch that introduces lives entirely *above* delta99, and
**nothing above delta99 is scored** -- velocity error is measured inside the
layer only, exactly as in the Jimenez case.

Reference fidelity: this is LES, not DNS
----------------------------------------
Every other case in the suite is direct numerical simulation. This one is
well-resolved large-eddy simulation, and the distinction is not pedantic: a
sub-grid model *is* a turbulence model, so a RANS closure scored here is being
compared against a model rather than against the equations. For a project whose
whole argument is about not mistaking a fit for a law, that weakens any claim
resting on this case alone.

Three practical consequences, in decreasing order of how much they matter here:

1. **k is not scored on this case.** Published LES statistics carry the
   resolved stresses only, so the reference k is biased low by whatever share
   the sub-grid model carries -- a systematic error, in a known direction, that
   cannot be corrected from the data provided. The k metric was removed rather
   than left in with a caveat.
2. **The thin-layer approximation is probably the larger error anyway.** At
   beta = 112 and delta99/c = 0.047 the parabolic equations are being pushed
   well past where they are comfortable, and that error is almost certainly
   bigger than the LES's.
3. **c_f and the integral thicknesses are the most trustworthy quantities
   here**, being first moments that a well-resolved LES gets close to DNS.

The case is kept because it is the only *external* aerofoil flow in the suite
and the only one with a severe adverse pressure gradient, but a claim about
separation should rest on the true-DNS separation-bubble case rather than on
this one.

Streamwise coordinate is arc length along the surface, from the station
coordinates. As with the fully turbulent Jimenez inlet, the transported scalars
are seeded from the LES at the first station and that station is excluded from
scoring, so no model is judged on the initial condition it was handed.
"""

from __future__ import annotations

import os

import numpy as np
import scipy.io as sio

from ..bl_solver import BLSolver
from ..registry import register_case
from .base import BenchmarkCase, log_rms, rel_rms
from .wrappers import SeededClosure

DATA = "data/kth-wing-sections/naca4412.mat"
#: MATLAB variable name for the suction side at each chord Reynolds number.
SIDES = {100_000: "top1n", 200_000: "top2n", 400_000: "top4n", 1_000_000: "top10n"}
DEFAULT_RE_C = (400_000, 1_000_000)


def _scalar(station, field):
    return float(np.atleast_1d(getattr(station, field)).ravel()[0])


def _vector(station, field):
    return np.atleast_1d(getattr(station, field)).ravel()


def load_wing(re_c, root="."):
    """Suction-side stations at one chord Reynolds number."""
    mat = sio.loadmat(os.path.join(root, DATA), squeeze_me=True,
                      struct_as_record=False)
    raw = mat[SIDES[int(re_c)]]
    yn = _vector(raw[0], "yn")
    out = []
    for st in raw:
        out.append({
            "xa": _scalar(st, "xa"), "ya": _scalar(st, "ya"),
            "Ue": _scalar(st, "Ue"), "ut": _scalar(st, "ut"),
            "cf": _scalar(st, "Cf"), "d99": _scalar(st, "delta99"),
            "theta": _scalar(st, "theta"), "dstar": _scalar(st, "deltas"),
            "beta": _scalar(st, "beta"), "nu": _scalar(st, "nu"),
            "y": yn, "U": _vector(st, "U"), "V": _vector(st, "V"),
            "uu": _vector(st, "uu"), "vv": _vector(st, "vv"),
            "ww": _vector(st, "ww"), "uv": _vector(st, "uv"),
        })
    return out


class Naca4412Suction(BenchmarkCase):
    family = "wing-apg"
    reference = "Vinuesa2018"
    # Well-resolved LES, not DNS -- the only case in the suite that is not.
    # See the "Reference fidelity" note in the module docstring.
    fidelity = "les"

    # The forward metrics use the same bar as the other turbulent boundary
    # layer in the suite. The two aft metrics cover x/c > 0.9, where beta runs
    # from 16 to 112 and c_f falls to 1.8e-4, and they are deliberately not
    # relative-c_f measures.
    #
    # A relative error on a quantity heading to zero diverges by construction:
    # near separation a perfectly respectable absolute c_f error reads as
    # several hundred per cent, and on that metric the laminar closure
    # "outscores" a real turbulence model. So the aft skin friction is scored
    # as an absolute error against a fixed scale -- the mean c_f over the
    # forward stations -- which stays well conditioned as c_f collapses.
    #
    # H is scored there too because the shape factor is the classical
    # separation indicator and is bounded: it climbs 1.67 -> 2.77 along this
    # surface in the LES, and a model that misses the approach to separation
    # misses it in H.
    TARGETS = {"cf_rel_rms": 0.02, "U_rms": 0.01, "theta_rel_rms": 0.05,
               "H_rel_rms": 0.02, "cf_aft_abs": 0.10, "H_aft_rel_rms": 0.05}

    #: Stations at or beyond this chord fraction are the near-separation tail.
    AFT_XC = 0.90

    def __init__(self, re_c=400_000, root=".", ny=241, nx=700,
                 y_max_factor=2.6):
        self.re_c = int(re_c)
        self.name = f"naca4412-suction-rec-{self.re_c}"
        self.stations = load_wing(re_c, root=root)
        self.nu = self.stations[0]["nu"]
        xa = np.array([s["xa"] for s in self.stations])
        ya = np.array([s["ya"] for s in self.stations])
        self.xc = xa
        self.s_stations = np.concatenate(
            ([0.0], np.cumsum(np.hypot(np.diff(xa), np.diff(ya)))))
        d99 = np.array([s["d99"] for s in self.stations])
        y_max = y_max_factor * d99.max()
        y1 = 0.3 * self.nu / self.stations[-1]["ut"]      # y+ ~ 0.3
        self.y = _stretched(y1, y_max, ny)
        self.s = np.linspace(self.s_stations[0], self.s_stations[-1], nx)
        self.idx = np.clip(np.searchsorted(self.s, self.s_stations),
                           0, nx - 1)
        Ue_st = np.array([s["Ue"] for s in self.stations])
        self.Ue = np.interp(self.s, self.s_stations, Ue_st)
        self.dUeds = np.gradient(self.Ue, self.s)
        s0 = self.stations[0]
        self.U0 = np.interp(self.y, s0["y"], s0["U"])
        self.V0 = np.interp(self.y, s0["y"], s0["V"])
        # Above the layer the thin-layer idealisation applies: relax to Ue.
        self.U0[self.y > 1.5 * s0["d99"]] = s0["Ue"]

    def closure_kwargs(self, spec=None):
        # The flow above an aerofoil in a clean tunnel has no measurable
        # free-stream turbulence, and nothing here decays a free stream.
        return {"k_inf": lambda xx: 1e-9, "freestream_decay": False}

    def _seed(self, grid, nu, U, Ue):
        s0 = self.stations[0]
        y = grid.y
        k_st = 0.5 * (s0["uu"] + s0["vv"] + s0["ww"])
        k = np.maximum(np.interp(y, s0["y"], k_st), 1e-14)
        dUdy = np.gradient(s0["U"], s0["y"])
        nut = -s0["uv"] / np.where(np.abs(dUdy) > 1e-9, dUdy, np.nan)
        good = np.isfinite(nut) & (nut > 0)
        nut_i = np.maximum(np.interp(y, s0["y"][good], nut[good]), 1e-10)
        w = np.maximum(k / nut_i, 1e-6)
        w[0] = 6.0 * nu / (0.072 * max(y[1], 1e-9) ** 2)
        g = np.ones_like(y)
        g[0] = 0.0
        return {"k": k, "omega": w, "gamma": g, "epsilon": 0.09 * k * w}

    def run(self, closure):
        solver = BLSolver(self.y, self.s, self.nu, self.Ue, self.dUeds,
                          SeededClosure(closure, self._seed), self.U0, self.V0)
        return solver.run()

    def _metrics(self, U):
        """c_f, theta and H at each station, on the solver grid."""
        cf, th, H = [], [], []
        for j, i in enumerate(self.idx):
            u = U[:, i]
            Ue = self.stations[j]["Ue"]
            dudy_w = (u[1] - u[0]) / (self.y[1] - self.y[0])
            cf.append(2.0 * self.nu * dudy_w / Ue ** 2)
            f = np.clip(u / Ue, 0.0, 1.0)
            t = np.trapezoid(f * (1.0 - f), self.y)
            ds = np.trapezoid(1.0 - f, self.y)
            th.append(t)
            H.append(ds / max(t, 1e-12))
        return np.array(cf), np.array(th), np.array(H)

    def _dns_metrics(self):
        """The same integrals on the LES profiles, so like is compared to like.

        The authors' own theta and delta* are in the file, but they were
        computed with a different edge treatment on a different grid. Running
        both through the same integral removes that as a source of apparent
        model error.
        """
        cf, th, H = [], [], []
        for s in self.stations:
            u = np.interp(self.y, s["y"], s["U"])
            u[self.y > 1.5 * s["d99"]] = s["Ue"]
            f = np.clip(u / s["Ue"], 0.0, 1.0)
            t = np.trapezoid(f * (1.0 - f), self.y)
            ds = np.trapezoid(1.0 - f, self.y)
            cf.append(s["cf"])
            th.append(t)
            H.append(ds / max(t, 1e-12))
        return np.array(cf), np.array(th), np.array(H)

    def errors(self, solution):
        U = solution["U"]
        if not np.all(np.isfinite(U)):
            return {"U_rms": np.inf}
        cf, th, H = self._metrics(U)
        cfd, thd, Hd = self._dns_metrics()
        # Skip the seeded inlet station.
        fore = np.zeros(len(self.stations), bool)
        fore[1:] = self.xc[1:] < self.AFT_XC
        aft = self.xc >= self.AFT_XC
        cf_scale = float(np.mean(cfd[fore]))
        errs = {
            "cf_rel_rms": rel_rms(cf[fore], cfd[fore]),
            "cf_aft_abs": float(np.sqrt(np.mean(
                (cf[aft] - cfd[aft]) ** 2)) / cf_scale),
            "theta_rel_rms": rel_rms(th[fore], thd[fore]),
            "H_rel_rms": rel_rms(H[fore], Hd[fore]),
            "H_aft_rel_rms": rel_rms(H[aft], Hd[aft]),
        }
        # Velocity inside the layer only: above delta99 the case imposes the
        # thin-layer idealisation, which is the case's assumption and not the
        # model's to answer for.
        u_err = []
        for j in np.flatnonzero(fore | aft):
            s = self.stations[j]
            sel = s["y"] <= s["d99"]
            um = np.interp(s["y"][sel], self.y, U[:, self.idx[j]])
            u_err.append((um - s["U"][sel]) ** 2)
        errs["U_rms"] = float(np.sqrt(np.mean(np.concatenate(u_err))))
        # k is deliberately NOT scored on this case. The reference is LES, and
        # published LES statistics carry the resolved stresses only, so the
        # reference k is biased low by whatever share the sub-grid model
        # carries. Scoring a model's total k against a resolved-only k would
        # charge every closure for the sub-grid model's contribution, in a
        # known direction, with no way to correct it from the data provided.
        # The mean-flow metrics above do not have this problem.
        return errs

    def describe(self):
        d = super().describe()
        d.update({
            "Re_c": self.re_c, "nu": self.nu,
            "x_c_range": [float(self.xc[0]), float(self.xc[-1])],
            "beta_max": max(s["beta"] for s in self.stations),
            "cf_min": min(s["cf"] for s in self.stations),
            "d99_over_c_max": max(s["d99"] for s in self.stations),
            "aft_x_c": self.AFT_XC, "ny": len(self.y), "nx": len(self.s),
        })
        return d


def _stretched(y1, y_max, n):
    lo, hi = 1.0 + 1e-12, 1.5
    f = lambda r: y1 * (r ** (n - 1) - 1.0) / (r - 1.0) - y_max
    while f(hi) < 0:
        hi *= 1.2
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    r = 0.5 * (lo + hi)
    y = y1 * (r ** np.arange(n) - 1.0) / (r - 1.0)
    return y / y[-1] * y_max


for _re in DEFAULT_RE_C:
    register_case(
        f"naca4412-suction-rec-{_re}",
        (lambda r: (lambda root=".": Naca4412Suction(r, root=root)))(_re),
        family="wing-apg",
        reference="Vinuesa2018",
        description=(f"NACA 4412 suction side at Re_c = {_re:,}. External "
                     "flow under a severe adverse pressure gradient, driven "
                     "to the verge of separation without separating."),
    )
