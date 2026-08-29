"""Temporally evolving turbulent mixing layer, before roll-up.

Lusher, Sansica, Coleman & Spalart DNS (Part I), from the JAXA DNS database:
a thin tanh shear layer between streams at +/- dU/2, seeded with random
perturbations and a weak soliton-like wave, at dU Lx/nu = 250,000. The layer
becomes turbulent almost immediately, thickens as a plane mixing layer, and
from t_hat ~ 0.4 rolls up into two co-rotating vortices that merge by
t_hat ~ 3. Part II scores nine RANS models on that later phase.

Why this case is here
---------------------
It is the first flow in the suite with no wall. Every closure in this
repository was built on a flat plate, and several are built *out of* wall
distance: a mixing length kappa*y with van Driest damping on y+, and a
transition gate on the vorticity Reynolds number y^2 |dU/dy| / nu. In a free
shear layer there is no wall for y to be the distance to. The case cannot be
passed by getting a transition location right or a log layer right; it asks
whether the model sustains turbulence at the right level and spreads the
layer at the right rate when the only length scale is the layer's own.

What is scored and what is not
------------------------------
The published data are time histories of peak and integral quantities over
the whole (x, y) plane -- no profiles, no fields -- so the case scores the
histories a one-dimensional description can reproduce, over the window in
which the DNS is still one-dimensional:

* momentum thickness    delta_theta(t) / delta_theta(0)
* peak shear stress     max(-u'v') / dU^2
* peak kinetic energy   max(k) / dU^2

The vorticity thickness delta_omega = dU / max|dU/dy| is computed and
returned but deliberately NOT scored. It is the reciprocal of a pointwise
peak of a derivative, and in an eddy-viscosity model that peak need not sit
at the center of the layer: Launder-Sharma puts it at the edge, where
nu_t falls to zero, and the front there sharpens without bound as the grid
is refined -- max|dU/dy| of 39, 84 and 169 at ny = 301, 601 and 1201,
while delta_theta moves by 1 % per refinement. That is the "sharp
boundary propagating into non-turbulent fluid" Part II attributes to the
structure of two-equation models, and it is a real defect, but a metric
that a model can make grid-dependent is not a comparison against the DNS.
The integral quantities are what the reference data can support.

Window: t_hat in [T_LO, T_HI] = [0.14, 0.40]. Below T_LO the DNS is still
relaxing from its perturbed laminar start and the RANS from its seeded
state, so scoring there would measure the initial condition (Part II,
Sec. 3.1, spends a page on how much that matters). Above T_HI the wave has
begun to roll up: the DNS momentum-thickness growth rate steps from
0.015 dU -- the self-similar value; Rogers & Moser give 0.014 -- to
0.021 dU, and opposite-signed vorticity appears. Neither is something a 1-D
model can or should follow. The vortex phase is a two-dimensional unsteady
problem and belongs in Tier 2.

Units
-----
Lx = 1 and dU = 1, so t = t_hat and nu = 1/250000. The initial profile is
U = (dU/2) tanh(y/y0) with y0 = 0.005 Lx (Part II Eq. 1, Table 1), which
makes delta_omega(0) = 2 y0 = 0.01 and delta_theta(0) = y0/2 = 0.0025. The
file's peak dU/dy at t = 0 is 100.3 = 1/(2 y0), which pins the units, and
its thicknesses are normalized by their own initial values, which is why the
case scores ratios rather than absolute thicknesses.

Initial condition
-----------------
Part II's own recipe for its RANS runs (Eqs. 2-4 and Table 1, SST column),
applied to every closure by the same rule through ``SeededClosure``:

    nu_t  = C_nut y0 dU / cosh^2(Phi),    C_nut = 0.0084
    k     = 0.0305 dU^2 / cosh^4(Phi)
    omega = k / nu_t + omega_inf,         omega_inf = 2.5e-5 dU^2 / nu

with Phi = y / y0, epsilon = C_mu k omega and gamma = 1. C_nut = 0.0084
rather than the 0.02 that would give the textbook peak stress of 0.01 dU^2
is the authors' choice, made to delay the eddy-viscosity build-up so the
RANS tracked the DNS; it puts the seeded peak -u'v' at 0.0042 dU^2 against
the DNS's 0.0033 at t = 0.

Frame
-----
The closures march in x with the local U as the convection speed, which
conflates the marching coordinate with the mean velocity. A temporal problem
has U of both signs and no marching direction. So the closure is handed the
flow in a frame translating at U_c = 1000 dU (``TranslatingFrame``), in
which x / U_c = t to within dU / U_c = 0.1 %; the shear, and with it every
production term, is unchanged. The momentum equation itself,
dU/dt = d/dy[(nu + nu_t) dU/dy], is stepped in the laboratory frame.
"""

from __future__ import annotations

import os
import re

import numpy as np
from scipy.optimize import brentq

from ..bl_solver import BLGrid, march_scalar
from ..registry import register_case
from .base import BenchmarkCase, rel_rms
from .wrappers import SeededClosure, TranslatingFrame

DATA_FILE = "data/jaxa-shear-layer-vortex/Vortex_DNS_Statistics_OpenSBLI.dat"

#: Scoring window in t_hat. See the module docstring for how both ends were
#: chosen from the DNS itself.
T_LO, T_HI = 0.14, 0.40

# Part II, Table 1 (SST column) and Eqs. 2-4, in units of dU and Lx.
RE = 250_000.0
Y0 = 0.005
C_NUT = 0.0084
K_PEAK = 0.0305
OMEGA_INF_SCALE = 2.5e-5


def load_histories(root="."):
    """The Tecplot time-history file as {variable: array}."""
    path = os.path.join(root, DATA_FILE)
    with open(path) as f:
        text = f.read()
    head = text[:text.index("ZONE")]
    names = re.findall(r'"([^"]*)"', head[head.index("VARIABLES"):])
    rows = [ln.split() for ln in text.splitlines()
            if ln and ln.lstrip()[0] in "0123456789-+."]
    arr = np.array([[float(v) for v in r] for r in rows])
    if arr.shape[1] != len(names):
        raise ValueError(f"{path}: {len(names)} names, {arr.shape[1]} columns")
    return {n: arr[:, i] for i, n in enumerate(names)}


def _centered_grid(Ly, n, dy_min):
    """Sinh-stretched nodes on [0, Ly], finest at the center."""
    s = np.linspace(-1.0, 1.0, n)
    target = dy_min * (n - 1) / Ly     # = a / sinh(a)

    def f(a):
        return a / np.sinh(a) - target

    a = brentq(f, 1e-6, 50.0)
    return 0.5 * Ly * (1.0 + np.sinh(a * s) / np.sinh(a))


class TemporalMixingLayer(BenchmarkCase):
    name = "temporal-mixing-layer"
    family = "free-shear"
    reference = "Lusher2026"

    # Thickness growth rates for a plane mixing layer scatter by about 10 %
    # across careful DNS and experiments (0.014 vs 0.015 dU here), so 5 % on
    # the momentum-thickness ratio is "on the DNS curve by eye". The peak stress and
    # kinetic energy are single-point maxima of an ensemble of five
    # realizations and the textbook value 0.01 dU^2 sits 30 % below this
    # DNS's 0.013, so 10 % is the resolution the reference data support.
    TARGETS = {"dtheta_rel_rms": 0.05,
               "uv_peak_rel_rms": 0.10, "k_peak_rel_rms": 0.10}

    def __init__(self, root=".", Ly=0.6, ny=301, dy_min=5e-4, dt=1e-3,
                 Uc=1000.0, t_end=T_HI):
        self.dns = load_histories(root)
        self.nu = 1.0 / RE
        self.Ly, self.Uc, self.dt = Ly, Uc, dt
        self.t_end = t_end
        self.y = _centered_grid(Ly, ny, dy_min)
        self.eta = (self.y - 0.5 * Ly) / Y0
        # DNS samples inside the scoring window
        t = self.dns["t_hat"]
        self._m = (t >= T_LO - 1e-9) & (t <= T_HI + 1e-9)

    # -- what the closure needs from the flow ------------------------------

    def closure_kwargs(self, spec=None):
        # Both streams are quiescent: no free-stream turbulence to decay and
        # no measured k_inf to supply. The floor keeps closures that divide
        # by k finite, as in the channel case.
        return {"k_inf": lambda xx: 1e-10, "freestream_decay": False}

    def _seed(self, grid, nu, U, Ue):
        """Part II's initial condition for the transported scalars."""
        sech2 = 1.0 / np.cosh(self.eta) ** 2
        nut = np.maximum(C_NUT * Y0 * sech2, 1e-12)
        k = np.maximum(K_PEAK * sech2 ** 2, 1e-10)
        w = k / nut + OMEGA_INF_SCALE / nu
        return {"k": k, "omega": w, "nut": nut, "gamma": np.ones_like(k),
                "epsilon": 0.09 * k * w}

    # -- running and scoring -----------------------------------------------

    def _thicknesses(self, U):
        dUdy = np.gradient(U, self.y)
        d_omega = 1.0 / max(float(np.max(np.abs(dUdy))), 1e-30)
        d_theta = float(np.trapezoid(0.25 - U ** 2, self.y))
        return d_theta, d_omega, dUdy

    def run(self, closure):
        grid = BLGrid(self.y)
        wrapped = TranslatingFrame(SeededClosure(closure, self._seed),
                                   self.Uc)
        U = 0.5 * np.tanh(self.eta)
        wrapped.initialize(grid, self.nu, U, U[-1])
        ones, zeros = np.ones_like(U), np.zeros_like(U)
        n_steps = int(round(self.t_end / self.dt))
        hist = {k: [] for k in ("t", "delta_theta", "delta_omega",
                                "uv_peak", "k_peak", "nut_peak")}

        def record(t):
            nut = wrapped.eddy_viscosity(U, self.nu, grid)
            d_theta, d_omega, dUdy = self._thicknesses(U)
            hist["t"].append(t)
            hist["delta_theta"].append(d_theta)
            hist["delta_omega"].append(d_omega)
            hist["uv_peak"].append(float(np.max(nut * np.abs(dUdy))))
            hist["nut_peak"].append(float(np.max(nut)))
            k = closure.state.get("k")
            hist["k_peak"].append(float(np.max(k)) if k is not None
                                  else np.nan)
            return nut

        nut = record(0.0)
        t = 0.0
        for _ in range(n_steps):
            U = march_scalar(grid, U, ones, zeros, self.nu + nut, zeros,
                             zeros, self.dt, wall_value=-0.5, free_value=0.5)
            if not np.all(np.isfinite(U)):
                raise ValueError("mixing-layer step diverged")
            t += self.dt
            # The closure sees x = Uc * t and marches at speed ~Uc.
            wrapped.advance(grid, U, zeros, self.nu, self.Uc * self.dt,
                            U[-1], self.Uc * t)
            nut = record(t)

        out = {k: np.asarray(v) for k, v in hist.items()}
        out["delta_theta"] /= out["delta_theta"][0]
        out["delta_omega"] /= out["delta_omega"][0]
        out["U"], out["y"], out["nut"] = U, self.y, nut
        return out

    def _dns_series(self):
        d = self.dns
        return {
            "dtheta": (d["t_hat"], d["delta_theta_0"]),
            "uv_peak": (d["t_hat"], -d["Max_uv"]),
            "k_peak": (d["t_hat"], d["Max_TKE"]),
        }

    def errors(self, solution):
        t = solution["t"]
        model = {"dtheta": solution["delta_theta"],
                 "uv_peak": solution["uv_peak"],
                 "k_peak": solution["k_peak"]}
        errs = {}
        for key, (t_dns, v_dns) in self._dns_series().items():
            v_model = model[key]
            if not np.all(np.isfinite(v_model)):
                if key == "k_peak":
                    continue          # closure carries no k; not scorable
                errs[f"{key}_rel_rms"] = np.inf
                continue
            m = self._m
            errs[f"{key}_rel_rms"] = rel_rms(
                np.interp(t_dns[m], t, v_model), v_dns[m])
        return errs

    # -- reporting ----------------------------------------------------------

    def growth_rate(self, t, d_theta):
        """d(delta_theta)/dt / dU over the scoring window, by least squares."""
        m = (t >= T_LO) & (t <= T_HI)
        p = np.polyfit(t[m], d_theta[m], 1)
        return float(p[0])

    def describe(self):
        d = super().describe()
        dns = self.dns
        d.update({
            "Re": RE, "y0": Y0, "Ly": self.Ly, "ny": len(self.y),
            "dt": self.dt, "Uc": self.Uc, "window": [T_LO, T_HI],
            "delta_theta_0": Y0 / 2, "delta_omega_0": 2 * Y0,
            "seed": {"C_nut": C_NUT, "k_peak": K_PEAK,
                     "omega_inf_scale": OMEGA_INF_SCALE},
            # Absolute growth rate in the window, from the normalized DNS
            # curve times delta_theta(0). 0.014 is the Rogers & Moser value.
            "dns_growth_rate": self.growth_rate(
                dns["t_hat"], dns["delta_theta_0"]) * Y0 / 2,
            "dns_uv_peak_plateau": float(-dns["Max_uv"][self._m].mean()),
            "dns_k_peak_plateau": float(dns["Max_TKE"][self._m].mean()),
        })
        return d


@register_case(
    "temporal-mixing-layer",
    family="free-shear",
    reference="Lusher2026",
    description=("Plane mixing layer at dU Lx/nu = 250,000 before it rolls "
                 "up, scored on thickness growth and peak stress histories. "
                 "The first case with no wall."),
)
def _make(root="."):
    return TemporalMixingLayer(root=root)
