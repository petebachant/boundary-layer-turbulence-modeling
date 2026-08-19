"""Candidate RANS closures for the parabolic boundary-layer solver.

Every closure exposes the same small interface so the driver can swap them:

    state_names      -- transported scalars it carries
    initialize(...)  -- set the inlet state
    eddy_viscosity() -- nut from the current state
    advance(...)     -- march its transport equations one station in x
"""

from __future__ import annotations

import numpy as np

from .bl_solver import march_scalar


def ddy(phi, y):
    return np.gradient(phi, y)


class Closure:
    state_names: tuple = ()

    def __init__(self, **coeffs):
        self.coeffs = coeffs
        self.state = {}

    def initialize(self, grid, nu, U, Ue):
        raise NotImplementedError

    def eddy_viscosity(self, U, nu, grid):
        raise NotImplementedError

    def advance(self, grid, U, V, nu, dx, Ue, x):
        raise NotImplementedError


class Laminar(Closure):
    """No model at all -- the lower bound."""

    state_names = ()

    def initialize(self, grid, nu, U, Ue):
        self.state = {}

    def eddy_viscosity(self, U, nu, grid):
        return np.zeros(grid.n)

    def advance(self, grid, U, V, nu, dx, Ue, x):
        pass


class LaunderSharma(Closure):
    """Standard low-Reynolds-number k-epsilon. The conventional baseline."""

    state_names = ("k", "epsilon")

    def __init__(self, Cmu=0.09, C1=1.44, C2=1.92, sigmak=1.0, sigmaEps=1.3,
                 k_inf=None, eps_inf=None, **kw):
        super().__init__(**kw)
        self.Cmu, self.C1, self.C2 = Cmu, C1, C2
        self.sigmak, self.sigmaEps = sigmak, sigmaEps
        self.k_inf, self.eps_inf = k_inf, eps_inf

    def initialize(self, grid, nu, U, Ue):
        k0 = self.k_inf(grid.y[0] * 0 + 30.0) if callable(self.k_inf) else 1e-4
        y = grid.y
        # Freestream turbulence everywhere, damped to zero at the wall
        kk = np.full(grid.n, k0) * np.tanh(y / 0.5) ** 2
        kk = np.maximum(kk, 1e-12)
        ee = np.maximum(self.Cmu * kk ** 1.5 / 2.0, 1e-12)
        self.state = {"k": kk, "epsilon": ee}

    def _Ret(self, nu):
        k, e = self.state["k"], self.state["epsilon"]
        return k ** 2 / (nu * np.maximum(e, 1e-20))

    def eddy_viscosity(self, U, nu, grid):
        k, e = self.state["k"], self.state["epsilon"]
        Ret = self._Ret(nu)
        fmu = np.exp(-3.4 / (1.0 + Ret / 50.0) ** 2)
        nut = self.Cmu * fmu * k ** 2 / np.maximum(e, 1e-20)
        return np.clip(nut, 0.0, 1e4 * nu)

    def advance(self, grid, U, V, nu, dx, Ue, x):
        y = grid.y
        k, e = self.state["k"], self.state["epsilon"]
        nut = self.eddy_viscosity(U, nu, grid)
        dUdy = ddy(U, y)
        P = nut * dUdy ** 2
        Ret = self._Ret(nu)
        f2 = 1.0 - 0.3 * np.exp(-np.minimum(Ret ** 2, 50.0))
        sqk = np.sqrt(np.maximum(k, 0.0))
        D = 2.0 * nu * ddy(sqk, y) ** 2
        d2U = ddy(dUdy, y)
        E = 2.0 * nu * nut * d2U ** 2

        kinf = self.k_inf(x) if callable(self.k_inf) else 1e-6
        einf = self.eps_inf(x) if callable(self.eps_inf) else 1e-8

        k_new = march_scalar(
            grid, k, U, V, nu + nut / self.sigmak,
            P, (e + D) / np.maximum(k, 1e-12), dx,
            wall_value=0.0, free_value=kinf,
        )
        k_new = np.maximum(k_new, 1e-14)
        e_new = march_scalar(
            grid, e, U, V, nu + nut / self.sigmaEps,
            self.C1 * P * e / np.maximum(k, 1e-12) + E,
            self.C2 * f2 * e / np.maximum(k, 1e-12), dx,
            wall_value=0.0, free_value=einf,
        )
        self.state = {"k": k_new, "epsilon": np.maximum(e_new, 1e-16)}


# --------------------------------------------------------------------------
# Clipping closures
# --------------------------------------------------------------------------

def mixing_length(y, U, Ue, nu, kappa=0.41, Aplus=26.0, Cl=0.09):
    """Damped mixing length capped at a fraction of the BL thickness."""
    dudy_w = max((U[1] - U[0]) / (y[1] - y[0]), 1e-12)
    utau = np.sqrt(nu * dudy_w)
    yplus = y * utau / nu
    delta = np.interp(0.99 * Ue, U, y) if U[-1] >= 0.99 * Ue else y[-1]
    ell = kappa * y * (1.0 - np.exp(-yplus / Aplus))
    return np.minimum(ell, Cl * max(delta, 1e-6)), delta, utau


THRESHOLD_PARAMS = {
    # Vorticity (shear) Reynolds number -- the classic local transition marker
    "Rev": lambda c: c["y"] ** 2 * np.abs(c["dUdy"]) / c["nu"],
    # Streak-amplitude Reynolds number
    "Rek": lambda c: np.sqrt(np.maximum(c["k"], 0.0)) * c["y"] / c["nu"],
    # Product of streak amplitude and shear -- a "clipping drive"
    "Rks": lambda c: (np.sqrt(np.maximum(c["ks"], 0.0)) * c["y"] / c["nu"]),
    # Shear-weighted streak energy
    "Sk": lambda c: (np.maximum(c["ks"], 0.0) * c["y"] ** 2
                     * np.abs(c["dUdy"]) / c["nu"] ** 2) ** (1.0 / 3.0),
}


class ClipGamma(Closure):
    """Transport total fluctuation energy k and an activation fraction gamma.

    gamma is the fraction of k that bears Reynolds shear stress. It obeys a
    *clipping* law: nothing happens until a local shear parameter exceeds a
    critical value, then the excess drives a logistic growth that saturates
    at gamma = 1 -- the rail. nut is proportional to the ACTIVE energy only,
    so the pre-transitional boundary layer stays laminar even while k is large.
    """

    state_names = ("k", "gamma", "ks")

    def __init__(self, Cmu=0.55, CD=0.16, Cgam=1.2e-2, Lam_c=440.0,
                 param="Rev", p=1.0, sigmak=1.0, sigmag=1.0, gamma0=0.10,
                 k_inf=None, Cl=0.09, **kw):
        super().__init__(**kw)
        self.Cmu, self.CD, self.Cgam = Cmu, CD, Cgam
        self.Lam_c, self.param, self.p = Lam_c, param, p
        self.sigmak, self.sigmag = sigmak, sigmag
        self.gamma0, self.k_inf, self.Cl = gamma0, k_inf, Cl

    def initialize(self, grid, nu, U, Ue):
        y = grid.y
        k0 = self.k_inf(30.0) if callable(self.k_inf) else 1e-4
        kk = np.maximum(np.full(grid.n, k0) * np.tanh(y / 0.3) ** 2, 1e-12)
        gg = np.full(grid.n, self.gamma0)
        gg[0] = 0.0
        self.state = {"k": kk, "gamma": gg, "ks": kk * (1 - gg)}
        self._nu = nu

    def _scales(self, U, nu, grid):
        ell, delta, utau = mixing_length(grid.y, U, U[-1], nu, Cl=self.Cl)
        return ell, delta, utau

    def eddy_viscosity(self, U, nu, grid):
        k = np.maximum(self.state["k"], 0.0)
        g = np.clip(self.state["gamma"], 0.0, 1.0)
        ell, _, _ = self._scales(U, nu, grid)
        return self.Cmu * g * np.sqrt(k) * ell

    def advance(self, grid, U, V, nu, dx, Ue, x):
        y = grid.y
        k = np.maximum(self.state["k"], 1e-14)
        g = np.clip(self.state["gamma"], 0.0, 1.0)
        dUdy = ddy(U, y)
        ell, delta, utau = self._scales(U, nu, grid)
        nut = self.Cmu * g * np.sqrt(k) * ell

        P = nut * dUdy ** 2
        eps_over_k = self.CD * np.sqrt(k) / np.maximum(ell, 1e-9)

        ctx = {"y": y, "dUdy": dUdy, "nu": nu, "k": k,
               "ks": k * (1.0 - g), "U": U, "delta": delta}
        Lam = THRESHOLD_PARAMS[self.param](ctx) / self.Lam_c
        excess = np.maximum(Lam - 1.0, 0.0) ** self.p

        # Clipping source: logistic in gamma, driven by the rectified excess
        Sg = self.Cgam * np.abs(dUdy) * excess * g * (1.0 - g)

        kinf = self.k_inf(x) if callable(self.k_inf) else 1e-6
        k_new = march_scalar(
            grid, k, U, V, nu + nut / self.sigmak,
            P, eps_over_k, dx, wall_value=0.0, free_value=kinf,
        )
        k_new = np.maximum(k_new, 1e-14)
        g_new = march_scalar(
            grid, g, U, V, nu + nut / self.sigmag,
            Sg, np.zeros(grid.n), dx,
            wall_value=0.0, free_value=self.gamma0,
        )
        g_new = np.clip(g_new, 0.0, 1.0)
        self.state = {"k": k_new, "gamma": g_new,
                      "ks": k_new * (1.0 - g_new)}


class ClipTwoReservoir(Closure):
    """Two-reservoir 'clipping' closure.

    Fluctuation energy is split into a streak (inactive) reservoir ks and an
    active reservoir ka. The mean flow feeds both through a total eddy
    viscosity, so mean-to-fluctuation energy transfer is exact. The two
    reservoirs then exchange energy through a *clipping* transfer R that
    conserves ks + ka identically:

        D ks/Dt = nus S^2 - R - eps_s + diff
        D ka/Dt = nut S^2 + R - eps_a + diff

    R is zero until a local shear parameter exceeds a critical value, then
    grows with the rectified excess -- the signal-clipping analogy. Only ka
    produces significant Reynolds shear stress, so the boundary layer stays
    laminar while ks grows, then transitions when the rail is exceeded.
    """

    state_names = ("ks", "ka", "nut", "R")

    def __init__(self, Cmu=0.45, Cs=0.18, CD=0.16, CDs=2.0,
                 CR=0.06, Lam_c=440.0, param="Rev", p=1.0,
                 sigmas=1.0, sigmaa=1.0, gamma0=0.10, Cl=0.09,
                 Cs_cap=0.30, k_inf=None, **kw):
        super().__init__(**kw)
        self.Cmu, self.Cs, self.CD, self.CDs = Cmu, Cs, CD, CDs
        self.CR, self.Lam_c, self.param, self.p = CR, Lam_c, param, p
        self.sigmas, self.sigmaa = sigmas, sigmaa
        self.gamma0, self.Cl, self.Cs_cap = gamma0, Cl, Cs_cap
        self.k_inf = k_inf

    def initialize(self, grid, nu, U, Ue):
        y = grid.y
        k0 = self.k_inf(30.0) if callable(self.k_inf) else 1e-4
        prof = np.tanh(y / 0.3) ** 2
        kk = np.maximum(np.full(grid.n, k0) * prof, 1e-14)
        self.state = {"ks": kk * (1 - self.gamma0), "ka": kk * self.gamma0,
                      "nut": np.zeros(grid.n), "R": np.zeros(grid.n)}

    def _viscosities(self, U, nu, grid):
        y = grid.y
        ks = np.maximum(self.state["ks"], 0.0)
        ka = np.maximum(self.state["ka"], 0.0)
        ell, delta, utau = mixing_length(y, U, U[-1], nu, Cl=self.Cl)
        nut = self.Cmu * np.sqrt(ka) * ell
        # Streak viscosity: wall-distance scaled, capped inside the BL
        ell_s = np.minimum(y, self.Cs_cap * max(delta, 1e-6))
        # Lift-up: streaks are stirred by WALL-NORMAL motions, which live in
        # the active reservoir. Scaling on ks instead would make streak
        # production self-amplifying and run away.
        nus = self.Cs * np.sqrt(ka) * ell_s
        return nut, nus, ell, ell_s, delta

    def eddy_viscosity(self, U, nu, grid):
        nut, nus, _, _, _ = self._viscosities(U, nu, grid)
        return nut + nus

    def advance(self, grid, U, V, nu, dx, Ue, x):
        y = grid.y
        ks = np.maximum(self.state["ks"], 1e-16)
        ka = np.maximum(self.state["ka"], 1e-16)
        dUdy = ddy(U, y)
        S = np.abs(dUdy)
        nut, nus, ell, ell_s, delta = self._viscosities(U, nu, grid)

        Ps = nus * dUdy ** 2
        Pa = nut * dUdy ** 2

        # Clipping transfer: rectified excess over the rail
        ctx = {"y": y, "dUdy": dUdy, "nu": nu, "k": ks + ka, "ks": ks,
               "U": U, "delta": delta}
        Lam = THRESHOLD_PARAMS[self.param](ctx) / self.Lam_c
        excess = np.maximum(Lam - 1.0, 0.0) ** self.p
        R_over_ks = self.CR * S * excess          # implicit sink on ks
        R = R_over_ks * ks

        # Dissipation: streaks decay viscously, active energy cascades
        eps_s_over_ks = self.CDs * nu / np.maximum(y ** 2, 1e-8)
        eps_a_over_ka = self.CD * np.sqrt(ka) / np.maximum(ell, 1e-9)

        kinf = self.k_inf(x) if callable(self.k_inf) else 1e-6
        ks_new = march_scalar(
            grid, ks, U, V, nu + nut / self.sigmas,
            Ps, R_over_ks + eps_s_over_ks, dx,
            wall_value=0.0, free_value=kinf * (1 - self.gamma0),
        )
        ka_new = march_scalar(
            grid, ka, U, V, nu + nut / self.sigmaa,
            Pa + R, eps_a_over_ka, dx,
            wall_value=0.0, free_value=kinf * self.gamma0,
        )
        self.state = {"ks": np.maximum(ks_new, 1e-16),
                      "ka": np.maximum(ka_new, 1e-16),
                      "nut": nut + nus, "R": R}


class ClipKGamma(Closure):
    """k-gamma 'clipping' closure -- the main candidate.

    Two transported quantities:

      k     total fluctuation energy
      gamma the ACTIVATION fraction: how much of k bears Reynolds shear
            stress. This is the new quantity, and it obeys a clipping law.

    Eddy viscosity is gated by the activation,

        nut = Cmu * gamma * sqrt(k) * ell

    so a boundary layer can carry large k while remaining laminar in the
    mean -- exactly what the DNS shows pre-transition. k is fed instead by a
    lift-up production built on the FREESTREAM fluctuation amplitude,

        nuL = CL * sqrt(k_inf) * ell_s ,   P_L = nuL * S^2

    which is not self-amplifying, so it cannot drive a spurious transition.
    Both nut and nuL appear in the momentum equation, so mean-to-fluctuation
    energy transfer is exact.

    The activation obeys a rectified, saturating (clipped) source:

        D gamma/Dt = Cgam * S * (gamma + gseed) * (1 - gamma)
                     * max(0, Lambda - 1)^p + diffusion

    Nothing happens below the rail (Lambda < 1); above it the excess drives
    logistic growth that saturates at gamma = 1 -- the DNS rail a1 = 0.137.
    """

    state_names = ("k", "gamma", "nut", "Lam")

    def __init__(self, Cmu=0.45, CL=0.03, CD=0.16, Cgam=0.6, Lam_c=440.0,
                 param="Rev", p=1.0, sigmak=1.0, sigmag=1.0,
                 gamma_fs=0.02, gseed=0.01, Cl=0.09, Cs_cap=0.30,
                 Cnu=2.0, a1=0.0, k_inf=None, **kw):
        super().__init__(**kw)
        self.Cmu, self.CL, self.CD, self.Cgam = Cmu, CL, CD, Cgam
        self.Lam_c, self.param, self.p = Lam_c, param, p
        self.sigmak, self.sigmag = sigmak, sigmag
        self.gamma_fs, self.gseed = gamma_fs, gseed
        self.Cl, self.Cs_cap, self.Cnu = Cl, Cs_cap, Cnu
        # a1 > 0 enables a hard stress limiter: -<u'v'> <= 2*a1*gamma*k,
        # i.e. the DNS rail imposed directly on the stress rather than
        # emerging from the gamma equation alone.
        self.a1 = a1
        self.k_inf = k_inf

    def initialize(self, grid, nu, U, Ue):
        y = grid.y
        k0 = self.k_inf(30.0) if callable(self.k_inf) else 1e-4
        kk = np.maximum(np.full(grid.n, k0) * np.tanh(y / 0.3) ** 2, 1e-14)
        gg = np.full(grid.n, self.gamma_fs)
        gg[0] = 0.0
        self.state = {"k": kk, "gamma": gg, "nut": np.zeros(grid.n),
                      "Lam": np.zeros(grid.n)}
        self._kinf_now = k0

    def _visc(self, U, nu, grid):
        y = grid.y
        k = np.maximum(self.state["k"], 0.0)
        g = np.clip(self.state["gamma"], 0.0, 1.0)
        ell, delta, utau = mixing_length(y, U, U[-1], nu, Cl=self.Cl)
        nut = self.Cmu * g * np.sqrt(k) * ell
        if self.a1 > 0.0:
            S = np.abs(ddy(U, y))
            nut = np.minimum(nut, 2.0 * self.a1 * g * k / np.maximum(S, 1e-9))
        ell_s = np.minimum(y, self.Cs_cap * max(delta, 1e-6))
        nuL = self.CL * np.sqrt(max(self._kinf_now, 0.0)) * ell_s
        return nut, nuL, ell, delta

    def eddy_viscosity(self, U, nu, grid):
        nut, nuL, _, _ = self._visc(U, nu, grid)
        return nut + nuL

    def advance(self, grid, U, V, nu, dx, Ue, x):
        y = grid.y
        kinf = self.k_inf(x) if callable(self.k_inf) else 1e-6
        self._kinf_now = kinf
        k = np.maximum(self.state["k"], 1e-16)
        g = np.clip(self.state["gamma"], 0.0, 1.0)
        dUdy = ddy(U, y)
        S = np.abs(dUdy)
        nut, nuL, ell, delta = self._visc(U, nu, grid)

        P = (nut + nuL) * dUdy ** 2
        # Turbulent cascade when activated; viscous decay when not
        eps_over_k = (self.CD * g * np.sqrt(k) / np.maximum(ell, 1e-9)
                      + self.Cnu * (1.0 - g) * nu / np.maximum(y ** 2, 1e-8))

        ctx = {"y": y, "dUdy": dUdy, "nu": nu, "k": k, "ks": k * (1 - g),
               "U": U, "delta": delta}
        Lam = THRESHOLD_PARAMS[self.param](ctx) / self.Lam_c
        excess = np.maximum(Lam - 1.0, 0.0) ** self.p
        Sg = self.Cgam * S * excess * (g + self.gseed) * (1.0 - g)

        k_new = march_scalar(
            grid, k, U, V, nu + nut / self.sigmak,
            P, eps_over_k, dx, wall_value=0.0, free_value=kinf,
        )
        g_new = march_scalar(
            grid, g, U, V, nu + nut / self.sigmag,
            Sg, np.zeros(grid.n), dx,
            wall_value=0.0, free_value=self.gamma_fs,
        )
        self.state = {"k": np.maximum(k_new, 1e-16),
                      "gamma": np.clip(g_new, 0.0, 1.0),
                      "nut": nut + nuL, "Lam": Lam}


class ClipKOmegaGamma(Closure):
    """k-omega-gamma clipping closure -- the OpenFOAM-portable form.

    Same physics as ClipKGamma, but the turbulence length scale comes from a
    transported omega instead of an algebraic mixing length. That removes the
    need for a boundary-layer thickness, which is non-local and fragile in a
    general-purpose CFD code, so this is the version that ports cleanly.

        nut = gamma * k / omega            (optionally stress-limited by a1)
        nuL = CL * sqrt(k_inf) * ell_s     (lift-up, freestream-driven)

        Dk/Dt     = (nut + nuL) S^2 - betaStar * gamma * k * omega
                    - Cnu * (1 - gamma) * nu * k / y^2 + diff
        Domega/Dt = alpha * S^2 * (nut + nuL) * omega / k
                    - beta * omega^2 + diff
        Dgamma/Dt = Cgam * S * (gamma + gseed) * (1 - gamma)
                    * max(0, Rev/Lam_c - 1)^p + diff
    """

    state_names = ("k", "omega", "gamma", "nut", "Lam")

    def __init__(self, alpha=0.52, beta=0.072, betaStar=0.09, CL=0.03,
                 Cgam=0.6, Lam_c=440.0, param="Rev", p=1.0,
                 sigmak=2.0, sigmaw=2.0, sigmag=1.0, gamma_fs=0.02,
                 gseed=0.01, Cs_cap=0.30, Cnu=2.0, a1=0.0,
                 omega_fs_scale=10.0, local_liftup=False, k_inf=None, **kw):
        super().__init__(**kw)
        self.alpha, self.beta, self.betaStar = alpha, beta, betaStar
        self.CL, self.Cgam, self.Lam_c = CL, Cgam, Lam_c
        self.param, self.p = param, p
        self.sigmak, self.sigmaw, self.sigmag = sigmak, sigmaw, sigmag
        self.gamma_fs, self.gseed, self.Cs_cap = gamma_fs, gseed, Cs_cap
        self.Cnu, self.a1 = Cnu, a1
        self.omega_fs_scale = omega_fs_scale
        # local_liftup: drive the lift-up term with the LOCAL active
        # amplitude sqrt(gamma*k) instead of the freestream sqrt(k_inf).
        # Pre-transition gamma is small and gated, so this is still not
        # self-amplifying, but it needs no non-local freestream input --
        # which is what makes the model portable to a general CFD code.
        self.local_liftup = local_liftup
        self.k_inf = k_inf

    def initialize(self, grid, nu, U, Ue):
        y = grid.y
        k0 = self.k_inf(30.0) if callable(self.k_inf) else 1e-4
        self._kinf_now = k0
        kk = np.maximum(np.full(grid.n, k0) * np.tanh(y / 0.3) ** 2, 1e-14)
        w0 = self.omega_fs_scale * np.sqrt(max(k0, 1e-16))
        ww = np.full(grid.n, w0)
        # Wilcox wall value; y[0] is the wall so use the first fluid node
        ww[0] = 6.0 * nu / (self.beta * max(y[1], 1e-9) ** 2)
        gg = np.full(grid.n, self.gamma_fs)
        gg[0] = 0.0
        self.state = {"k": kk, "omega": ww, "gamma": gg,
                      "nut": np.zeros(grid.n), "Lam": np.zeros(grid.n)}

    def _visc(self, U, nu, grid):
        y = grid.y
        k = np.maximum(self.state["k"], 0.0)
        w = np.maximum(self.state["omega"], 1e-12)
        g = np.clip(self.state["gamma"], 0.0, 1.0)
        nut = g * k / w
        if self.a1 > 0.0:
            S = np.abs(ddy(U, y))
            nut = np.minimum(nut, 2.0 * self.a1 * g * k / np.maximum(S, 1e-9))
        if self.local_liftup:
            # Fully local: length scale limited by sqrt(k)/omega, amplitude
            # by the local active energy.
            ell_s = np.minimum(y, self.Cs_cap * np.sqrt(k) / w)
            amp = np.sqrt(np.maximum(g * k, 0.0))
        else:
            _, delta, _ = mixing_length(y, U, U[-1], nu)
            ell_s = np.minimum(y, self.Cs_cap * max(delta, 1e-6))
            amp = np.sqrt(max(self._kinf_now, 0.0))
        nuL = self.CL * amp * ell_s
        return np.clip(nut, 0.0, 1e5 * nu), nuL

    def eddy_viscosity(self, U, nu, grid):
        nut, nuL = self._visc(U, nu, grid)
        return nut + nuL

    def advance(self, grid, U, V, nu, dx, Ue, x):
        y = grid.y
        kinf = self.k_inf(x) if callable(self.k_inf) else 1e-6
        self._kinf_now = kinf
        k = np.maximum(self.state["k"], 1e-16)
        w = np.maximum(self.state["omega"], 1e-12)
        g = np.clip(self.state["gamma"], 0.0, 1.0)
        dUdy = ddy(U, y)
        S = np.abs(dUdy)
        nut, nuL = self._visc(U, nu, grid)
        P = (nut + nuL) * dUdy ** 2

        ctx = {"y": y, "dUdy": dUdy, "nu": nu, "k": k, "ks": k * (1 - g),
               "U": U, "delta": 1.0}
        Lam = THRESHOLD_PARAMS[self.param](ctx) / self.Lam_c
        excess = np.maximum(Lam - 1.0, 0.0) ** self.p
        Sg = self.Cgam * S * excess * (g + self.gseed) * (1.0 - g)

        w_fs = self.omega_fs_scale * np.sqrt(max(kinf, 1e-16))
        k_new = march_scalar(
            grid, k, U, V, nu + nut / self.sigmak, P,
            self.betaStar * g * w + self.Cnu * (1 - g) * nu / np.maximum(y ** 2, 1e-8),
            dx, wall_value=0.0, free_value=kinf,
        )
        w_new = march_scalar(
            grid, w, U, V, nu + nut / self.sigmaw,
            self.alpha * (nut + nuL) * dUdy ** 2 * w / np.maximum(k, 1e-14),
            self.beta * w, dx,
            wall_value=6.0 * nu / (self.beta * max(y[1], 1e-9) ** 2),
            free_value=w_fs,
        )
        g_new = march_scalar(
            grid, g, U, V, nu + nut / self.sigmag, Sg, np.zeros(grid.n), dx,
            wall_value=0.0, free_value=self.gamma_fs,
        )
        self.state = {"k": np.maximum(k_new, 1e-16),
                      "omega": np.maximum(w_new, 1e-12),
                      "gamma": np.clip(g_new, 0.0, 1.0),
                      "nut": nut + nuL, "Lam": Lam}


class GrammarKOmegaGamma(ClipKOmegaGamma):
    """k-omega-gamma closure whose activation source comes from the grammar.

    Identical to ClipKOmegaGamma except that the gamma source term is
    assembled from a py_package.grammar.Candidate, so a search can vary the
    OPERATORS and DERIVED QUANTITIES in the PDE rather than only the
    coefficients multiplying a fixed set of terms.
    """

    def __init__(self, candidate=None, gcoeffs=None, **kw):
        super().__init__(**kw)
        self.candidate = candidate
        self.gcoeffs = dict(gcoeffs or {})

    def _gamma_source(self, grid, U, nu, dUdy):
        ctx = {
            "y": grid.y, "nu": nu, "k": np.maximum(self.state["k"], 0.0),
            "gamma": np.clip(self.state["gamma"], 0.0, 1.0),
            "omega": np.maximum(self.state["omega"], 1e-12),
            "dUdy": dUdy,
        }
        src = self.candidate.source(ctx, self.gcoeffs)
        return np.nan_to_num(src, nan=0.0, posinf=0.0, neginf=0.0)

    def advance(self, grid, U, V, nu, dx, Ue, x):
        y = grid.y
        kinf = self.k_inf(x) if callable(self.k_inf) else 1e-6
        self._kinf_now = kinf
        k = np.maximum(self.state["k"], 1e-16)
        w = np.maximum(self.state["omega"], 1e-12)
        g = np.clip(self.state["gamma"], 0.0, 1.0)
        dUdy = ddy(U, y)
        nut, nuL = self._visc(U, nu, grid)
        P = (nut + nuL) * dUdy ** 2
        Sg = np.maximum(self._gamma_source(grid, U, nu, dUdy), 0.0)

        w_fs = self.omega_fs_scale * np.sqrt(max(kinf, 1e-16))
        k_new = march_scalar(
            grid, k, U, V, nu + nut / self.sigmak, P,
            self.betaStar * g * w
            + self.Cnu * (1 - g) * nu / np.maximum(y ** 2, 1e-8),
            dx, wall_value=0.0, free_value=kinf,
        )
        w_new = march_scalar(
            grid, w, U, V, nu + nut / self.sigmaw,
            self.alpha * (nut + nuL) * dUdy ** 2 * w / np.maximum(k, 1e-14),
            self.beta * w, dx,
            wall_value=6.0 * nu / (self.beta * max(y[1], 1e-9) ** 2),
            free_value=w_fs,
        )
        g_new = march_scalar(
            grid, g, U, V, nu + nut / self.sigmag, Sg, np.zeros(grid.n), dx,
            wall_value=0.0, free_value=self.gamma_fs,
        )
        self.state = {"k": np.maximum(k_new, 1e-16),
                      "omega": np.maximum(w_new, 1e-12),
                      "gamma": np.clip(g_new, 0.0, 1.0),
                      "nut": nut + nuL, "Lam": np.zeros(grid.n)}


HMAX = np.log(3.0)


class EntropyKOmegaH(Closure):
    """k-omega-H closure: an information/entropy balance for the fluctuations.

    The new transported quantity is the COMPONENT ENTROPY of the fluctuation
    energy partition,

        H = -sum_i p_i ln p_i ,   p_i = <u_i u_i> / 2k

    which is ln3 for isotropic turbulence and 0 for a purely streamwise
    (perfectly ordered) field. Measured from the DNS, H is *non-monotone*: it
    falls from 1.07 at the inlet to a minimum of 0.499 at x = 233 as the mean
    shear organises the fluctuations into streaks, then rises back to 0.995
    once they break down. Transition onset coincides with the point of
    maximum order.

    The model reproduces that shape as a competition between two terms rather
    than imposing it:

        DH/Dt = -Cord * S * (H - Hfloor)          (shear injects ORDER;
                                                   entropy-reducing, and it
                                                   costs mean-flow work)
                + Cmix * omega * (Hmax - H)       (breakdown RELAXES toward
                                                   isotropy; entropy-producing,
                                                   always >= 0)
                + diffusion

    Early on the turbulence is weak, omega is small, ordering wins and H
    falls. As k and omega build, mixing takes over and H recovers. The
    turning point emerges from the balance; it is not prescribed.

    The constitutive law comes straight from the DNS breakdown branch,
    a1 = 0.1368*(H/Hmax)^1.61 with correlation 0.996:

        nut = 2 * a1(H) * k / max(S, ...)     (stress-limited form)
    """

    state_names = ("k", "omega", "H", "nut")

    def __init__(self, alpha=0.52, beta=0.072, betaStar=0.09,
                 A1=0.1368, nH=1.61, Cord=0.6, Cmix=2.0, Hfloor=0.05,
                 CL=0.03, Cs_cap=0.30, Cnu=2.0, Cmu=0.55,
                 sigmak=2.0, sigmaw=2.0, sigmaH=1.0,
                 omega_fs_scale=10.0, stress_limited=True,
                 k_inf=None, **kw):
        super().__init__(**kw)
        self.alpha, self.beta, self.betaStar = alpha, beta, betaStar
        self.A1, self.nH = A1, nH
        self.Cord, self.Cmix, self.Hfloor = Cord, Cmix, Hfloor
        self.CL, self.Cs_cap, self.Cnu, self.Cmu = CL, Cs_cap, Cnu, Cmu
        self.sigmak, self.sigmaw, self.sigmaH = sigmak, sigmaw, sigmaH
        self.omega_fs_scale = omega_fs_scale
        self.stress_limited = stress_limited
        self.k_inf = k_inf

    def initialize(self, grid, nu, U, Ue):
        y = grid.y
        k0 = self.k_inf(30.0) if callable(self.k_inf) else 1e-4
        self._kinf_now = k0
        kk = np.maximum(np.full(grid.n, k0) * np.tanh(y / 0.3) ** 2, 1e-14)
        ww = np.full(grid.n, self.omega_fs_scale * np.sqrt(max(k0, 1e-16)))
        ww[0] = 6.0 * nu / (self.beta * max(y[1], 1e-9) ** 2)
        # Inlet freestream turbulence is nearly isotropic
        HH = np.full(grid.n, 0.98 * HMAX)  # inlet FST is nearly isotropic
        self.state = {"k": kk, "omega": ww, "H": HH, "nut": np.zeros(grid.n)}

    def _a1(self):
        Hn = np.clip(self.state["H"] / HMAX, 0.0, 1.0)
        return self.A1 * Hn ** self.nH

    def _visc(self, U, nu, grid):
        y = grid.y
        k = np.maximum(self.state["k"], 0.0)
        w = np.maximum(self.state["omega"], 1e-12)
        a1 = self._a1()
        S = np.abs(ddy(U, y))
        if self.stress_limited:
            # Entropy sets how much stress a given energy can carry
            nut = np.minimum(self.Cmu * k / w,
                             2.0 * a1 * k / np.maximum(S, 1e-9))
        else:
            nut = (a1 / self.A1) * self.Cmu * k / w
        ell_s = np.minimum(y, self.Cs_cap * np.sqrt(k) / w)
        nuL = self.CL * np.sqrt(np.maximum(k * (a1 / max(self.A1, 1e-12)), 0.0)) * ell_s
        return np.clip(nut, 0.0, 1e5 * nu), nuL

    def eddy_viscosity(self, U, nu, grid):
        nut, nuL = self._visc(U, nu, grid)
        return nut + nuL

    def advance(self, grid, U, V, nu, dx, Ue, x):
        y = grid.y
        kinf = self.k_inf(x) if callable(self.k_inf) else 1e-6
        self._kinf_now = kinf
        k = np.maximum(self.state["k"], 1e-16)
        w = np.maximum(self.state["omega"], 1e-12)
        H = np.clip(self.state["H"], 0.0, HMAX)
        dUdy = ddy(U, y)
        S = np.abs(dUdy)
        nut, nuL = self._visc(U, nu, grid)
        P = (nut + nuL) * dUdy ** 2

        w_fs = self.omega_fs_scale * np.sqrt(max(kinf, 1e-16))
        k_new = march_scalar(
            grid, k, U, V, nu + nut / self.sigmak, P,
            self.betaStar * w, dx, wall_value=0.0, free_value=kinf,
        )
        w_new = march_scalar(
            grid, w, U, V, nu + nut / self.sigmaw,
            self.alpha * (nut + nuL) * dUdy ** 2 * w / np.maximum(k, 1e-14),
            self.beta * w, dx,
            wall_value=6.0 * nu / (self.beta * max(y[1], 1e-9) ** 2),
            free_value=w_fs,
        )
        # Entropy balance: ordering by shear vs entropy-producing breakdown.
        # Written implicitly in H so both terms stay stable.
        H_new = march_scalar(
            grid, H, U, V, nu + nut / self.sigmaH,
            self.Cord * S * self.Hfloor + self.Cmix * w * HMAX,
            self.Cord * S + self.Cmix * w, dx,
            wall_value=0.0, free_value=0.98 * HMAX, wall_flux=0.0,
        )
        self.state = {"k": np.maximum(k_new, 1e-16),
                      "omega": np.maximum(w_new, 1e-12),
                      "H": np.clip(H_new, 0.0, HMAX),
                      "nut": nut + nuL}
