"""Fast parabolic boundary-layer solver for screening RANS closures.

Marches the steady 2-D incompressible boundary-layer equations in x, which is
orders of magnitude cheaper than a full OpenFOAM run and lets us test many
candidate closures against the JHTDB transitional BL DNS.

    U dU/dx + V dU/dy = Ue dUe/dx + d/dy[(nu + nut) dU/dy]
    dU/dx + dV/dy = 0

Closures supply nut (and any transported scalars) via a Closure object.
"""

from __future__ import annotations

import numpy as np


def tdma(a, b, c, d):
    """Solve a tridiagonal system (Thomas algorithm).

    a: sub-diagonal (a[0] unused), b: diagonal, c: super-diagonal
    (c[-1] unused), d: rhs.
    """
    n = len(b)
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    xs = np.empty(n)
    xs[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        xs[i] = dp[i] - cp[i] * xs[i + 1]
    return xs


class BLGrid:
    """Wall-normal grid plus the finite-difference weights it implies."""

    def __init__(self, y):
        self.y = np.asarray(y, dtype=float)
        self.n = len(self.y)
        # Face positions and spacings for a conservative d/dy[G dphi/dy]
        self.yf = np.empty(self.n + 1)
        self.yf[1:-1] = 0.5 * (self.y[:-1] + self.y[1:])
        self.yf[0] = 0.0                      # wall
        self.yf[-1] = self.y[-1]
        self.dyc = np.diff(self.yf)           # cell heights
        self.dyn = np.diff(self.y)            # node spacings


def diffusion_coeffs(grid, G):
    """Conservative coefficients for d/dy[G dphi/dy] on the node grid.

    Returns (lower, upper) such that the operator at node j is
    lower[j]*phi[j-1] - (lower[j]+upper[j])*phi[j] + upper[j]*phi[j+1].
    """
    n = grid.n
    Gf = np.empty(n + 1)
    Gf[1:-1] = 0.5 * (G[:-1] + G[1:])
    Gf[0] = G[0]
    Gf[-1] = G[-1]
    lower = np.zeros(n)
    upper = np.zeros(n)
    lower[1:] = Gf[1:-1] / (grid.dyn * grid.dyc[1:])
    upper[:-1] = Gf[1:-1] / (grid.dyn * grid.dyc[:-1])
    return lower, upper


def march_scalar(grid, phi_old, U, V, Gamma, source_ex, source_im,
                 dx, wall_value, free_value, wall_flux=None):
    """Advance one parabolic scalar one station in x.

    Solves U dphi/dx + V dphi/dy = d/dy[Gamma dphi/dy] + source_ex
                                   - source_im * phi
    implicitly in y. source_im >= 0 keeps the system diagonally dominant.
    """
    n = grid.n
    lower, upper = diffusion_coeffs(grid, Gamma)
    a = np.zeros(n)
    b = np.zeros(n)
    c = np.zeros(n)
    d = np.zeros(n)

    Uc = np.maximum(U, 1e-6)          # marching requires U > 0
    # Upwind the wall-normal convection on V
    Vp = np.maximum(V, 0.0)
    Vm = np.minimum(V, 0.0)

    for j in range(1, n - 1):
        dym = grid.y[j] - grid.y[j - 1]
        dyp = grid.y[j + 1] - grid.y[j]
        a[j] = -lower[j] - Vp[j] / dym
        c[j] = -upper[j] + Vm[j] / dyp
        b[j] = (Uc[j] / dx + lower[j] + upper[j]
                + Vp[j] / dym - Vm[j] / dyp + source_im[j])
        d[j] = Uc[j] * phi_old[j] / dx + source_ex[j]

    if wall_flux is None:
        b[0] = 1.0
        c[0] = 0.0
        d[0] = wall_value
    else:  # zero-gradient / prescribed-flux wall
        b[0] = 1.0
        c[0] = -1.0
        d[0] = wall_flux * (grid.y[1] - grid.y[0])
    a[-1] = 0.0
    b[-1] = 1.0
    d[-1] = free_value
    return tdma(a, b, c, d)


class BLSolver:
    """March a boundary layer in x under a given closure."""

    def __init__(self, y, x, nu, Ue, dUedx, closure, U_init, V_init=None):
        self.grid = BLGrid(y)
        self.x = np.asarray(x, dtype=float)
        self.nu = float(nu)
        self.Ue = np.asarray(Ue, dtype=float)
        self.dUedx = np.asarray(dUedx, dtype=float)
        self.closure = closure
        self.U0 = np.asarray(U_init, dtype=float)
        self.V0 = (np.zeros_like(self.U0) if V_init is None
                   else np.asarray(V_init, dtype=float))

    def run(self, n_inner=2):
        grid = self.grid
        nx = len(self.x)
        n = grid.n
        U = np.zeros((n, nx))
        V = np.zeros((n, nx))
        nut = np.zeros((n, nx))
        U[:, 0] = self.U0
        V[:, 0] = self.V0
        self.closure.initialize(grid, self.nu, self.U0, self.Ue[0])
        state = {name: np.zeros((n, nx)) for name in self.closure.state_names}
        for name in self.closure.state_names:
            state[name][:, 0] = self.closure.state[name]
        nut[:, 0] = self.closure.eddy_viscosity(self.U0, self.nu, grid)

        for i in range(nx - 1):
            dx = self.x[i + 1] - self.x[i]
            Ui, Vi = U[:, i].copy(), V[:, i].copy()
            Un, Vn = Ui.copy(), Vi.copy()
            for _ in range(n_inner):
                nut_n = self.closure.eddy_viscosity(Un, self.nu, grid)
                Gam = self.nu + nut_n
                src = np.full(n, self.Ue[i + 1] * self.dUedx[i + 1])
                Un = march_scalar(
                    grid, Ui, Ui, Vi, Gam, src, np.zeros(n), dx,
                    wall_value=0.0, free_value=self.Ue[i + 1],
                )
                # Continuity: V from integrating -dU/dx up from the wall
                dUdx = (Un - Ui) / dx
                Vn = np.zeros(n)
                Vn[1:] = -np.cumsum(
                    0.5 * (dUdx[1:] + dUdx[:-1]) * grid.dyn
                )
            U[:, i + 1] = Un
            V[:, i + 1] = Vn
            self.closure.advance(grid, Un, Vn, self.nu, dx,
                                 self.Ue[i + 1], self.x[i + 1])
            for name in self.closure.state_names:
                state[name][:, i + 1] = self.closure.state[name]
            nut[:, i + 1] = self.closure.eddy_viscosity(Un, self.nu, grid)
        return {"U": U, "V": V, "nut": nut, **state}
