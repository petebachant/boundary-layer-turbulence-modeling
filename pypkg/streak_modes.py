"""Free-stream-forced streak modes carried on a laminar boundary layer.

Two marchers on one staggered wall-normal grid (docs/ideas-log.md, section
7.14):

  mean   the laminar boundary-layer equations under a given edge velocity
         U_e(x), U at cell centers and V at faces
  modes  the steady linearized boundary-region equations for one spanwise
         wavenumber beta, about that mean: perturbations u, w and p at
         cell centers and v at faces, with w written as i beta w so every
         coefficient is real,

             u_x + v_y + w = 0
             U u_x + V u_y + U_x u + U_y v = nu (u_yy - beta^2 u)
             U v_x + V v_y + V_x u + V_y v + p_y = nu (v_yy - beta^2 v)
             U w_x + V w_y - beta^2 p = nu (w_yy - beta^2 w)

         parabolic in x (no u_xx, no p_x), forced by a free-stream
         streamwise vortex, v = eps sin(kappa y), w = -eps kappa
         cos(kappa y), held at the top and given as the initial field

Both march by backward Euler in x with central differences in y; the
staggering keeps p coupled to v without a pressure boundary condition.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

#: von Karman constant in the spectrum's denominator
VK_A = 1.339


def blasius(eta_max=12.0):
    """eta, f' of the Blasius profile in eta = y sqrt(U / (nu x))."""
    sol = solve_ivp(
        lambda e, f: [f[1], f[2], -0.5 * f[0] * f[2]],
        (0.0, eta_max),
        [0.0, 0.0, 0.332057336],
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
    )
    eta = np.linspace(0.0, eta_max, 2401)
    return eta, sol.sol(eta)[1]


class Grid:
    """N cells between the wall and y_top, geometrically stretched from a
    first cell dy0. Faces yf[0..N], centers yc[0..N-1]."""

    def __init__(self, n, y_top, dy0):
        lo, hi = 1.0 + 1e-9, 2.0
        for _ in range(200):
            r = 0.5 * (lo + hi)
            if dy0 * (r**n - 1) / (r - 1) > y_top:
                hi = r
            else:
                lo = r
        dy = dy0 * r ** np.arange(n)
        dy *= y_top / dy.sum()
        self.n = n
        self.yf = np.concatenate([[0.0], np.cumsum(dy)])
        self.yc = 0.5 * (self.yf[1:] + self.yf[:-1])
        self.dy = dy
        # Center-to-center spacing, with a ghost below the wall and above
        # the top mirrored about the face
        self.dc = np.diff(
            np.concatenate([[-self.yc[0]], self.yc, [2 * y_top - self.yc[-1]]])
        )


def _center_ops(g):
    """First and second derivative weights at centers from neighbors
    (j-1, j, j+1), with the ghost spacings at the ends."""
    hm, hp = g.dc[:-1], g.dc[1:]
    d1 = np.stack(
        [-hp / (hm * (hm + hp)), (hp - hm) / (hm * hp), hm / (hp * (hm + hp))]
    )
    d2 = np.stack([2 / (hm * (hm + hp)), -2 / (hm * hp), 2 / (hp * (hm + hp))])
    return d1, d2


def _face_ops(g):
    """The same at interior faces 1..N-1 from faces (f-1, f, f+1)."""
    hm, hp = g.dy[:-1], g.dy[1:]
    d1 = np.stack(
        [-hp / (hm * (hm + hp)), (hp - hm) / (hm * hp), hm / (hp * (hm + hp))]
    )
    d2 = np.stack([2 / (hm * (hm + hp)), -2 / (hm * hp), 2 / (hp * (hm + hp))])
    return d1, d2


class Mean:
    """The laminar boundary layer at stations x, U[i] at centers and V[i]
    at faces, marched from a Blasius profile at x[0], whose distance from
    the virtual leading edge is x0_virtual."""

    def __init__(self, g, x, ue, due, nu, x0_virtual, n_picard=3):
        self.g, self.x, self.ue, self.nu = g, x, ue, nu
        eta, fp = blasius()
        n, nx = g.n, len(x)
        U = np.empty((nx, n))
        V = np.zeros((nx, n + 1))
        U[0] = ue[0] * np.interp(
            g.yc * np.sqrt(ue[0] / (nu * x0_virtual)), eta, fp, right=1.0
        )
        d1, d2 = _center_ops(g)
        self.separated_at = None
        for i in range(1, nx):
            dx = x[i] - x[i - 1]
            Ui = U[i - 1].copy()
            Vi = V[i - 1].copy()
            for _ in range(n_picard):
                Uc = 0.5 * (Ui + U[i - 1])
                Vc = 0.5 * (Vi[1:] + Vi[:-1])
                # U Uc/dx (U - U_prev) + V U_y = ue due + nu U_yy
                diag = Uc / dx - nu * d2[1] + Vc * d1[1]
                lo = Vc * d1[0] - nu * d2[0]
                hi = Vc * d1[2] - nu * d2[2]
                rhs = Uc / dx * U[i - 1] + ue[i] * due[i]
                # wall: ghost = -U_0; top: ghost = 2 ue - U_{N-1}
                diag[0] -= lo[0]
                diag[-1] -= hi[-1]
                rhs[-1] -= hi[-1] * 2 * ue[i]
                A = np.zeros((3, n))
                A[0, 1:] = hi[:-1]
                A[1] = diag
                A[2, :-1] = lo[1:]
                Ui = solve_banded((1, 1), A, rhs)
                Vi = np.concatenate(
                    [[0.0], np.cumsum(-(Ui - U[i - 1]) / dx * g.dy)]
                )
            U[i], V[i] = Ui, Vi
            if self.separated_at is None and Ui[0] <= 0:
                self.separated_at = float(x[i])
        self.U, self.V = U, V

    def delta99(self, i):
        U, ue = self.U[i], self.ue[i]
        j = int(np.argmax(U >= 0.99 * ue))
        return float(self.g.yc[j]) if j > 0 else float(self.g.yf[-1])


def march_mode(mean, beta, eps, inside=None):
    """u at cell centers along every station for one mode of spanwise
    wavenumber beta and cross-plane wavenumber kappa = beta, forced by a
    free-stream vortex of amplitude eps[i]. inside, if given, multiplies
    the initial field (0 where the inflow holds no disturbance)."""
    g, x, nu = mean.g, mean.x, mean.nu
    n, nx = g.n, len(x)
    kappa = beta
    b2 = beta**2
    d1c, d2c = _center_ops(g)
    d1f, d2f = _face_ops(g)
    yf, yc = g.yf, g.yc
    # Unknown layout: u[0:n], w[n:2n], p[2n:3n], v at faces 1..n-1 [3n:4n-1]
    iu = np.arange(n)
    iw = n + iu
    ip = 2 * n + iu
    iv = 3 * n + np.arange(n - 1)
    size = 4 * n - 1
    out = np.empty((nx, n))
    # Initial field: the free-stream vortex
    v = eps[0] * np.sin(kappa * yf)
    w = -eps[0] * kappa * np.cos(kappa * yc)
    if inside is not None:
        v *= np.interp(yf, yc, inside)
        w *= inside
    v[0] = 0.0
    u = np.zeros(n)
    out[0] = u
    for i in range(1, nx):
        dx = x[i] - x[i - 1]
        U = mean.U[i]
        Vf = mean.V[i]
        Vc = 0.5 * (Vf[1:] + Vf[:-1])
        Ux = (mean.U[i] - mean.U[i - 1]) / dx
        Vx = (mean.V[i] - mean.V[i - 1]) / dx
        # U_y at centers, with the wall and top ghosts
        Ug = np.concatenate([[-U[0]], U, [2 * mean.ue[i] - U[-1]]])
        Uy = d1c[0] * Ug[:-2] + d1c[1] * Ug[1:-1] + d1c[2] * Ug[2:]
        v_top = eps[i] * np.sin(kappa * yf[-1])
        w_top = -eps[i] * kappa * np.cos(kappa * yf[-1])
        rows, cols, vals = [], [], []
        rhs = np.zeros(size)

        def add(r, c, val):
            rows.append(r)
            cols.append(c)
            vals.append(val)

        for j in range(n):
            # neighbors with ghosts: wall u = w = 0, top u = 0, w = w_top
            def stencil(row, base, coef_lo, coef_mid, coef_hi, top_val):
                add(row, base[j], coef_mid)
                if j > 0:
                    add(row, base[j - 1], coef_lo)
                else:
                    add(row, base[j], -coef_lo)
                if j < n - 1:
                    add(row, base[j + 1], coef_hi)
                else:
                    add(row, base[j], -coef_hi)
                    rhs[row] -= coef_hi * 2 * top_val

            # v at the cell's lower and upper faces, as unknown or known
            def face_v(row, f, coef):
                if f == 0:
                    return
                if f == n:
                    rhs[row] -= coef * v_top
                else:
                    add(row, iv[f - 1], coef)

            # continuity
            r = iu[j]
            add(r, iu[j], 1.0 / dx)
            rhs[r] += u[j] / dx
            face_v(r, j + 1, 1.0 / g.dy[j])
            face_v(r, j, -1.0 / g.dy[j])
            add(r, iw[j], 1.0)
            # x-momentum
            r = iw[j]
            stencil(
                r,
                iu,
                Vc[j] * d1c[0, j] - nu * d2c[0, j],
                U[j] / dx
                + Ux[j]
                + nu * b2
                + Vc[j] * d1c[1, j]
                - nu * d2c[1, j],
                Vc[j] * d1c[2, j] - nu * d2c[2, j],
                0.0,
            )
            rhs[r] += U[j] / dx * u[j]
            face_v(r, j, 0.5 * Uy[j])
            face_v(r, j + 1, 0.5 * Uy[j])
            # z-momentum
            r = ip[j]
            stencil(
                r,
                iw,
                Vc[j] * d1c[0, j] - nu * d2c[0, j],
                U[j] / dx + nu * b2 + Vc[j] * d1c[1, j] - nu * d2c[1, j],
                Vc[j] * d1c[2, j] - nu * d2c[2, j],
                w_top,
            )
            rhs[r] += U[j] / dx * w[j]
            add(r, ip[j], -b2)
        # y-momentum at interior faces
        for f in range(1, n):
            r = iv[f - 1]
            Uf = 0.5 * (U[f - 1] + U[f])
            k = f - 1  # index into face operators
            add(
                r,
                iv[f - 1],
                Uf / dx
                + Vy_face(Vf, g, f)
                + nu * b2
                + Vf[f] * d1f[1, k]
                - nu * d2f[1, k],
            )
            rhs[r] += Uf / dx * v[f]
            for df, c in ((-1, 0), (1, 2)):
                coef = Vf[f] * d1f[c, k] - nu * d2f[c, k]
                ff = f + df
                if ff == 0:
                    continue
                if ff == n:
                    rhs[r] -= coef * v_top
                else:
                    add(r, iv[ff - 1], coef)
            add(r, iu[f - 1], 0.5 * Vx[f])
            add(r, iu[f], 0.5 * Vx[f])
            h = yc[f] - yc[f - 1]
            add(r, ip[f], 1.0 / h)
            add(r, ip[f - 1], -1.0 / h)
        A = coo_matrix((vals, (rows, cols)), shape=(size, size)).tocsc()
        sol = spsolve(A, rhs)
        u = sol[iu]
        w = sol[iw]
        v = np.concatenate([[0.0], sol[iv], [v_top]])
        out[i] = u
    return out


def Vy_face(Vf, g, f):
    """dV/dy at interior face f, central over the neighboring faces."""
    return (Vf[f + 1] - Vf[f - 1]) / (g.yf[f + 1] - g.yf[f - 1])


def von_karman_transverse(beta, L):
    """The von Karman transverse one-dimensional spectrum, up to a
    constant, with integral scale L."""
    kl = VK_A * beta * L
    return L * (1 + 8.0 / 3.0 * kl**2) / (1 + kl**2) ** (11.0 / 6.0)


def streak_rms(mean, eps, L, betas, inside=None):
    """u_rms at centers along the stations, per unit receptivity: the modes
    summed in quadrature with von Karman weights, normalized to one over
    the wavenumber grid."""
    lb = np.log(betas)
    wts = von_karman_transverse(betas, L) * betas * np.gradient(lb)
    wts /= wts.sum()
    total = np.zeros((len(mean.x), mean.g.n))
    for b, wt in zip(betas, wts):
        total += wt * march_mode(mean, b, eps, inside) ** 2
    return np.sqrt(total)
