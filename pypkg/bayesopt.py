"""A small Bayesian optimizer for expensive, sometimes-diverging objectives.

Gaussian-process surrogate with an anisotropic squared-exponential kernel,
expected improvement as the acquisition, and a feasibility rule for runs
that diverge: an infeasible point is recorded at the worst finite value seen
so far plus a margin, so the surrogate learns to avoid the region without a
second model. That is the standard cheap treatment and it is enough here,
where a diverged solve is a legitimate answer ("this coefficient set is
unstable") rather than a failure to be hidden.

Kept dependency-free on purpose: numpy and scipy only, so it lives in the
compute environment without a torch stack, and the whole method is on one
page for the reader who wants to know what the search actually did.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf


def _kernel(X1, X2, ls, var):
    d = (X1[:, None, :] - X2[None, :, :]) / ls
    return var * np.exp(-0.5 * np.sum(d * d, axis=-1))


class GP:
    """Zero-mean GP on standardized targets with a fitted length scale."""

    def __init__(self, X, y, noise=1e-6):
        self.X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mu, self.sd = float(y.mean()), float(y.std() + 1e-12)
        self.y = (y - self.mu) / self.sd
        self.noise = noise
        self.ls = np.full(self.X.shape[1], 0.3)
        self.var = 1.0
        self._fit()

    def _nll(self, theta):
        ls = np.exp(theta[:-1])
        var = np.exp(theta[-1])
        K = _kernel(self.X, self.X, ls, var) + self.noise * np.eye(len(self.X))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
        a = np.linalg.solve(L.T, np.linalg.solve(L, self.y))
        return float(0.5 * self.y @ a + np.sum(np.log(np.diag(L))))

    def _fit(self):
        theta0 = np.concatenate([np.log(self.ls), [math.log(self.var)]])
        best = None
        for start in (theta0, theta0 - 1.0, theta0 + 1.0):
            r = minimize(self._nll, start, method="L-BFGS-B",
                         bounds=[(-4.0, 2.0)] * (len(theta0) - 1) + [(-4, 4)])
            if best is None or r.fun < best.fun:
                best = r
        self.ls = np.exp(best.x[:-1])
        self.var = float(np.exp(best.x[-1]))
        K = _kernel(self.X, self.X, self.ls, self.var) + self.noise * np.eye(
            len(self.X))
        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.y))

    def predict(self, Xs):
        Xs = np.atleast_2d(Xs)
        Ks = _kernel(Xs, self.X, self.ls, self.var)
        mean = Ks @ self.alpha
        v = np.linalg.solve(self.L, Ks.T)
        var = np.maximum(self.var - np.sum(v * v, axis=0), 1e-12)
        return mean * self.sd + self.mu, np.sqrt(var) * self.sd


def expected_improvement(mean, sd, best):
    z = (best - mean) / sd
    cdf = 0.5 * (1.0 + erf(z / math.sqrt(2.0)))
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return (best - mean) * cdf + sd * pdf


def bayes_opt(objective, bounds, n_init=12, n_iter=60, seed=0,
              infeasible_margin=0.5, callback=None, x0=None):
    """Minimize ``objective`` over a box, returning the full history.

    ``objective(x) -> float`` may return ``inf`` or ``nan`` for a diverged
    evaluation. ``bounds`` is a list of (lo, hi). Any points in ``x0`` are
    evaluated first -- a known-feasible start such as the origin belongs
    there, since the surrogate can only steer away from divergence once it
    has seen something that did not diverge -- then ``n_init`` uniform
    random points; each later point maximizes expected improvement over
    2000 random candidates refined by L-BFGS-B from the best few.
    """
    rng = np.random.default_rng(seed)
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    dim = len(bounds)
    X, raw = [], []

    def evaluate(x):
        val = float(objective(x))
        X.append(np.asarray(x, dtype=float))
        raw.append(val)
        if callback is not None:
            callback(len(raw), x, val)

    for x in x0 or []:
        evaluate(np.asarray(x, dtype=float))
    for _ in range(n_init):
        evaluate(lo + rng.random(dim) * (hi - lo))
    for _ in range(n_iter):
        y = np.array(raw)
        finite = np.isfinite(y)
        if finite.any():
            worst = y[finite].max()
            spread = max(worst - y[finite].min(), 1e-6)
            y = np.where(finite, y, worst + infeasible_margin * spread)
        else:
            evaluate(lo + rng.random(dim) * (hi - lo))
            continue
        Xn = (np.array(X) - lo) / (hi - lo)
        gp = GP(Xn, y)
        best = y.min()
        cand = rng.random((2000, dim))
        m, s = gp.predict(cand)
        ei = expected_improvement(m, s, best)
        starts = cand[np.argsort(-ei)[:5]]

        def neg_ei(z):
            m1, s1 = gp.predict(z[None, :])
            return -float(expected_improvement(m1, s1, best)[0])

        xbest, fbest = starts[0], neg_ei(starts[0])
        for z0 in starts:
            r = minimize(neg_ei, z0, method="L-BFGS-B",
                         bounds=[(0.0, 1.0)] * dim)
            if r.fun < fbest:
                xbest, fbest = r.x, r.fun
        evaluate(lo + xbest * (hi - lo))
    y = np.array(raw)
    finite = np.isfinite(y)
    i_best = int(np.argmin(np.where(finite, y, np.inf)))
    return {
        "X": np.array(X),
        "y": y,
        "best_x": X[i_best],
        "best_y": float(y[i_best]),
        "n_infeasible": int((~finite).sum()),
    }
