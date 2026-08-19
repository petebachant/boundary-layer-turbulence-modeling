"""A grammar of candidate RANS closure terms, for structural PDE search.

Coefficient tuning alone cannot change what a model is *able* to represent.
This module defines a small, dimensionally-consistent grammar so a search can
add and remove OPERATORS and DERIVED QUANTITIES, not just constants.

A candidate activation source is assembled as

    S_gamma = Cgam * rate * shape(gamma) * prod_j f_j(D_j)

where

    rate      a quantity with dimensions 1/time (what sets the pace)
    shape     how the source saturates in gamma (the clipping rail)
    D_j       dimensionless local drivers (the derived quantities)
    f_j       a response function applied to a driver (rectifier, etc.)

Everything in the grammar is built from local invariants, so every candidate
is dimensionally consistent, Galilean invariant, and has admissible wall and
freestream limits by construction. Constraining the grammar this way is worth
far more than a clever acquisition function: it removes the vast majority of
nonsense candidates before they are ever evaluated.
"""

from __future__ import annotations

import numpy as np

SMALL = 1e-30


def _d(phi, y):
    return np.gradient(phi, y)


def drivers(ctx):
    """Dimensionless local drivers available to the grammar.

    ctx carries y, nu, k, gamma, omega, dUdy and the wall-normal grid.
    Each driver is O(1) at the point where the corresponding physics matters,
    so a threshold coefficient near 1 is meaningful.
    """
    y, nu, k, g = ctx["y"], ctx["nu"], ctx["k"], ctx["gamma"]
    S = np.abs(ctx["dUdy"])
    w = ctx.get("omega")
    kk = np.maximum(k, 0.0)
    out = {}
    # Shear (vorticity) Reynolds number -- classic local transition marker
    out["Rev"] = y ** 2 * S / nu
    # Turbulence Reynolds number based on wall distance
    out["Rek"] = np.sqrt(kk) * y / nu
    # Streak (inactive) Reynolds number
    out["Reks"] = np.sqrt(kk * (1.0 - g)) * y / nu
    # Total-energy gradient, normalised -- how sharply the energy profile
    # varies over a wall distance
    out["Gk"] = np.abs(_d(kk, y)) * y / np.maximum(kk, SMALL)
    # Shear-to-turbulence timescale ratio
    if w is not None:
        out["Sw"] = S / np.maximum(w, SMALL)
        out["Ret"] = kk / np.maximum(nu * w, SMALL)
    else:
        out["Sw"] = S * y / np.maximum(np.sqrt(kk), SMALL)
        out["Ret"] = kk / np.maximum(nu * S, SMALL)
    # Production-to-dissipation proxy
    out["Pe"] = out["Sw"] ** 2
    # Component entropy of the energy partition, if the closure carries it.
    # Normalised so 1 is isotropic and 0 is a perfectly ordered (streaky)
    # field. Measured from DNS this is non-monotone through transition.
    if "H" in ctx:
        out["Hn"] = np.clip(ctx["H"] / np.log(3.0), 0.0, 1.0)
        out["Hdef"] = 1.0 - out["Hn"]          # entropy deficit = "order"
    return out


RATES = {
    "S": lambda ctx, D: np.abs(ctx["dUdy"]),
    "omega": lambda ctx, D: np.maximum(ctx.get("omega", np.abs(ctx["dUdy"])), SMALL),
    "kOverNuY2": lambda ctx, D: ctx["nu"] / np.maximum(ctx["y"] ** 2, SMALL),
    "sqrtKOverY": lambda ctx, D: (np.sqrt(np.maximum(ctx["k"], 0.0))
                                  / np.maximum(ctx["y"], SMALL)),
}

SHAPES = {
    # Logistic with a seed: self-exciting, saturates at gamma = 1 (the rail)
    "logistic_seeded": lambda g, a: (g + a) * (1.0 - g),
    # Pure logistic: needs gamma > 0 to start
    "logistic": lambda g, a: g * (1.0 - g),
    # Linear approach to the rail: no self-excitation
    "linear": lambda g, a: (1.0 - g),
    # Quadratic approach: slower near the rail
    "quadratic": lambda g, a: (1.0 - g) ** 2,
}

RESPONSES = {
    # Rectified excess -- the clip. Identically zero below the threshold.
    "rectify": lambda D, t, p: np.maximum(D / t - 1.0, 0.0) ** p,
    # Smooth (soft) clip
    "softclip": lambda D, t, p: np.log1p(np.exp(np.clip((D / t - 1.0) * p, -50, 50))) / p,
    # Saturating ramp
    "tanh": lambda D, t, p: np.tanh(np.maximum(D / t - 1.0, 0.0) * p),
    # Plain power law -- no threshold at all (a useful null hypothesis)
    "power": lambda D, t, p: (D / t) ** p,
    # Inverse: suppression rather than activation
    "inverse": lambda D, t, p: 1.0 / (1.0 + (D / t) ** p),
}


class Candidate:
    """One structural candidate for the activation source term."""

    def __init__(self, rate, shape, terms):
        # terms: list of (driver_name, response_name)
        self.rate = rate
        self.shape = shape
        self.terms = list(terms)

    # -- description -------------------------------------------------------
    def key(self):
        ts = "*".join(f"{r}({d})" for d, r in sorted(self.terms))
        return f"{self.rate}|{self.shape}|{ts}"

    def n_terms(self):
        return len(self.terms)

    def coeff_names(self):
        """Continuous coefficients this structure needs."""
        names = ["Cgam", "gseed"]
        for i, _ in enumerate(self.terms):
            names += [f"t{i}", f"p{i}"]
        return names

    # -- evaluation --------------------------------------------------------
    def source(self, ctx, coeffs):
        D = drivers(ctx)
        g = np.clip(ctx["gamma"], 0.0, 1.0)
        val = RATES[self.rate](ctx, D) * SHAPES[self.shape](g, coeffs.get("gseed", 0.01))
        for i, (dname, rname) in enumerate(self.terms):
            t = max(coeffs.get(f"t{i}", 1.0), 1e-6)
            p = coeffs.get(f"p{i}", 1.0)
            val = val * RESPONSES[rname](D[dname], t, p)
        return coeffs.get("Cgam", 1.0) * val

    # -- feature vector for a surrogate model ------------------------------
    def features(self):
        """Bag-of-terms descriptor, so a GP/BO surrogate has something to
        put a kernel on. Discrete structures have no natural metric; this
        gives them one."""
        feats = []
        for r in RATES:
            feats.append(1.0 if self.rate == r else 0.0)
        for s in SHAPES:
            feats.append(1.0 if self.shape == s else 0.0)
        active = {(d, r) for d, r in self.terms}
        for d in drivers_names():
            for r in RESPONSES:
                feats.append(1.0 if (d, r) in active else 0.0)
        feats.append(float(len(self.terms)))
        return np.array(feats, dtype=float)


def drivers_names():
    return ["Rev", "Rek", "Reks", "Gk", "Sw", "Ret", "Pe"]


def drivers_names_entropy():
    return drivers_names() + ["Hn", "Hdef"]


def random_candidate(rng, max_terms=2):
    rate = rng.choice(list(RATES))
    shape = rng.choice(list(SHAPES))
    n = int(rng.integers(1, max_terms + 1))
    dn = drivers_names()
    chosen = rng.choice(len(dn), size=n, replace=False)
    terms = [(dn[c], rng.choice(list(RESPONSES))) for c in chosen]
    return Candidate(rate, shape, terms)


def mutate(cand, rng, max_terms=3):
    """One structural mutation: swap a rate, shape, driver, or response,
    or add/remove a term."""
    c = Candidate(cand.rate, cand.shape, list(cand.terms))
    op = rng.choice(["rate", "shape", "driver", "response", "add", "drop"])
    dn = drivers_names()
    if op == "rate":
        c.rate = rng.choice(list(RATES))
    elif op == "shape":
        c.shape = rng.choice(list(SHAPES))
    elif op == "driver" and c.terms:
        i = int(rng.integers(len(c.terms)))
        c.terms[i] = (rng.choice(dn), c.terms[i][1])
    elif op == "response" and c.terms:
        i = int(rng.integers(len(c.terms)))
        c.terms[i] = (c.terms[i][0], rng.choice(list(RESPONSES)))
    elif op == "add" and len(c.terms) < max_terms:
        used = {d for d, _ in c.terms}
        free = [d for d in dn if d not in used]
        if free:
            c.terms.append((rng.choice(free), rng.choice(list(RESPONSES))))
    elif op == "drop" and len(c.terms) > 1:
        i = int(rng.integers(len(c.terms)))
        c.terms.pop(i)
    return c


def crossover(a, b, rng):
    """Recombine two candidates."""
    rate = a.rate if rng.random() < 0.5 else b.rate
    shape = a.shape if rng.random() < 0.5 else b.shape
    pool = list({*a.terms, *b.terms})
    n = max(1, min(len(pool), int(round((a.n_terms() + b.n_terms()) / 2))))
    idx = rng.choice(len(pool), size=n, replace=False)
    return Candidate(rate, shape, [pool[i] for i in idx])
