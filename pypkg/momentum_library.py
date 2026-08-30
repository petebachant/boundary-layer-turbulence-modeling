"""A term library for the mean momentum equation, beyond eddy viscosity.

An eddy-viscosity closure lets turbulence act on the mean flow through one
term, d/dy[nu_t dU/dy]. The DNS says that is structurally wrong before
transition -- the stress and strain eigenframes are 44 degrees apart -- so
this module lets a closure add other streamwise forces built from the same
transported quantities, and leaves the data to say which matter.

Design
------
* Every term is a force per unit mass F_x(y) assembled from y-derivatives of
  U, k and nu_t = k/omega only, so it runs in the parabolic tier. Terms with
  x-derivatives break the marching assumption and belong in OpenFOAM.
* Coefficients are dimensionless. Each term is normalized with local
  turbulence scales -- sqrt(k) for velocity, k/omega for a diffusivity --
  so that a coefficient fitted at one viscosity means the same thing at
  another. A dimensional coefficient cannot transfer between cases with
  different nu even in principle (ideas-log 7.1), and leaving that
  possibility in would guarantee a negative result for a trivial reason.
* Every term vanishes where k vanishes, so the library cannot act outside
  the turbulent region, and every term is built from gradients, so it is
  Galilean invariant. Those are the two of Spalart's hard requirements that
  the terms once hard-coded into the OpenFOAM solver violated.

The library is applied on top of a base closure -- Launder-Sharma by default,
since it is the model that transfers best -- so that a zero coefficient
vector recovers the base exactly and the library's whole effect is what the
coefficients buy.

Terms
-----
    nut_Upp   nu_t * U''                 the two halves of the eddy-viscosity
    nutp_Up   nu_t' * U'                 stress divergence, freed to differ
    kp        k'                         the isotropic-stress gradient
    sqk_Up    sqrt(k) * U'               a stress ~ sqrt(k) * velocity, i.e.,
                                         a structure-parameter-like term
    kp_Up_w   k' * U' / omega            gradient-of-energy times strain
    Up2_k_w   (k / omega) * U'^2 / sqrt(k)   strain-squared, normalized
"""

from __future__ import annotations

import numpy as np

TERM_NAMES = ("nut_Upp", "nutp_Up", "kp", "sqk_Up", "kp_Up_w", "Up2_k_w")

#: Coefficient bounds for the search. Every term is O(1) in the units above,
#: so O(1) coefficients are the meaningful range; wider bounds spend the
#: optimizer's budget on diverged runs.
DEFAULT_BOUNDS = {name: (-1.0, 1.0) for name in TERM_NAMES}


def ddy(phi, y):
    return np.gradient(phi, y)


def library_terms(grid, U, k, nut, omega, nu):
    """Evaluate every library term as a force per unit mass on the grid."""
    y = grid.y
    Up = ddy(U, y)
    Upp = ddy(Up, y)
    k = np.maximum(np.asarray(k, dtype=float), 0.0)
    sqk = np.sqrt(k)
    kp = ddy(k, y)
    nutp = ddy(nut, y)
    w = np.maximum(np.asarray(omega, dtype=float), 1e-12)
    return {
        "nut_Upp": nut * Upp,
        "nutp_Up": nutp * Up,
        "kp": kp,
        "sqk_Up": sqk * Up,
        "kp_Up_w": kp * Up / w,
        "Up2_k_w": (k / w) * Up ** 2 / np.maximum(sqk, 1e-12),
    }


class MomentumLibraryClosure:
    """A base closure plus a coefficient-weighted momentum-term library.

    Delegates the transported state and the eddy viscosity to ``base`` and
    adds ``sum_i c_i T_i(y)`` to the streamwise momentum equation through
    the solver's ``momentum_source`` hook. ``omega`` is taken from the base
    state if it carries one, else from epsilon as omega = epsilon / (Cmu k).

    Implements the closure interface by delegation rather than inheritance,
    so this module does not import ``closures`` (which registers it) at
    import time.
    """

    def __init__(self, base=None, base_kwargs=None, cap=5.0, **coeffs):
        c = {name: float(coeffs.pop(name, 0.0)) for name in TERM_NAMES}
        # Whatever is left is for the base closure: the case's k_inf and
        # friends, which the library does not use
        kw = dict(base_kwargs or {})
        kw.update(coeffs)
        if base is None:
            from .closures import LaunderSharma as base
        self.base = base(**kw)
        self.coeffs = c
        self.c = c
        self.state_names = self.base.state_names
        #: Cap on |force| relative to the eddy-viscosity term, so a wild
        #: coefficient stalls the solve rather than sending it to infinity;
        #: None disables it. The default matches the fit stages, since a
        #: coefficient set scored under a different cap is a different model
        self.cap = cap

    # -- delegation to the base closure ------------------------------------

    @property
    def state(self):
        return self.base.state

    @state.setter
    def state(self, value):
        self.base.state = value

    def initialize(self, grid, nu, U, Ue):
        self.base.initialize(grid, nu, U, Ue)

    def eddy_viscosity(self, U, nu, grid):
        return self.base.eddy_viscosity(U, nu, grid)

    def advance(self, grid, U, V, nu, dx, Ue, x):
        self.base.advance(grid, U, V, nu, dx, Ue, x)

    # -- the library ------------------------------------------------------

    def _omega(self, k):
        st = self.base.state
        if "omega" in st:
            return np.asarray(st["omega"], dtype=float)
        eps = np.asarray(st.get("epsilon", np.zeros_like(k)), dtype=float)
        Cmu = getattr(self.base, "Cmu", 0.09)
        return np.maximum(eps, 1e-16) / (Cmu * np.maximum(k, 1e-12))

    def momentum_source(self, grid, U, nu):
        if not any(self.c.values()):
            return None
        k = np.maximum(np.asarray(self.base.state.get("k", 0.0), dtype=float),
                       0.0)
        if np.ndim(k) == 0:
            return None
        nut = self.base.eddy_viscosity(U, nu, grid)
        terms = library_terms(grid, U, k, nut, self._omega(k), nu)
        F = np.zeros(grid.n)
        for name, val in self.c.items():
            if val:
                F += val * terms[name]
        if self.cap is not None:
            ref = np.max(np.abs(terms["nut_Upp"])) + np.max(
                np.abs(terms["nutp_Up"])) + 1e-30
            F = np.clip(F, -self.cap * ref, self.cap * ref)
        return F
