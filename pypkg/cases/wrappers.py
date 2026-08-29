"""Proxies that adapt a closure to a case without modifying the closure.

Every closure in this repository was written for one flow: a boundary layer
marching along a flat plate, with a free stream above it. Two of the cases in
the suite are not that, and both mismatches are properties of the *flow*, not
defects in the models:

* a fully turbulent inlet needs an initial state the closures do not produce
  (they all seed a thin pre-transitional profile);
* a channel centerline is a symmetry plane, but every closure pins a Dirichlet
  free-stream value at the top node;
* a temporally evolving flow has no marching direction, but every closure
  marches in x at the local mean velocity.

Fixing either inside the closures would mean editing all seven, and would mean
a contributor's new closure has to know about every case before it can be
scored. So the adaptation lives here, in proxies applied identically to every
model, and is stated in each case's docstring so nobody has to guess what was
done to their closure.

A proxy delegates everything it does not override, so a closure sees no
difference and the harness reads ``state`` and ``state_names`` through it.
"""

from __future__ import annotations

import numpy as np


class ClosureProxy:
    """Delegates every attribute to the wrapped closure."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    @property
    def inner(self):
        return object.__getattribute__(self, "_inner")

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def initialize(self, grid, nu, U, Ue):
        return self.inner.initialize(grid, nu, U, Ue)

    def eddy_viscosity(self, U, nu, grid):
        return self.inner.eddy_viscosity(U, nu, grid)

    def advance(self, grid, U, V, nu, dx, Ue, x):
        return self.inner.advance(grid, U, V, nu, dx, Ue, x)


class SeededClosure(ClosureProxy):
    """Overwrite the transported state from data at initialization.

    For cases whose inlet is a developed turbulent profile. Only the state
    entries the closure actually carries are touched, and only once.
    """

    def __init__(self, inner, seed_fn):
        super().__init__(inner)
        object.__setattr__(self, "_seed_fn", seed_fn)

    def initialize(self, grid, nu, U, Ue):
        inner = self.inner
        inner.initialize(grid, nu, U, Ue)
        seed = object.__getattribute__(self, "_seed_fn")(grid, nu, U, Ue)
        for key, val in seed.items():
            if key in inner.state:
                inner.state[key] = np.asarray(val, dtype=float).copy()


class SymmetryTop(ClosureProxy):
    """Turn the closure's free-stream top boundary into a symmetry plane.

    Needed for the channel, where the top of the domain is the centerline. The
    closures apply a Dirichlet free-stream value there -- ``free_value=kinf``
    and friends inside ``advance`` -- which would clamp the centerline
    turbulence to a free-stream level that does not exist in a duct.

    After each step the top node is overwritten with the node below it. At
    convergence the two are equal, so the interior equation at the second-from-
    top node has effectively been solved against a zero-gradient ghost, which
    is the symmetry condition. The pinned free-stream value never survives into
    the converged answer.
    """

    def advance(self, grid, U, V, nu, dx, Ue, x):
        inner = self.inner
        out = inner.advance(grid, U, V, nu, dx, Ue, x)
        for name in inner.state_names:
            arr = inner.state.get(name)
            if arr is not None and np.ndim(arr) == 1 and len(arr) >= 2:
                arr[-1] = arr[-2]
        return out


class TranslatingFrame(ClosureProxy):
    """Show the closure the flow in a frame moving at -Uc, i.e. U + Uc.

    For temporal problems. The closures march in x using the local U as the
    convection speed, so they conflate the marching coordinate with the mean
    velocity; a temporally evolving flow with U of both signs has no marching
    direction at all. Adding a constant Uc >> |U| to what the closure sees
    makes its convection speed uniform to within |U|/Uc, so its x is time
    times Uc. The shear is unchanged, and with it every production term.

    The case must call ``advance`` with ``dx = Uc*dt`` and ``x = Uc*t``. The
    momentum equation is the case's own business and is not transformed.
    """

    def __init__(self, inner, Uc):
        super().__init__(inner)
        object.__setattr__(self, "_Uc", float(Uc))

    @property
    def Uc(self):
        return object.__getattribute__(self, "_Uc")

    def initialize(self, grid, nu, U, Ue):
        return self.inner.initialize(grid, nu, U + self.Uc, Ue + self.Uc)

    def eddy_viscosity(self, U, nu, grid):
        return self.inner.eddy_viscosity(U + self.Uc, nu, grid)

    def advance(self, grid, U, V, nu, dx, Ue, x):
        return self.inner.advance(grid, U + self.Uc, V, nu, dx,
                                  Ue + self.Uc, x)
