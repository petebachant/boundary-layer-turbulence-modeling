"""JHTDB bypass-transitional flat-plate boundary layer.

The case the whole project was built on, and therefore the one every closure
here was calibrated against. It is registered like any other case so that the
leaderboard shows it next to the others and marks it in-sample -- the point of
the harness is that this case gets no special standing.
"""

from __future__ import annotations

import numpy as np

from ..dns_case import Case as _MarchingCase
from ..registry import register_case
from .base import BenchmarkCase, log_rms


class JhtdbTransitionalBL(BenchmarkCase):
    name = "jhtdb-transitional-bl"
    family = "transitional-bl"
    reference = "Zaki2013"

    # Inherited from dns_case.Case so the harness and the coefficient searches
    # cannot drift apart. See dns_case.Case.TARGETS for why each one is here.
    TARGETS = dict(_MarchingCase.TARGETS)

    def __init__(self, root=".", x_stride=4, x_lo=60.0, x_hi=990.0):
        self.case = _MarchingCase(root=root, x_stride=x_stride)
        self.x_lo = x_lo
        self.x_hi = x_hi

    def closure_kwargs(self, spec=None):
        return {"k_inf": self.case.kinf_fn()}

    def run(self, closure):
        return self.case.solve(closure)

    def errors(self, solution):
        U = solution["U"]
        if not np.all(np.isfinite(U)):
            return {"U_rms": np.inf}
        k = solution.get("k")
        sc = self.case.score(U, x_lo=self.x_lo, x_hi=self.x_hi, k=k)
        errs = {m: sc[m] for m in ("cf_rel_rms", "U_rms", "theta_rel_rms")
                if m in sc}
        if "k_log_rms" in sc:
            errs["k_log_rms"] = sc["k_log_rms"]
        # Free-stream decay is scored from the SOLVED k at the top of the
        # domain rather than from the model's analytic decay law. Scoring the
        # law instead is how a free stream 15x too energetic once passed
        # unnoticed; see pypkg/search.freestream_error.
        if k is not None:
            dns = np.array([float(self.case.kinf_fn()(x))
                            for x in self.case.x])
            errs["freestream_rel_rms"] = log_rms(k[-1, :], dns)
        return errs


@register_case(
    "jhtdb-transitional-bl",
    family="transitional-bl",
    reference="Zaki2013",
    description=("JHTDB bypass-transitional flat plate, Re_theta ~ 100-1400. "
                 "The calibration case for every closure in this repository."),
)
def _make(root=".", x_stride=4):
    return JhtdbTransitionalBL(root=root, x_stride=x_stride)
