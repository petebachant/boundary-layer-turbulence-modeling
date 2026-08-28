"""The interface every benchmark case implements.

Scoring convention
------------------
A case declares ``TARGETS``: for each metric, the error at which that quantity
counts as "arrived" -- i.e. the point where a careful reader looking at the
plot would say the model matches the data. Every error is then divided by its
own target, so

    normalized == 1.0   means at target on every metric
    normalized <  1.0   means better than "matches by inspection"

and that number means the same thing on a transitional boundary layer as on a
channel, which is what makes a cross-case leaderboard legible. The unnormalised
``total`` is the sum rather than the mean and is kept because the existing
coefficient searches optimize it.

The targets are judgment calls and they are the most contestable thing in the
harness, so they live in one visible place per case rather than being buried in
a weighting inside an objective function.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


class BenchmarkCase(ABC):
    """A flow, its reference data, and how to score a closure against it."""

    #: Registry name; set by the case module.
    name: str = ""
    #: Coarse grouping for the leaderboard, e.g. "zpg-tbl", "channel".
    family: str = ""
    #: BibTeX key for the reference data.
    reference: str = ""
    #: How the reference data was produced. Recorded per case because it bounds
    #: what a score on that case can mean, and because it is the first thing a
    #: reader should be told rather than something they have to infer from the
    #: citation.
    #:
    #: "dns"  -- every scale resolved; the reference is the equations.
    #: "les"  -- well-resolved large-eddy simulation. The sub-grid model is
    #:           itself a turbulence model, so a RANS closure scored against it
    #:           is being compared to a model rather than to the equations.
    #:           Two consequences that matter here: published LES statistics
    #:           normally contain the RESOLVED stresses only, so any k-based
    #:           metric is biased low by the sub-grid share; and agreement can
    #:           never be claimed tighter than the LES's own uncertainty.
    #: "experiment" -- measured.
    fidelity: str = "dns"
    #: metric -> error at which that metric counts as matching the data
    TARGETS: dict[str, float] = {}

    # -- what the closure needs from the flow ------------------------------

    def closure_kwargs(self, spec=None) -> dict:
        """Case-specific context passed to the closure constructor.

        Free-stream conditions are a property of the flow, not of the model,
        so the case supplies them rather than the closure guessing.
        """
        return {}

    # -- running and scoring -----------------------------------------------

    @abstractmethod
    def run(self, closure) -> dict:
        """Solve this case with ``closure``. Returns a solution dict."""

    @abstractmethod
    def errors(self, solution) -> dict[str, float]:
        """Raw errors against the reference data, keyed as in ``TARGETS``."""

    def score(self, solution) -> dict:
        """Errors plus the normalized aggregate. Infinite if the run diverged."""
        if solution is None:
            return self._failed()
        errs = self.errors(solution)
        if not all(np.isfinite(v) for v in errs.values()):
            return self._failed(errs)
        terms = {k: errs[k] / t for k, t in self.TARGETS.items() if k in errs}
        if not terms:
            raise ValueError(f"case {self.name!r} produced no scorable metrics")
        total = float(sum(terms.values()))
        return {
            **{k: float(v) for k, v in errs.items()},
            "total": total,
            "normalized": total / len(terms),
            "n_terms": len(terms),
            "diverged": False,
        }

    def _failed(self, errs=None):
        out = {k: float(v) for k, v in (errs or {}).items()}
        out.update({"total": math.inf, "normalized": math.inf,
                    "n_terms": len(self.TARGETS), "diverged": True})
        return out

    def evaluate(self, closure) -> dict:
        """Run and score, turning a blow-up into an infinite score.

        Candidate closures are expected to diverge -- that is what a search
        does -- so this must never raise.
        """
        try:
            sol = self.run(closure)
        except Exception as exc:
            # Deliberately broad. A benchmark exists to be pointed at code its
            # author has not seen, and one candidate raising must not lose the
            # results of every closure after it in the loop. The exception is
            # reported rather than swallowed, so a crash is still visible as a
            # crash and not as a merely bad score.
            out = self._failed()
            out["error"] = f"{type(exc).__name__}: {exc}"
            return out
        return self.score(sol)

    # -- reporting ----------------------------------------------------------

    def reference_score(self) -> dict:
        """Score of the reference data against itself.

        Zero by definition where it is defined, but cases that subsample or
        re-derive quantities can have a nonzero floor, and a leaderboard entry
        below the floor is measuring noise rather than skill.
        """
        return {}

    def describe(self) -> dict:
        return {"name": self.name, "family": self.family,
                "reference": self.reference, "fidelity": self.fidelity,
                "targets": dict(self.TARGETS)}


def rel_rms(model, data, floor=1e-30):
    """RMS of the relative error, the harness's default error measure."""
    model = np.asarray(model, dtype=float)
    data = np.asarray(data, dtype=float)
    d = np.where(np.abs(data) < floor, floor, data)
    return float(np.sqrt(np.mean(((model - d) / d) ** 2)))


def log_rms(model, data, floor=1e-30):
    """RMS of log(model/data). For quantities spanning decades, such as k."""
    model = np.maximum(np.asarray(model, dtype=float), floor)
    data = np.maximum(np.asarray(data, dtype=float), floor)
    return float(np.sqrt(np.mean(np.log(model / data) ** 2)))
