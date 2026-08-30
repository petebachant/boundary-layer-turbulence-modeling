"""Registries for the benchmarking harness (the "RANS gym").

A closure and a case are registered independently, and the harness runs the
cross product. The point of the indirection is one line in ``ClosureSpec``:

    calibrated_on

Every closure has to declare which cases its coefficients were fitted on, so
the leaderboard can split in-sample from out-of-sample *by construction*
rather than by whoever writes the results table remembering to. This project
exists because in-sample agreement was mistaken for a constitutive law; a
harness that made the same mistake easy to repeat would be worse than no
harness.

Adding a closure
----------------
    from pypkg.closures import Closure
    from pypkg.registry import register_closure

    @register_closure("my-model", calibrated_on=(), coeffs={"C1": 0.09})
    class MyModel(Closure):
        ...

Adding a case: see ``pypkg.cases``.

Third-party closures and cases are picked up from any module listed in the
``RANS_GYM_PLUGINS`` environment variable (comma-separated), so a new idea can
be scored without editing this package.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

# Tier 1 is the fast Python screen, tier 2 the OpenFOAM confirmation. A closure
# may live in both; a case belongs to exactly one.
TIER_PYTHON = "python"
TIER_OPENFOAM = "openfoam"


@dataclass
class ClosureSpec:
    """A registered closure and everything the harness needs to run it."""

    name: str
    cls: type
    description: str = ""
    #: Case names whose data was used to fit these coefficients. Empty means
    #: the coefficients are published/derived rather than fitted here, which is
    #: the only way a closure is out-of-sample everywhere.
    calibrated_on: tuple[str, ...] = ()
    #: Default coefficients, or a callable returning them (used when they live
    #: in a pipeline output that may not exist yet).
    coeffs: dict[str, Any] | Callable[[], dict[str, Any]] = field(
        default_factory=dict
    )
    #: Name of the matching OpenFOAM RAS model, if this closure has one.
    openfoam_model: str | None = None
    #: False for a closure that exists only as an OpenFOAM model (the
    #: published baselines), which the Python tier must skip rather than
    #: report as crashed.
    python_tier: bool = True
    reference: str = ""

    def get_coeffs(self) -> dict[str, Any]:
        c = self.coeffs() if callable(self.coeffs) else self.coeffs
        return dict(c or {})

    def is_in_sample(self, case_name: str) -> bool:
        return case_name in self.calibrated_on

    def build(self, case=None, **overrides):
        """Instantiate the closure, letting the case supply its own context.

        Cases hand over things like the free-stream ``k`` history that the
        closure needs but that are a property of the flow, not of the model.
        """
        kw = self.get_coeffs()
        if case is not None:
            kw.update(case.closure_kwargs(self))
        kw.update(overrides)
        return self.cls(**kw)


@dataclass
class CaseSpec:
    """A registered benchmark case, built lazily.

    Cases load DNS data and are expensive to construct, so the registry holds
    a factory and the harness builds only what it is asked to run.
    """

    name: str
    factory: Callable[..., Any]
    family: str = ""
    tier: str = TIER_PYTHON
    description: str = ""
    reference: str = ""
    _built: Any = None

    def build(self, **kwargs):
        if kwargs:
            return self.factory(**kwargs)
        if self._built is None:
            self._built = self.factory()
        return self._built


CLOSURES: dict[str, ClosureSpec] = {}
CASES: dict[str, CaseSpec] = {}


def register_closure(name, *, description="", calibrated_on=(), coeffs=None,
                     openfoam_model=None, reference="", python_tier=True):
    """Class decorator registering a closure under ``name``."""

    def deco(cls):
        if name in CLOSURES:
            raise ValueError(f"closure {name!r} is already registered")
        CLOSURES[name] = ClosureSpec(
            name=name,
            cls=cls,
            description=description or (cls.__doc__ or "").strip().split("\n")[0],
            calibrated_on=tuple(calibrated_on),
            coeffs=coeffs if coeffs is not None else {},
            openfoam_model=openfoam_model,
            python_tier=python_tier,
            reference=reference,
        )
        cls.gym_name = name
        cls.openfoam_model = openfoam_model
        return cls

    return deco


def register_case(name, factory=None, *, family="", tier=TIER_PYTHON,
                  description="", reference=""):
    """Register a case. Usable directly or as a decorator on the factory."""

    def deco(fn):
        if name in CASES:
            raise ValueError(f"case {name!r} is already registered")
        CASES[name] = CaseSpec(
            name=name, factory=fn, family=family, tier=tier,
            description=description or (fn.__doc__ or "").strip().split("\n")[0],
            reference=reference,
        )
        return fn

    if factory is not None:
        deco(factory)
        return factory
    return deco


def coeffs_from_json(path, key="coeffs", fallback=None):
    """Late-bound coefficients read from a pipeline output.

    Registration happens at import time, when the pipeline output may not
    exist yet, so this returns a callable rather than reading immediately.
    """

    def load():
        if not os.path.isfile(path):
            if fallback is None:
                raise FileNotFoundError(
                    f"{path} not found -- run the stage that produces it, or "
                    f"register a fallback"
                )
            return dict(fallback)
        with open(path) as f:
            d = json.load(f)
        for part in key.split("/"):
            d = d[part]
        return dict(d)

    return load


def load_plugins():
    """Import any modules named in ``RANS_GYM_PLUGINS`` so they can register."""
    mods = os.environ.get("RANS_GYM_PLUGINS", "")
    loaded = []
    for m in [s.strip() for s in mods.split(",") if s.strip()]:
        importlib.import_module(m)
        loaded.append(m)
    return loaded


def load_builtins():
    """Import the bundled closures and cases so the registries are populated."""
    importlib.import_module("pypkg.closures")
    importlib.import_module("pypkg.cases")
    load_plugins()


def closures(tier=None):
    load_builtins()
    if tier == TIER_OPENFOAM:
        return {k: v for k, v in CLOSURES.items() if v.openfoam_model}
    if tier == TIER_PYTHON:
        return {k: v for k, v in CLOSURES.items() if v.python_tier}
    return dict(CLOSURES)


def cases(tier=TIER_PYTHON, family=None):
    load_builtins()
    out = {k: v for k, v in CASES.items() if tier is None or v.tier == tier}
    if family is not None:
        out = {k: v for k, v in out.items() if v.family == family}
    return out
