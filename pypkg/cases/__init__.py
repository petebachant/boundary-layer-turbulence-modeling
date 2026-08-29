"""Benchmark cases for the RANS gym.

Importing this package registers every bundled case. To add one, write a
module here that subclasses :class:`~pypkg.cases.base.BenchmarkCase` and
decorates a factory with ``@register_case``, then import it below.

A case owns three things and nothing else: how to run a closure on this flow,
what the reference data is, and what error counts as "matches the data" for
each metric it scores. Everything about *which* closures exist stays in the
closure registry, so a case never has to know about them.
"""

from . import jhtdb_transitional_bl  # noqa: F401
from . import channel  # noqa: F401
from . import jimenez_zpg  # noqa: F401
from . import naca4412  # noqa: F401
from . import temporal_mixing_layer  # noqa: F401
from .base import BenchmarkCase, log_rms, rel_rms  # noqa: F401

__all__ = ["BenchmarkCase", "rel_rms", "log_rms"]
