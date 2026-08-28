# The RANS gym: adding a closure, adding a case

> **Authorship.** Written by Claude (Anthropic's Claude Code), prompted by Pete
> Bachant.

A benchmarking harness for RANS closures. You supply a closure; it is solved
and scored **a posteriori** — actually run in a solver — against every case in
the suite, and the result separates the flows your coefficients were fitted on
from the flows they were not.

```
calkit run run-benchmark          # or:
python scripts/run-benchmark.py
```

---

## 1. Why it is built this way

Three design decisions, each of which exists because of something that went
wrong in this project.

**It scores a posteriori, not a priori.** The obvious cheap benchmark is a
table of features and DNS labels, and you score regression error against it.
That is what the standard open dataset provides \citep{McConkey2021}. The
problem is that a-priori accuracy does not imply a-posteriori accuracy — the
field's most consistently reported methodological finding
\citep{Duraisamy2021, Zhao2020} — and we have reproduced it in-house: the
a-priori term regression in `scripts/regress-pde-terms.py` reaches R² = 0.953
in-sample while the a-posteriori search selects different terms. So the harness
runs a solver. It is slower and it is the only version that measures the thing
a user will experience.

**Every closure declares what it was calibrated on.** `calibrated_on` in the
registry is not documentation; the leaderboard uses it to split the results.
The finding this whole project rests on is that in-sample agreement did not
distinguish a constitutive law from a curve fit. A benchmark reporting one
aggregate number would make that mistake the default, so the split is
structural rather than a reporting convention.

**Targets are declared per case, in one visible place.** A case's `TARGETS`
maps each metric to the error at which that quantity counts as matching the
data. Errors are divided by their target, so a `normalized` score of 1.0 means
"matches the data by inspection on every metric", and it means the same thing
on a transitional plate as on a channel. Those thresholds are judgement calls
and the most contestable thing in the harness, which is exactly why they sit at
the top of each case class instead of being buried as weights inside an
objective function.

---

## 2. Add a closure

A closure implements three methods. `state_names` lists the scalars it
transports; the solver allocates and carries them.

```python
import numpy as np
from pypkg.closures import Closure, ddy
from pypkg.registry import register_closure


@register_closure(
    "my-model",
    description="One line, shown in the leaderboard.",
    calibrated_on=(),        # case names you fitted coefficients on
    coeffs={"Cmu": 0.09},    # defaults, or a callable returning them
)
class MyModel(Closure):
    state_names = ("k",)

    def __init__(self, Cmu=0.09, k_inf=None, **kw):
        super().__init__(**kw)
        self.Cmu = Cmu
        self.k_inf = k_inf          # cases pass the free-stream history here

    def initialize(self, grid, nu, U, Ue):
        self.state = {"k": np.full(grid.n, 1e-8)}

    def eddy_viscosity(self, U, nu, grid):
        return self.Cmu * np.sqrt(np.maximum(self.state["k"], 0.0)) * grid.y

    def advance(self, grid, U, V, nu, dx, Ue, x):
        ...                          # march self.state one station in x
```

**`calibrated_on` is the field to get right.** Leave it empty only if the
coefficients are published or derived rather than fitted here — a closure with
an empty tuple is out-of-sample everywhere, which is a strong claim. Anything
tuned against a case in this suite must name it.

Register it without editing the package by pointing at your module:

```bash
RANS_GYM_PLUGINS=my_closures python scripts/run-benchmark.py
```

Bundled closures are registered at the bottom of `pypkg/closures.py`,
deliberately as calls rather than decorators so the fitted coefficients — which
live in pipeline outputs that may not exist yet — load lazily.

### Coefficients that live in pipeline outputs

```python
from pypkg.registry import coeffs_from_json

coeffs=coeffs_from_json("results/closure-params.json", "coeffs", fallback={})
```

If a closure's coefficients were fitted under particular solver settings, ship
the settings with them. Scoring coefficients under a different configuration
measures the configuration, not the model — see `_clip_k_omega_gamma_coeffs`,
and ideas-log §4.11 for the ablation table that had to be retracted for exactly
this reason.

---

## 3. Add a case

A case owns three things: how to run a closure on this flow, what the reference
data is, and what error counts as matching it.

```python
from pypkg.cases.base import BenchmarkCase, rel_rms
from pypkg.registry import register_case


class MyCase(BenchmarkCase):
    name = "my-case"
    family = "zpg-tbl"
    reference = "SomeBibKey"                  # add it to references.bib
    TARGETS = {"cf_rel_rms": 0.02, "U_rms": 0.01}

    def closure_kwargs(self, spec=None):
        return {"k_inf": self.kinf_fn()}      # flow properties, not model ones

    def run(self, closure):
        ...                                    # returns {"U": ..., "k": ...}

    def errors(self, solution):
        return {"cf_rel_rms": ..., "U_rms": ...}


@register_case("my-case", family="zpg-tbl", reference="SomeBibKey")
def _make(root="."):
    return MyCase(root=root)
```

Then import the module from `pypkg/cases/__init__.py`.

Notes from building the existing cases:

- **The case supplies flow properties, the closure supplies the model.**
  Free-stream turbulence history is a property of the flow, so it goes in
  `closure_kwargs`, not in a closure default.
- **If your inlet is fully turbulent, seed the transported scalars.** Every
  closure here initialises a thin pre-transitional state suited to the JHTDB
  plate, so starting them at Re_θ = 4000 with that state measures the initial
  condition, not the model. `jimenez_zpg.SeededClosure` wraps a closure and
  overwrites its state from DNS at initialisation — a proxy, so no closure had
  to change. Seed every model by the same rule and exclude the seeded station
  from scoring.
- **Check grid convergence and say what you found.** The Jiménez case notes in
  its constructor that between (181, 300) and (361, 700) the c_f error moves
  1–4 % and the ranking does not change. Without that, a leaderboard ordering
  is not evidence.
- **`evaluate()` never raises.** A candidate that blows up scores infinity and
  records the exception, because a benchmark is pointed at code its author has
  not seen.

---

## 4. What the suite currently contains

| case | family | Re range | status |
|---|---|---|---|
| `jhtdb-transitional-bl` | transitional-bl | Re_θ ≈ 100–1400 | the calibration case for everything here |
| `jimenez-zpg-tbl` | zpg-tbl | Re_θ 4000–6500 | out-of-sample for every closure |

Planned, in priority order: turbulent channel at Re_τ = 180–5200
\citep{LeeMoser2015}, an adverse-pressure-gradient layer \citep{Bobke2017},
Falkner–Skan laminar, then the Tier-2 OpenFOAM cases (NACA 0012, cylinder). See
[roadmap.md §2](roadmap.md).

---

## 5. What it found on the first run

The harness earned its keep before the second case was finished.

**Two closures had never run.** `EntropyKOmegaH` and `ClipGamma` both read
attributes (`freestream_decay`, `x0`) that their constructors never set, so
they raised `AttributeError` the moment anything called them with a realistic
free-stream history. Both are now fixed. Neither was reachable from any
existing script, which is how they stayed broken.

**The project's own closure loses out of sample to a textbook model.** On the
Jiménez ZPG layer, with every model given the same DNS seed and the same grid:

| closure | c_f rel. RMS | normalized |
|---|---:|---:|
| `launder-sharma` (published coefficients) | 0.024 | 1.57 |
| `clip-k-omega-gamma` (8 coefficients fitted here) | 0.065 | 1.77 |

On the transitional plate it was fitted to, `clip-k-omega-gamma` scores 1.95
against Launder–Sharma's 8.48 — a 4.4× advantage. Move to an equilibrium
turbulent layer it never saw and the advantage inverts, with c_f error 2.7×
worse than the textbook model's. The fitted closure is better at k (0.24
against 0.48 log-RMS) and at momentum thickness (1.1 % against 2.8 %), and
worse where it matters for drag.

That is the paper's thesis, measured rather than asserted, from one case that
required no new data — the profiles had been sitting in `data/jiminez/` being
used only for a-priori regression.
