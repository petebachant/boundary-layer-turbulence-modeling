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
on a transitional plate as on a channel. Those thresholds are judgment calls
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
  closure here initializes a thin pre-transitional state suited to the JHTDB
  plate, so starting them at Re_θ = 4000 with that state measures the initial
  condition, not the model. `jimenez_zpg.SeededClosure` wraps a closure and
  overwrites its state from DNS at initialization — a proxy, so no closure had
  to change. Seed every model by the same rule and exclude the seeded station
  from scoring.
- **Check grid convergence and say what you found.** The Jiménez case notes in
  its constructor that between (181, 300) and (361, 700) the c_f error moves
  1–4 % and the ranking does not change; the NACA 4412 case moves ≤ 3 % over a
  ten-fold increase in cell count. Without that, a leaderboard ordering is not
  evidence.
- **Watch out for relative error on a quantity heading to zero.** The first
  version of the NACA 4412 aft metric was a relative c_f error over the last
  10 % of chord, where c_f collapses to 1.8 × 10⁻⁴. It diverged by
  construction — errors of 700 % for a respectable absolute miss, and the
  *laminar* closure outscoring real turbulence models. It is now an absolute
  error against a fixed scale, plus the shape factor, which is the classical
  separation indicator and is bounded.
- **Seed every case whose inlet is already turbulent, including the ones that
  do not look like inlets.** The channel case originally did not seed, and
  Launder–Sharma returned *exactly zero* eddy viscosity at Re_τ = 180 — fully
  laminar, Ub⁺ = 60 against the DNS 15.69 — while working normally at 550 and
  1000. That was the case's fault, not the model's: k = 0 is a fixed point of
  every closure here, since production is proportional to ν_t and ν_t vanishes
  with k, so a model handed k ≈ 0 can never start. Seeded, the same model
  scores 3.33. The unseeded case was quietly asking whether a low-Re model can
  self-start from nothing, which is a real question but not the one the case
  was built to answer.
- **Verify the case against the reference data's own quantities before
  trusting a single model number.** The NACA 4412 case reproduces the LES's
  published c_f from its wall gradient to 0.2–1 %, and the authors' momentum
  thickness to 0.1–1 %. That check is what separates "the models are bad here"
  from "my case setup is bad here".
- **`evaluate()` never raises.** A candidate that blows up scores infinity and
  records the exception, because a benchmark is pointed at code its author has
  not seen.
- **A temporal flow needs a frame change, not a new solver.** The closures
  march in x at the local U, which conflates the marching coordinate with the
  mean velocity; a temporally evolving layer has U of both signs. The mixing
  layer case wraps every closure in `TranslatingFrame`, which adds U_c ≫ ΔU
  to what the closure sees so that x/U_c = t, and steps the momentum equation
  itself in the laboratory frame. Scores are unchanged to four figures
  between U_c = 100 and 10⁴. The interface fix that would make this proxy
  unnecessary is listed in [roadmap.md §2.3b](roadmap.md).
- **Do not score a peak the model can make grid-dependent.** The vorticity
  thickness ΔU / max|dU/dy| looked like a natural metric for the mixing layer
  until Launder–Sharma's peak gradient turned out to sit at the layer *edge*
  — a front into non-turbulent fluid that sharpens without bound under
  refinement (max|dU/dy| = 39, 84, 169 at n_y = 301, 601, 1201) while the
  momentum thickness moved 1 %. It is now reported but unscored. Integral
  quantities first.

---

## 4. What the suite currently contains

| case | family | what it tests |
|---|---|---|
| `jhtdb-transitional-bl` | transitional-bl | bypass transition, Re_θ ≈ 100–1400. The calibration case for every closure here |
| `jimenez-zpg-tbl` | zpg-tbl | the fully turbulent log layer, Re_θ 4000–6500 |
| `channel-retau-180/1000/5200` | channel | a different geometry: no free stream, no edge, no streamwise development. Also whether the model reaches a steady state at all |
| `naca4412-suction-rec-400000/1000000` | wing-apg | external flow under a severe adverse pressure gradient (β up to 112), driven to the verge of separation |
| `temporal-mixing-layer` | free-shear | a plane mixing layer at ΔU L_x/ν = 250,000 before it rolls up \citep{Lusher2026}: no wall at all, scored on momentum-thickness growth and peak stress and TKE histories over t̂ ∈ [0.14, 0.40] |

Every case except the first is out-of-sample for every closure in the
repository.

**Separated flow is Tier 2 by mathematics, not by preference.** The parabolic
boundary-layer equations carry the Goldstein singularity at separation, so a
marching solver cannot pass a station where c_f = 0. The NACA 4412 case is the
closest the fast tier can get: c_f falls to 1.8 × 10⁻⁴ without ever going
negative.

Still planned: Falkner–Skan laminar as a cheap sanity tier, then the Tier-2
OpenFOAM cases (NACA 0012, cylinder, periodic hill, and the unsteady
roll-up/vortex-merger continuation of the mixing layer). See
[roadmap.md §2](roadmap.md).

---

## 5. What it has found so far

**Three closures had never run.** `EntropyKOmegaH` and `ClipGamma` read
attributes (`freestream_decay`, `x0`) their constructors never set, so they
raised `AttributeError` the moment anything called them with a realistic
free-stream history. Neither was reachable from any existing script, which is
how they stayed broken. Both are fixed.

**The clip closures cannot reach a steady state in a channel.** Not slowly —
at all. `clip-k-omega-gamma` and `clip-k-gamma` stall at residuals of 1e-4 to
4e-3 and their answer *drifts with the pseudo-step*, while `launder-sharma`
lands on an identical answer for pseudo-steps spanning a factor of 800. So the
method is right and the closures are the problem. This is the third
independent sighting of ideas-log §4.3 — "a hard rectifier makes cells toggle
across the threshold" — which previously stalled the OpenFOAM run at its
iteration limit. Different solver, different geometry, same defect. The
harness refuses to score an unconverged run rather than reporting a number
read off a state that is still moving.

**Nothing predicts the approach to separation.** On the NACA 4412 suction side
the shape factor climbs 1.67 → 2.77 toward the trailing edge, which is the
classical separation signature. Over the last 10 % of chord every closure gets
H wrong by 10–36 %, and forward-region c_f wrong by 6–42 %. This is the case
that matters for stall and it is the case nothing passes.

**Three closures produce nothing without a wall.** On the mixing layer,
`clip-gamma`, `clip-k-gamma` and `clip-two-reservoir` score identically to
laminar — peak eddy viscosity 9 × 10⁻⁹, peak shear stress 9 × 10⁻⁷ ΔU². Their
algebraic length scale is a van Driest-damped κy built from wall distance and
wall shear, and with no wall the damping is zero everywhere. Nothing crashed;
the models are simply undefined off the plate. `clip-k-omega-gamma`, whose
length scale is the transported k/ω, is the best model on the case (2.37: the
momentum thickness within 5 %, the peak stress 28 % low), ahead of
Launder–Sharma (4.26). Every transition-gate driver in the closure family
also contains y. See
[shear-layer-vortex-lessons.md §2.1](shear-layer-vortex-lessons.md).

**A momentum-term library fitted on one flow makes every other flow
worse; fitted on four at once, it chooses to be Launder–Sharma.** Six
Galilean-invariant force terms beyond the eddy viscosity
(`pypkg/momentum_library.py`), coefficients found by Bayesian optimization
of the a-posteriori score (`pypkg/bayesopt.py`). On the plate alone: 8.48
→ 3.60 in sample, 3.6 → 8.0 out of sample. On the plate, ZPG layer,
channel and mixing layer jointly: the optimum over 66 evaluations (22
diverged) is the zero vector. `ls-momentum-library` and
`ls-momentum-library-multi` sit in the leaderboard so the two outcomes are
visible side by side.

**The project's headline closure does not survive leaving home.**

| closure | in-sample | out-of-sample (n) | transfer penalty |
|---|---:|---:|---:|
| `launder-sharma` (published coefficients) | — | **4.25** (8) | — |
| `clip-k-gamma` | 1.96 | 5.43 (5) | 2.77 |
| `clip-k-omega-gamma` (8 coefficients fitted here) | **1.95** | 7.55 (5) | **3.87** |

`clip-k-omega-gamma` is the best model in the suite on the one flow it was
fitted to and the worst-transferring of the three that work. Textbook
Launder–Sharma, with no coefficient fitted in this repository, is the best
model out of sample and the only one that runs on all eight cases. The
simpler `clip-k-gamma` transfers better than the elaborate one — though on
the wall-free case it produces nothing at all, which the aggregate hides and
the per-case matrix shows.

That is the paper's thesis, measured across eight flows in five geometries
rather than asserted.

### One correction worth recording

An earlier version of the channel case did not seed the transported scalars,
and Launder–Sharma returned exactly zero eddy viscosity at Re_τ = 180 —
apparently a catastrophic failure, at 127.9. It was the case's fault: k = 0 is
a fixed point of every closure here, so a model handed k ≈ 0 can never start.
Seeded from the DNS, the same model scores **3.33**. The lesson is in §3: a
benchmark can be wrong in a way that looks exactly like a model being wrong,
and only checking the case against the reference data's own quantities tells
the two apart.
