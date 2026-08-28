# Roadmap and open TODOs

> **Authorship.** Written by Claude (Anthropic's Claude Code), prompted by Pete
> Bachant. This consolidates the GitHub issue tracker, the open threads in
> [ideas-log.md](ideas-log.md), and the plan for the benchmarking harness
> ("RANS gym") agreed 2026-08-28. Status assessments of GitHub issues are
> Claude's reading of the repository at that date, not the issue author's.

---

## 1. The plan, in one paragraph

The project so far is one DNS case, one closure family, and a negative result:
coefficients discovered on the JHTDB bypass-transitional boundary layer do not
transfer. Two things follow. **Generality has to be tested against more than
one flow**, so we add cases. And **the check should be routine rather than
heroic**, so we build a harness in which any closure can be dropped in and
scored across every case in a common framework. The
[literature review](learning-closures-lit-review.md) §6 argues the harness is
the defensible contribution: curated *datasets* for a-priori work exist
\citep{McConkey2021}, but an open a-posteriori harness that runs a supplied
closure across a case suite does not appear to.

---

## 2. RANS gym: the harness

Two tiers, per PB's decision on 2026-08-28.

**Tier 1 — Python screen.** A closure is one class implementing
`initialize` / `eddy_viscosity` / `advance` (the existing `Closure` ABC in
`pypkg/closures.py`). Solved with the fast parabolic solver or the new
equilibrium solver, scored in seconds against every Tier-1 case.

**Tier 2 — OpenFOAM confirm.** A closure may additionally register a matching
OpenFOAM model. Slower, needs the Docker toolchain, and is where the elliptic
cases (airfoil, cylinder) live because they cannot be marched.

### 2.1 Core (blocking everything else)
- [ ] `pypkg/registry.py` — decorator registries for closures and cases,
      plus discovery so a user can drop in a file without editing the package.
- [ ] Each closure declares `calibrated_on`, so the leaderboard can separate
      in-sample from out-of-sample **by construction**. This is the single most
      important design decision in the harness; it is what stops the tool from
      reproducing the failure mode the paper is about.
- [ ] `pypkg/cases/base.py` — a `BenchmarkCase` interface: `name`,
      `family`, `tier`, `run(closure)`, `score(solution)`.
- [ ] Refactor `dns_case.Case` into `cases/jhtdb_transitional_bl.py` behind
      that interface, keeping the current import path working so the existing
      scripts and stages do not break.
- [ ] Register the existing closures (`Laminar`, `LaunderSharma`, `ClipGamma`,
      `ClipTwoReservoir`, `ClipKGamma`, `ClipKOmegaGamma`, `EntropyKOmegaH`).
- [ ] `scripts/run-benchmark.py` + pipeline stage → `results/benchmark.json`.
- [ ] Leaderboard table and figure; wire into `make-paper-numbers.py`.
- [ ] `docs/rans-gym.md` — how to add a closure, how to add a case, what the
      scoring protocol is and why.

### 2.2 Tier-1 cases
- [x] **Jiménez/Sillero ZPG TBL, Re_θ = 4000–6500** \citep{Sillero2013}.
      **Done.** Data was already in `data/jiminez/`, used only by
      `regress-pde-terms.py`. Marches from the Re_θ = 4000 station with the
      spacing recovered from the ZPG momentum integral dθ/dx = c_f/2 — nothing
      fitted. Tests the fully-turbulent log layer at Re_θ 3–5× above anything
      the JHTDB case reaches.
- [x] **Turbulent channel, Re_τ = 180/1000/5200** \citep{LeeMoser2015}.
      **Done.** No momentum BVP is needed: with δ = u_τ = 1 the total stress is
      exactly linear, so U follows from one integration once the closure
      supplies ν_t, and the transported scalars relax with the same `advance`
      the marching solver uses. Verified dx-independent — launder-sharma lands
      on the same answer for pseudo-steps spanning a factor of 800. Re_τ 550
      and 2000 are fetched and reachable via `make_channel` but not registered,
      so one geometry does not dominate the leaderboard by weight of numbers.
- [x] **NACA 4412 suction side** \citep{Vinuesa2018}. **Done**, at
      Re_c = 400k and 1M. External flow under a severe adverse pressure
      gradient — β reaches 112 — driven to the verge of separation, with c_f
      falling to 1.8e-4 without ever going negative. That last part is what
      makes it possible in the fast tier at all: the parabolic equations carry
      the Goldstein singularity at separation, so genuinely separated flow is
      Tier 2 by mathematics rather than by preference.

      Two decisions worth knowing about. The outer boundary imposes the LES's
      own Ue(s) rather than the measured profile at a fixed height: imposing
      the latter was tried and rejected because across domain heights of
      1.1–4.7 δ99 it distorted the streamwise pressure gradient by 18–38 % RMS,
      so the case would have measured the domain-height choice as much as the
      closure. Nothing above δ99 is scored, so the thin-layer idealization that
      introduces never enters a model's score. And the aft metrics are
      deliberately *not* relative c_f: a relative error on a quantity heading
      to zero diverges by construction, and on that metric the laminar closure
      outscored real turbulence models.

      Setup validated against the LES's own numbers: c_f recovered from the
      wall gradient to 0.2–1 %, the authors' momentum thickness to 0.1–1 %,
      and H climbing 1.67 → 2.77 toward the trailing edge. Grid-converged to
      within 3 % over a tenfold increase in cells.
- [ ] **Flat-plate APG TBL** \citep{Bobke2017}. Fetchable (one 305 MB Google
      Drive .mat). Lower priority now the wing data is in hand: weaker
      pressure gradient, simpler geometry.
- [ ] **Falkner–Skan laminar family.** Self-generated ODE reference, no
      download. Cheap sanity tier: catches closures that pollute laminar
      regions or misread favorable/adverse `dUedx`. Not a headline result,
      but it is the fastest possible regression test.

### 2.3 Tier-2 cases
- [ ] Formalise the existing OpenFOAM runs (`sim/cases/*`) as registered
      Tier-2 cases rather than ad hoc stages.
- [ ] **NACA 0012** \citep{Ladson1988, RumseyTMR} — the airfoil case PB asked
      for. NASA TMR supplies grids and reference data, so the setup is not
      ours to invent.
- [ ] **Circular cylinder** — separation from a smooth surface, which is the
      canonical place RANS fails. Reference data and Reynolds number still to
      be chosen.
- [ ] Cross-tier consistency check: a closure implemented in both tiers should
      agree on the cases both can run. Any disagreement is a bug in one of the
      two implementations, and the harness should say so loudly.

---

## 2b. Tooling: environments and pipeline hygiene — **DONE 2026-08-28**

### Environments converted to uv, split three ways
uv, not pixi — the whole dependency set resolves and builds on **Python 3.14**
(numpy 2.5.2, scipy 1.18.1, tables 3.11.1, matplotlib 3.11.1).

| environment | `envs/compute` | `envs/viz` | `envs/notebook` |
|---|---|---|---|
| contents | numerics only | numerics + plotting | + Jupyter, calkit |
| stages | 16 | 6 | 1 |

Each is a uv *project* env with its own `uv.lock`, so a package added for
plotting invalidates only the six figure stages and not the sixteen that carry
closure searches, fits and benchmark runs. `pypkg` is installed editable
into all three, which removes the `sys.path` boilerplate; note that an editable
dependency's *contents* are not hashed into `uv.lock`, so stages must keep
naming the specific modules they use as inputs. That is finer-grained than
hashing the package anyway — editing a plotting helper should not invalidate a
closure search.

`environment.yml` (a ~200-package `conda env export` dump) is gone.

**The numpy 1.20 → 2.5 jump needed real fixes**, all now applied: `np.trapz`
was removed in numpy 2 and appeared 27 times across six files (now
`np.trapezoid`), and `pandas.read_csv(delim_whitespace=)` is gone. All ten
compute stages were re-run under 3.14 and pass; `results/blasius-validation.json`
moves only in the 13th significant figure, which is the right size for a change
of BLAS and accumulation order.

`pyjhtdb` is out of every pipeline environment. It needs a JHTDB token, imports
`pkg_resources` in `setup.py`, and uses numpy APIs removed in 1.24, so it would
pin the project to an ancient numpy. It now lives in
`scripts/standalone/requirements.txt`, installed separately only when someone
actually needs the web service.

### Environments, corrected after review
- **`py-jhtdb` is now a declared environment** (`envs/jhtdb/`), pinned to
  Python 3.11 and numpy < 1.24, rather than a loose requirements file. No
  pipeline stage uses it, so no stage depends on its lock and it cannot
  invalidate a result -- but the token-gated path is reproducible instead of
  folklore. Each environment carries a `.python-version`.
- **`envs/notebook` went from 301 packages to 80.** It had `calkit-python`,
  `jupyter` and `nbconvert` in it, all wrong: `calkit nb execute` drives the
  environment *from outside* and runs nbconvert with its own `sys.executable`,
  so only `ipykernel` is needed inside. The notebook does not import calkit.
- **`scikit-learn` was missing** from that environment. The notebook does
  `from sklearn.linear_model import LinearRegression` in four cells, and the
  first dependency audit only grepped `.py` files. Environment contents are now
  derived by walking each stage script's AST and following into `pypkg`,
  with the two runtime deps that imports cannot reveal (`tables` for
  `pandas.read_hdf`, `kaleido` for `plotly.to_image`) noted in the file.
- **`envs/viz` trimmed** to what the six viz stages actually import: `plotly`,
  `kaleido` and `IPython` moved to the notebook environment, since
  `pypkg.plotting` is the only thing that uses them and only the notebook
  imports it.

### Naming and ordering
- Paper figures lost the `paper-` prefix: `dissipation`, `transfer`,
  `collapse`, `fit-noise`. `paper-benchmark` became **`model-comparison`**
  rather than `benchmark`, because the gym now owns that word
  (`results/benchmark.json`, and a leaderboard figure is a planned stage).
- `plot-paper-figures` -> stage `plot`, script `scripts/plot.py`.
- Pipeline stages in `calkit.yaml` are **reordered into execution order**,
  topologically sorted from the real dependency graph in the generated
  `dvc.yaml` (with the previous file order as the tie-break, to keep the diff
  readable) rather than by hand.

### Pipeline pruned
- **`compute-coeffs` deleted.** It was literally
  `mkdir -p results && echo "{}" > results/coeffs.json` — a placeholder
  producing an empty file. `results/coeffs.json` and its dataset entry are gone
  with it.
- **`extract-jhtdb-stats` replaced by `make-dns-stats-table`.** The old stage
  called `setup-dns.py` → `read_stats()`, which *gdown-downloaded a prebuilt
  file from Google Drive* if one was missing. That made `all-stats.h5` an
  opaque fetch rather than a derivation, and put `pyJHTDB` on the critical path
  of a pipeline that never called it. The new stage is a pure function of the
  tracked `time-ave-profiles.h5`, runs in `py-compute`, needs no token, and was
  verified to reproduce the old table **bit-for-bit on all 23 columns that have
  a consumer**. The 36 JHTDB web-service columns were dropped: they were
  populated at 100 of 743,680 rows (0.013 %) and nothing reads them —
  `plot-bl-dns.py` and the three scripts under `sim/` use only `y` and `u`.
- `pypkg/dns_stats.py` is the new pyJHTDB-free home for `read_profiles`,
  which previously sat in a module that imports `pyJHTDB` and calls
  `matplotlib.use("nbAgg")` at import time.
- `notebook.ipynb` → `notebooks/main.ipynb`.

### Still open here
- [ ] `save-mesh-snapshot-isometric` still emits a 0-byte PNG and reports
      success (issue #14, ideas-log §4.15). Make it fail loudly or fix it.
- [ ] Audit the remaining `datasets:` entries against reality — one is marked
      "virtual since it doesn't actually exist".
- [ ] Consider converting the remaining `_system` stages
      (`save-mesh-snapshot-isometric`, `paper-numbers-to-latex`) to declared
      environments, and some `shell-command` stages to `python-script` /
      `shell-script` kinds with args.

---

## 3. GitHub issues

Status is Claude's assessment against the repository at 2026-08-28 and should
be confirmed before closing anything.

| # | title | assessment |
|---|---|---|
| 19 | Run and compare laminar simulation | **Appears done.** `laminar-sim` stage exists; `laminar-wall-resolved` and `laminar-dns-domain` cases run and appear in the paper's model table. |
| 18 | Create dedicated environment for notebook | **Open.** Needed since `calkit` left `main-python`. |
| 17 | Wrap this up and summarize results | **Superseded.** `summarize-findings` exists, but the project is being extended rather than wrapped up. Reframe or close. |
| 16 | Figure showing the RANS momentum balance works out | **Open.** No such figure in `figures/`. Worth doing — it is the sanity check underneath every term-library claim, including §7.1. |
| 15 | Clean up terms to ensure they're all vectors | **Probably obsolete**, from the notebook era. Confirm and close. |
| 14 | Generate a figure from the RANS mesh | **Open, and worse than it looks.** `scripts/save-mesh-snapshot.sh` touches the output, so `figures/rans-mesh-snapshot-isometric.png` is a 0-byte file the pipeline reports as green. See ideas-log §4.15. Should fail loudly until it renders. |
| 12 | Visualize streamwise evolution of velocity profile | **Appears done** — `plot-bl-dns` → `figures/bl-profile-dns.pdf`. Confirm this is what was wanted. |
| 6 | Validate with streamwise force on the plate | **Open.** We report c_f pointwise but never integrate to a total force. A cheap, genuinely independent check on the whole chain. |
| 4 | Simulate custom turbulence model(s) with steady RANS | **Done** — `clip-k-gamma-sim`. |
| 1 | See these references | **Addressed** by [learning-closures-lit-review.md](learning-closures-lit-review.md) §9, which covers all four links. |

---

## 4. Carried over from the ideas log

Open threads that are not gym work. See [ideas-log.md](ideas-log.md) for the
full context on each.

- **§7.1 momentum-equation term library.** Not yet tried; the review argues it
  should be run with out-of-sample transfer as the primary metric and with
  dominant-balance regions identified first \citep{Callaham2021}.
- **§7.2 Bayesian coefficients.** The prize is the posterior width, not the
  optimum. Per-case posteriors across the gym give a calibrated
  transferability measure. Note \citet{Edeling2014} is close prior art and
  must be cited before we claim novelty.
- **§4.15** mesh snapshot stub (= issue #14).
- **§5** transition *length* rather than onset; `Cgam` railed at its bound.
- **§5** reformulate γ as coherence, given §1.4.
- **§6** the evolutionary structure search stagnated and was under-powered; a
  properly powered run is still owed.
- **§4.12** split the OpenFOAM model library per closure so editing one model
  does not invalidate the other's results. This becomes more pressing once
  Tier 2 has several models.

---

## 5. Paper

Decision on 2026-08-28: **extend the current paper** rather than split it.

- [ ] Add an out-of-sample generality section over the new cases.
- [ ] Present the harness as the tool that makes the paper's own recommended
      diagnostics routine — the current conclusion already recommends
      out-of-sample transfer, a null baseline and seed repetition.
- [ ] Add the identifiability row to the diagnostics table once §7.2 exists.
- [ ] **Soften the novelty claim on non-transferability.** \citet{Edeling2014}
      showed coefficient variability across boundary-layer flows with
      posteriors in 2014. What is ours is the three-independent-routes
      demonstration and the harness.
- [ ] The abstract is still the DRAFT placeholder. Per `AGENTS.md` the paper is
      to be written by a human; the numbers and skeleton are the agent's job.
