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
- [x] **Temporally evolving mixing layer, pre-roll-up** \citep{Lusher2026}.
      **Done 2026-08-29.** The first case with no wall. Built on the only
      public form of the DNS — the JAXA DNS database's file of peak and
      integral time histories \citep{JAXADNSDatabase} — so it scores the
      momentum-thickness growth and the peak stress and TKE histories over
      t̂ ∈ [0.14, 0.40], the window in which the DNS is still 1-D. Seeded
      with Part II's own recipe \citep{Sansica2026}. Converged in domain
      height, grid, time step and frame speed for the closure that works;
      the vorticity thickness is reported but not scored because
      Launder–Sharma's peak gradient is a grid-dependent front at the layer
      edge. Found that the three mixing-length clip closures produce
      *nothing* without a wall. See
      [shear-layer-vortex-lessons.md](shear-layer-vortex-lessons.md).
- [ ] **Flat-plate APG TBL** \citep{Bobke2017}. Fetchable (one 305 MB Google
      Drive .mat). Lower priority now the wing data is in hand: weaker
      pressure gradient, simpler geometry.
- [ ] **Falkner–Skan laminar family.** Self-generated ODE reference, no
      download. Cheap sanity tier: catches closures that pollute laminar
      regions or misread favorable/adverse `dUedx`. Not a headline result,
      but it is the fastest possible regression test.
- [ ] **Requirements tier** \citep{Spalart2023}: score deterministically,
      below any DNS case, the analytical properties a usable closure must
      have — decaying free-stream turbulence, the log law, behavior at the
      edge of the turbulent region, Galilean invariance. Generalizes the
      Falkner–Skan idea; see lit review §7c–7d.
- [ ] **Multi-case fitting** \citep{Waschkowski2022, Fang2023}: fit the
      clip closure against several gym cases at once and report what
      in-sample accuracy the generality costs. The benchmark makes this a
      one-script experiment.
- [ ] **Multi-fidelity optimization** (PB, 2026-08-30): treat the tiers as
      fidelities, not case sets — tier 1 for coefficient search and
      evolution, tier 2 as the benchmark of record. First step: fit the
      tier-1→tier-2 discrepancy from the closures that have run on the
      plate in both tiers (the paper already reports the ~3x c_f bias);
      then a BoTorch multi-fidelity BO (own environment; torch must not
      touch the compute lock) that spends OpenFOAM evaluations only where
      the tier-2 posterior could change the ranking. End state, with the
      one-definition spec (ideas-log §7.3–7.4): a user submits a
      functional form and bounds, tier 1 pre-optimizes, tier 2 confirms
      and enters the leaderboard.

### 2.3 Tier-2 cases
- [ ] Formalise the existing OpenFOAM runs (`sim/cases/*`) as registered
      Tier-2 cases rather than ad hoc stages.
- [ ] **The Closure Challenge's DNS cases** \citep{McConkey2026}, fetched
      2026-08-30 into `data/closure-challenge/` (DVC) pinned to a commit:
      complete OpenFOAM cases (mesh, BCs, schemes) with the DNS interpolated
      onto the RANS mesh — four parameterized periodic hills
      \citep{Xiao2020} and four square/rectangular ducts
      \citep{Vinuesa2014}, all from the challenge's test set so scores are
      comparable with its leaderboard. DNS only, per PB: the challenge's
      curved step and Re = 10,595 hill are LES and its hump is an
      experiment, and its k-ω SST baseline solutions are ML training data
      that nothing here trains on. Both flows are Tier 2 by mathematics:
      the hills separate, and the ducts' secondary flow is exactly what an
      eddy-viscosity model cannot produce — the sharpest available test of
      the momentum-term library (ideas-log §7.1). Plan: a
      `pypkg/cases/openfoam.py` `OpenFoamCase` that copies the case, writes
      `turbulenceProperties` for the closure's `openfoam_model`, runs
      `simpleFoam` in `blsim`, and scores U (and k) against the DNS at the
      challenge's evaluation points with declared targets; a
      `run-benchmark-openfoam` stage separate from the fast one.
      **Built and run 2026-08-30** (`pypkg/cases/openfoam.py`, stage
      `run-benchmark-openfoam`, 48 runs, ~2.5 h with the eight cases in
      parallel on one machine). Cost per run: hills ~10–20 min at 20,000
      iterations, ducts 20 s–80 min (AR = 14 is the expensive one for the
      least new information). A **routine subset** — one hill per α at the
      4048 resolution and the AR = 1 ducts at both Re_τ — would give the
      same rankings in a quarter of the time; the other four stay
      registered as an extended set. Not yet done: a `frozen`/manual
      extended stage so a code change does not re-run all 48.
- [ ] **NACA 0012** \citep{Ladson1988, RumseyTMR} — the airfoil case PB asked
      for. NASA TMR supplies grids and reference data, so the setup is not
      ours to invent.
- [ ] **Circular cylinder** — separation from a smooth surface, which is the
      canonical place RANS fails. Reference data and Reynolds number still to
      be chosen.
- [ ] **Shear-layer roll-up and vortex merger** \citep{Lusher2026,
      Sansica2026}. The Tier-2 continuation of the mixing-layer case: 2-D
      **unsteady** RANS on [−L_x/2, L_x/2] × [−2L_x, 2L_x], periodic in x,
      symmetry top and bottom, 800 × 720 baseline mesh, Δt̂ = 3.33e-4, to
      t̂ = 6, incompressible (the authors say so explicitly). Scored on the
      histories in the public file: max/min ω_z, min p, max |u|, max ν_t,
      integrated KE. Nine published RANS baselines to compare against. This
      would be the project's first URANS — all OpenFOAM stages so far are
      steady `simpleFoam` — so it needs a `pimpleFoam` setup, time-resolved
      sampling, and a Δt sensitivity study in place of residual convergence.
      It is also the case on which rotation/curvature sensitivity decides the
      result, which no closure here has (lessons §2.2).
- [ ] Cross-tier consistency check: a closure implemented in both tiers should
      agree on the cases both can run. Any disagreement is a bug in one of the
      two implementations, and the harness should say so loudly. Part II's
      two-solver agreement (FaSTAR vs FUN3D, < 1 %) is the model for this.

### 2.3b Interface changes suggested by the vortex study
From [shear-layer-vortex-lessons.md §3](shear-layer-vortex-lessons.md):
- [ ] Composable modifiers: `register_closure(name, base=..., modifiers=[...])`
      for production multipliers (rotation/curvature) and constitutive
      relations (QCR), so "SA-R95-QCR2000" is one line, not a subclass.
- [ ] A `stress()` method on `Closure` with a Boussinesq default, so
      non-linear EVMs and RSMs can register and the TAM diagnostic can be
      evaluated for any model.
- [ ] A `seed_from(k, nut, ...)` hook so closures with non-standard state
      (H, ks/ka) are seeded by the same rule as everyone else.
- [ ] A temporal `step(dt)` or separate convection speed in `advance`, so
      `TranslatingFrame` is no longer needed.
- [ ] Per-closure canonical description (bib key, TMR page) and a
      code-to-code verification case.
- [ ] **One closure definition for both tiers** (PB, 2026-08-30): a
      declarative spec (expressions parsed by SymPy) compiled to the Python
      `Closure` and to an OpenFOAM `RASModel`, with the paper's model
      equations generated from the same file. See ideas-log §7.3 for the
      design and why TeX is the view rather than the source. The same spec
      should accept a learned model as an expression node, exported to
      ONNX so one artifact runs in both tiers (`onnxruntime` in Python,
      the ONNX Runtime C API in a templated `RASModel`), with inputs
      restricted to the spec's invariants and outputs entering through a
      stabilizable form.
- [ ] **Predictors as a second kind of entrant** (PB, 2026-08-30): models
      that emit the whole field from a case description, scored on the
      same targets as closures with `calibrated_on` = training set, plus
      a momentum-residual diagnostic in place of convergence. Ideas-log
      §7.3, last paragraph.

### 2.4 A living benchmark: fork, don't extend in place

Raised by PB on 2026-08-29, and decided the same day. The gym scores every
registered closure against every registered case; the question was how a
group that *produces* a DNS, or has a closure to test, gets it in so the
leaderboard stays alive.

**Decision: derived projects, not a shared repository.** One repository
that everyone registers into would couple every group's paper to every
other group's edits, and a leaderboard that anyone can change is not a
result anyone can cite. Instead a group forks this project, registers its
closure or case, runs the pipeline, regenerates the paper with its own
results, and publishes the fork as its own project with a `derived_from`
record in `calkit.yaml` pointing here. The benchmark then grows as a tree
of derived projects, each paper is regenerated from the fork that produced
it, and the suite and scoring rule a result inherited are traceable
through the `derived_from` chain.

What that needs from the project (all cheap, all here):
- A contributed case or closure is a registration, not an edit to a stage:
  `RANS_GYM_PLUGINS` already does this for a module; a `cases/<name>/`
  drop-in directory with a `case.yaml` (`imported_from.doi`, family, tier,
  fidelity, targets) plus a reader into the common validation schema would
  remove the last package edit.
- `run-benchmark` as one stage instance per case via `iterate_over`, so a
  new dataset re-runs only its own case and the leaderboard stage merges.
- The paper regenerates from the fork's results by construction, since
  every number is injected; the case and model sections should be
  generated from the registries too, so a fork's paper lists its own.

PB's steer (2026-08-29): getting *this* project working well comes first;
what follows are ideas for others to use the repository later, not work
to do now.

What it needs from calkit:
- **Fork templating.** A fork is not a copy: the model under test changes,
  some prose is dropped, some inputs are the fork's own. Calkit should
  make those inputs explicit — a declared set of "fork points" (the
  closure registered as the subject, the cases in scope, the paper's
  generated sections versus its inherited prose) so that `calkit new
  project --derived-from` produces a project that is clearly its own
  rather than a duplicate, and so the derivation tree shows what each fork
  changed.
- `calkit.yaml` `derived_from` already exists; the hub should render the
  derivation tree and let a reader walk from a fork's leaderboard to the
  suite it inherited and to the upstream project's version at the fork
  point.
- `iterate_over` values discovered from a glob or from the `datasets`
  list rather than written by hand.
- A first-class "how to read this dataset" declaration on a dataset entry,
  so a drop-in dataset becomes validation data without code in our package.
- `calkit import dataset --doi` for data published as another project.

**Also needed from calkit** (PB, 2026-08-30; both posted on calkit PR
#1579):
- Credential-gated stages, so the JHTDB gradient fetch can be a stage: a
  `file`/`env-var` requirement with alternatives, requirements on
  non-system environments or on stages, a uv environment nested inside a
  system one, stages *skipped with outputs kept* when their requirements
  are unmet, and secrets kept out of every lock and log.
- Downloads as stages: a dataset with a `url` or `git` source and no
  `stage` is a hand-run download, and the reproducibility check should
  flag it. Every fetched dataset here now records both (`imported_from`
  says where from, `stage: fetch-dns-data` says how).

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

- **§7.1 momentum-equation term library.** Tried 2026-08-29 in the fast
  tier: `pypkg/momentum_library.py` (six Galilean-invariant force terms
  from y-derivatives of U, k, nu_t, dimensionless coefficients, on
  Launder–Sharma) fitted by Bayesian optimization (`pypkg/bayesopt.py`) of
  the a-posteriori gym score, once on the plate and once on four cases
  jointly (`fit-momentum-library*` stages). Both fitted closures sit in the
  leaderboard as `ls-momentum-library` and `ls-momentum-library-multi`.
  Still owed: x-derivative terms in OpenFOAM, and a posterior rather than a
  near-best interval per coefficient.
- **§7.2 Bayesian coefficients.** The prize is the posterior width, not the
  optimum. Per-case posteriors across the gym give a calibrated
  transferability measure. Note \citet{Edeling2014} is close prior art and
  must be cited before we claim novelty.
- **§6.1 `evolve-closure` does not reproduce.** Same machine, same
  environment, same seed: 47 structures / best 9.121 on one run, 50 / 6.718 on
  the next. Not an unseeded RNG — every stochastic step is seeded. It is the
  elite-selection step amplifying last-bit float differences. Fix by pinning
  worker and BLAS thread counts, breaking ranking ties on the candidate key,
  and reporting the archive rather than a single best. **Highest-priority bug:
  a pipeline stage that does not reproduce undermines the whole project's
  claim.**
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

- [x] **Provenance markers in the paper** (2026-08-29). The paper is built
      with `provenance: true`: every `\result[key]` and `\finding[key]` is
      a colored value logged with its stage; figures go through
      `\ckfigure`; `paper/main.provenance.json` lists every injection with
      its stage, inputs and hash. Drop the `provenance` package option for
      the submitted version. The Q&A appendix was removed on 2026-08-31
      (the questions stay in `calkit.yaml` for the hub and the checker; a
      paper carries facts, not the project's notebook), and the case,
      closure and score tables are now generated by
      `scripts/make-benchmark-tables.py` from the registries and the two
      benchmark files, so the manuscript covers every case and closure by
      construction.
- [ ] Add an out-of-sample generality section over the new cases.
- [x] **Restructure the paper to the project's current goal** (PB,
      2026-08-29): one coefficient set across all cases. New outline —
      reference cases; models tested and setup; the closure developed here;
      the benchmark; results (calibration case, leaderboard, what transfers,
      what does not); conclusions; generated Q&A appendix. Skeleton only,
      every number injected; provisional title. Prose is the human's.
- [ ] **Describe the gym's scoring method in the paper** (PB, 2026-08-29).
      A stub subsection with the definitions now sits in `paper/main.tex`
      (`sec:scoring`); the prose is the human's. It must state: the
      per-case targets and the normalized score (error / target, averaged
      over metrics, 1.0 = matches the data by inspection); that in-sample
      and out-of-sample are split by each closure's declared calibration
      case and never averaged together; the seeding rule (every closure
      seeded by the same case-supplied rule, the seeded station or
      transient excluded from scoring); that a non-converged or diverged
      run scores infinity rather than a number; and that peaks a model can
      make grid-dependent are reported but not scored.
- [ ] Present the harness as the tool that makes the paper's own recommended
      diagnostics routine — the current conclusion already recommends
      out-of-sample transfer, a null baseline and seed repetition.
- [ ] Add the identifiability row to the diagnostics table once §7.2 exists.
- [ ] **Soften the novelty claim on non-transferability.** \citet{Edeling2014}
      showed coefficient variability across boundary-layer flows with
      posteriors in 2014, and \citet{RodiMansour1993} found the same for a
      DNS-fitted damping function in 1993. What is ours is the
      three-independent-routes demonstration and the harness.
- [ ] Cite \citet{Ge2014} beside Langtry–Menter wherever the Re_v threshold
      appears, and present the fitted value near 440 as agreement with prior
      calibration, not discovery; frame the rectifier result as the data
      rejecting the *form*, with \citet{Durbin1991} as precedent (lit
      review §7b, §7d).
- [ ] The abstract is still the DRAFT placeholder. Per `AGENTS.md` the paper is
      to be written by a human; the numbers and skeleton are the agent's job.
