# Learning fluid conservation laws and closures: a review

> **Authorship.** Written by Claude (Anthropic's Claude Code), prompted by Pete
> Bachant. Citations are in `references.bib`. Where a claim is this project's
> own measurement rather than the literature's, it is marked **[ours]** and
> points at the script that produced it.

This is not a neutral survey. It is organised around the questions this project
actually has to answer: why our closure fits one case and not another, whether
a term library on the momentum equation is worth building
([ideas-log §7.1](ideas-log.md)), whether a Bayesian treatment buys anything
([ideas-log §7.2](ideas-log.md)), and whether a plug-in benchmarking harness is
a real gap or a solved problem.

---

## 1. The central split: inverse (a-priori) versus forward (a-posteriori)

Almost every methodological disagreement in this field reduces to *where the
loss is evaluated*.

**A-priori / inverse.** Compute the closure target from DNS, fit a model to it
offline by regression. Cheap, convex-ish, and every candidate can be screened
without a solve.

**A-posteriori / forward.** Put the candidate in a solver, run it, and score the
solution. Expensive, non-convex, and the only setting in which the number you
report is the number a user will get.

The literature's hard-won conclusion is that **the two do not agree, and
a-priori accuracy does not imply a-posteriori accuracy**. \cite{Duraisamy2021}
makes model-consistent training the organising principle of the whole review:
generalisable models require the training procedure to be consistent with the
model the closure will live inside. \cite{Zhao2020} operationalise this as
"CFD-driven" training — GEP with the forward solve inside the optimisation
loop, precisely because their earlier a-priori-trained expressions did not
survive deployment. \cite{Um2020} make the same argument outside turbulence via
differentiable solvers: a closure trained without the solver is trained on the
wrong input distribution, because at deployment it sees its own errors fed
back.

**[ours]** We have this disagreement in-house and it is stark.
`scripts/regress-pde-terms.py` (a-priori) reaches R² = 0.953 on the stress
target, while the a-posteriori structure search in `pypkg/search.py`
selects different terms. The a-priori fit also collapses out of sample
(R² = −472 predicting Jiménez). So we are not choosing between two roughly
equivalent routes — we have independently reproduced the field's main
methodological finding.

The practical consequence for us: **the gym must score a-posteriori.** A
harness that ships feature/label tables and scores regression error would be
reproducing the diagnostic the literature has already found insufficient.

---

## 2. What form the learned object takes

Five families, roughly in order of how much structure they impose.

### 2.1 Correction fields on an existing model (FIML)
\cite{ParishDuraisamy2016} invert for a spatially varying multiplier β(x) on a
term in an existing transport equation, then regress β against local features
to get a deployable model. The strength is that the base model's asymptotic
behaviour is retained by construction; the weakness is that the answer is only
as good as the base model's structure, and β is not unique — many β fields
reproduce the same observable.

Our `sim/evolve_pde_structure.py` term multipliers are a coarse, global version
of this: constant multipliers rather than fields. **[ours]** ideas-log §2.2
records that this did not produce structural discovery, which is the expected
outcome — a multiplier cannot add a term that is not already present.

### 2.2 Invariant neural representations
\cite{Ling2016} is the canonical result: predict the Reynolds-stress anisotropy
on an integrity basis of tensor invariants, so Galilean invariance and frame
properties hold architecturally rather than being learned. Embedding the
invariance measurably beat an unconstrained MLP. This established the field's
dominant prior — *constrain the hypothesis space with physics, do not hope the
network infers it*.

### 2.3 Sparse and symbolic regression
The line most relevant to us.
- \cite{Brunton2016} (SINDy) and \cite{Rudy2017} (PDE-FIND): build a large
  library of candidate terms, regress sparsely, keep few terms. PDE-FIND
  recovered Navier–Stokes and Kuramoto–Sivashinsky from data.
- \cite{Schmelzer2020} (SpaRTA): the same idea specialised to RANS. Tensor
  polynomials from a candidate library, elastic-net sparsity, validated on
  periodic hills, converging–diverging channel and curved backward-facing step,
  with a genuine out-of-sample test (Re = 10595 → 37000).
- \cite{Weatheritt2016} and \cite{Zhao2020}: gene expression programming, with
  the forward solver in the loop in the later work.
- \cite{BeethamCapecelatro2020}: sparse regression with form invariance
  embedded, i.e. §2.2's prior applied to §2.3's method.

- \cite{Callaham2021}: not a closure method, but the most directly useful
  neighbour for §7.1. They cluster in *equation space* to find which terms
  actually balance in each region of a flow, automatically flagging negligible
  ones. Demonstrated on turbulence among other fields.

**This is the family our §7.1 idea belongs to**, and the important detail is
that *every successful member constrains the library*. SpaRTA regresses on a
tensor polynomial basis; Beetham and Capecelatro embed form invariance; our own
`pypkg/grammar.py` enforces dimensional consistency and calls it a design
virtue. The proposal to build a momentum-equation library "regardless of
dimensions" is deliberately the opposite, and the review's honest read is that
this makes it a *stronger negative control* rather than a likely winner. That
is still worth running, but it should be framed and reported that way.

\cite{Callaham2021} suggests the constructive version. Our coefficients fail to
transfer between the pre-transitional and turbulent regions
(`coeff_disagreement = 1.0`) **[ours]**, which is precisely a statement that
*different terms dominate in different regions*. Fitting one global coefficient
set across a flow containing several distinct balances is guaranteed to produce
a compromise that fits none of them. Identifying the balance regimes first, and
only then regressing within each, is the version of §7.1 that could plausibly
work rather than merely fail informatively.

### 2.4 Full neural closures and differentiable physics
Neural networks replacing the closure entirely, trained through the solver
\cite{Um2020}. Best accuracy where it works; worst interpretability, worst
extrapolation guarantees, and it cannot be shipped as coefficients in a paper.

### 2.5 PINNs
\cite{Raissi2019} impose the PDE residual in the loss. Widely used for
inference and flow reconstruction from sparse data. As a route to a *reusable
closure* the record is weaker — a PINN is usually fitted per problem, which is
the exact failure mode this project is documenting.

---

## 3. Learning conservation laws proper — and why it transfers less than it looks

There is a separate, mathematically cleaner literature on discovering conserved
quantities: Hamiltonian neural networks \cite{Greydanus2019}, AI Poincaré
\cite{LiuTegmark2021}, symbolic regression for physical law
\cite{UdrescuTegmark2020}, and symbolic distillation of learned models
\cite{Cranmer2020}.

**It is important not to over-read this for our purposes.** These methods
recover *invariants of a closed dynamical system*. A RANS closure is not a
conservation law — it is a constitutive relation, and Reynolds averaging is
precisely the step that destroys the closed conservation structure the
Hamiltonian/Noether machinery relies on. The unclosed terms exist because
information was discarded. No amount of invariant-discovery on averaged fields
recovers it.

What *does* carry over is weaker but real:
1. **Structure-preserving parameterisation.** Build the model so the
   conservation and realisability properties hold identically, then learn only
   what is left. This is §2.2's lesson arriving from a different direction.
2. **Symbolic distillation as a reporting discipline** \cite{Cranmer2020}: fit
   flexibly, then compress to a closed form you can print, cite and criticise.
3. **The honest negative framing.** If a "law" you have discovered has
   coefficients that must be refitted per case, it is a fit. This is our
   paper's thesis, and the conservation-law literature supplies the standard
   against which the word "law" should be used.

---

## 4. Generalisation: what has actually been shown to fail

This is where the field's published record is thinner than its published
enthusiasm, and where our contribution sits.

- \cite{Duraisamy2021} states the consensus plainly: benefits typically do not
  extend to configurations significantly different from the training set.
- The transition-modelling branch shows the same pattern. Data-driven
  augmentations of γ–Re_θ and k–ω–γ–A_r style models
  \citep{LangtryMenter2009, Menter2006} improve the trained case; the
  generalisation evidence is much weaker than the in-sample evidence.
- \cite{XiaoCinnella2019} review model-form uncertainty and make the same point
  from the UQ side: structural error dominates, and calibrating coefficients
  moves error around rather than removing it.

**The diagnostics that actually discriminate**, collected from the above and
from our own experience **[ours]**:

| diagnostic | what it catches | our instance |
|---|---|---|
| out-of-sample transfer to an independent DNS | curve fits | R² = −472 on Jiménez |
| trivial-coordinate / null baseline | claimed collapses that are kinematic | ideas-log §4.14 area |
| repeat the fit under different seeds | structure selection inside search noise | repeat moves score by 1.1, structures differ by 0.04 |
| leave-one-region-out refit | coefficients that are locally, not globally, valid | upstream/downstream `coeff_disagreement = 1.0` |
| posterior width per coefficient | coefficients the data never constrained | **not yet done** (§7.2) |

The last row is the gap, and §5 says why it is worth closing.

---

## 5. Uncertainty and identifiability

\cite{OliverMoser2011} and \cite{Edeling2014} treat turbulence-model
coefficients as random variables and produce posteriors conditioned on data.
\cite{Edeling2014} is the directly relevant one: Bayesian estimates of k–ε
coefficient variability *across a set of boundary-layer flows*, finding that no
single coefficient set is supported by all of them. That is our paper's claim,
made a decade earlier, with error bars.

Two things follow.

1. **We should cite this as prior art rather than present non-transferability
   as new.** What is new in our work is not that coefficients vary — it is the
   discovery-loop context (structure search, a-priori regression and an
   integral budget all agreeing in-sample and all failing out of sample), and
   the harness that makes the check routine.
2. **The identifiability framing is the upgrade available to us.** The
   recurring finding in this literature is that posteriors are well-informed
   for some configurations and barely constrained for others. Reporting a
   credible interval per coefficient per case, and the *overlap* of those
   intervals across cases, converts our qualitative claim into a measurement.
   Cost is the obstacle — MCMC on RANS is quoted at ~10⁵ solves — which is
   exactly why our fast parabolic solver matters, and why the BO/surrogate
   route in §7.2 is the practical version.

---

## 6. Benchmarks and datasets: the actual gap

What exists:

- \cite{McConkey2021}: the reference open dataset. 29 cases per turbulence
  model across periodic hills, square duct, parametric bumps,
  converging–diverging channel and curved backward-facing step, ~896k points,
  RANS features paired with DNS/LES labels. Purpose-built for ML closure work.
- \cite{RumseyTMR}: NASA's Turbulence Modeling Resource — grids, reference
  solutions and validation data for flat plate, bump-in-channel, NACA 0012
  \citep{Ladson1988}, backward-facing step. The de facto verification standard
  for *implementations*.
- Canonical DNS: \cite{LeeMoser2015} channel to Re_τ = 5200,
  \cite{Sillero2013} ZPG boundary layer to Re_θ ≈ 6500, \cite{Bobke2017} APG
  boundary layers, \cite{Breuer2009} periodic hills, \cite{SchlatterOrlu2010}
  ZPG assessment.
- Recent ML benchmarking frameworks (automotive aerodynamics, aerodynamic shape
  optimisation, neural surrogates) are configuration-driven and reproducible —
  but they benchmark **surrogates that predict flow fields**, not closures that
  run inside a solver.

**What does not exist, as far as this review found:** an open harness where you
supply a *closure* — a term in a transport equation — and it is solved and
scored a-posteriori across a standard suite of cases, with in-sample and
out-of-sample results separated by construction.

The distinction from \cite{McConkey2021} is precise and worth stating in the
paper: a curated dataset supports **a-priori** evaluation, which is the
diagnostic \cite{Duraisamy2021} and \cite{Zhao2020} both argue is insufficient.
A harness that runs the closure closes that gap. This is a defensible claim of
novelty for the gym, and it is a claim about *tooling*, which is the kind that
survives.

Caveat: absence of evidence. This section is based on targeted search, not an
exhaustive one, and the framework space moves quickly. Before making the
novelty claim in the paper, it should be re-checked against
`turbmodels.larc.nasa.gov`, recent JFM/PRFluids software papers, and whatever
has appeared since.

---

## 7. Where this project sits

| the field's finding | our status |
|---|---|
| a-priori ≠ a-posteriori | independently reproduced **[ours]** |
| constrain the hypothesis space with invariance/dimensions | done in `grammar.py`; §7.1 proposes deliberately relaxing it as a control |
| coefficients vary across flows | reproduced; \cite{Edeling2014} got there first with error bars |
| generalisation is the unsolved problem | our contribution: make checking it routine |
| curated datasets exist for a-priori work | our contribution: an a-posteriori harness |

**The honest summary.** Our negative result is not novel as a phenomenon — it
is well-supported prior art. What is novel, and worth building, is (i) three
independent discovery routes failing *the same way on the same flow*, which is
a sharper demonstration than any one of them, and (ii) the harness. The
literature says generalisation is the problem; almost nobody ships the tool
that makes testing it the default. That is the contribution to defend.

---

## 8. Reading list, priority order

1. \cite{Duraisamy2021} — the frame for everything above. Read first.
2. \cite{Edeling2014} — closest prior art to our thesis. Read before finalising
   the paper's novelty claims.
3. \cite{Schmelzer2020} — the template for doing §7.1 well.
4. \cite{Zhao2020} — forward-solver-in-the-loop training, done properly.
5. \cite{McConkey2021} — what the gym must differentiate itself from.
6. \cite{Ling2016} — the invariance prior, still the field's most-cited result.
7. \cite{Rudy2017} — the mechanics of library regression on a PDE.

---

## 9. Also flagged in the repository's own issue tracker (GitHub #1, #3)

Recorded here so the threads are not lost:

- `xiaoh/turbulence-modeling-PIML` — code for \cite{WangWuXiao2017}, the
  physics-informed reconstruction of Reynolds-stress discrepancies from DNS.
  Worth reading for its feature set and its treatment of the discrepancy field,
  which is the same object FIML calls $\beta$ (§2.1).
- arXiv:2001.10019 is \cite{Callaham2021} — see §2.3.
- `LorenzoPiu/aPrioriDNS` — tooling for a-priori DNS analysis. Prior art for
  the *a-priori* half of a harness, and therefore something the gym should cite
  rather than duplicate.
- The Springer link in issue #1 is \cite{Schmelzer2020} — see §2.3.
- The McConkey Kaggle dataset noted in issue #3 is \cite{McConkey2021}; see §6.
  Issue #3 also suggested PySINDy, the reference implementation for
  \cite{Brunton2016} and \cite{Rudy2017}, and the obvious tool for §7.1.
