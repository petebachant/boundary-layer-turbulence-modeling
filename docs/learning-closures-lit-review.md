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
generalizable models require the training procedure to be consistent with the
model the closure will live inside. \cite{Zhao2020} operationalise this as
"CFD-driven" training — GEP with the forward solve inside the optimization
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
behavior is retained by construction; the weakness is that the answer is only
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
  neighbor for §7.1. They cluster in *equation space* to find which terms
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
1. **Structure-preserving parameterization.** Build the model so the
   conservation and realisability properties hold identically, then learn only
   what is left. This is §2.2's lesson arriving from a different direction.
2. **Symbolic distillation as a reporting discipline** \cite{Cranmer2020}: fit
   flexibly, then compress to a closed form you can print, cite and criticise.
3. **The honest negative framing.** If a "law" you have discovered has
   coefficients that must be refitted per case, it is a fit. This is our
   paper's thesis, and the conservation-law literature supplies the standard
   against which the word "law" should be used.

---

## 4. generalization: what has actually been shown to fail

This is where the field's published record is thinner than its published
enthusiasm, and where our contribution sits.

- \cite{Duraisamy2021} states the consensus plainly: benefits typically do not
  extend to configurations significantly different from the training set.
- The transition-modeling branch shows the same pattern. Data-driven
  augmentations of γ–Re_θ and k–ω–γ–A_r style models
  \citep{LangtryMenter2009, Menter2006} improve the trained case; the
  generalization evidence is much weaker than the in-sample evidence.
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
  optimization, neural surrogates) are configuration-driven and reproducible —
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
| generalization is the unsolved problem | our contribution: make checking it routine |
| curated datasets exist for a-priori work | our contribution: an a-posteriori harness |

**The honest summary.** Our negative result is not novel as a phenomenon — it
is well-supported prior art. What is novel, and worth building, is (i) three
independent discovery routes failing *the same way on the same flow*, which is
a sharper demonstration than any one of them, and (ii) the harness. The
literature says generalization is the problem; almost nobody ships the tool
that makes testing it the default. That is the contribution to defend.

---

## 7b. Deterministic RANS models fitted to DNS: the older lineage

> Added 2026-08-29 by Claude (Anthropic's Claude Code), prompted by Pete
> Bachant, after a broader search for studies in which a deterministic (not
> machine-learned) RANS closure was sought to fit DNS data, and for
> perspective on the black-box end of the field. Sections 7b--7d are new;
> the rest of the file predates them.

The question this project asks --- can a closure with a fixed algebraic
form be made to reproduce a DNS, and does what it learns transfer --- is
older than machine learning in turbulence, and the older answers are worth
having in front of us because they were reached with the same data and
much smaller hypothesis spaces.

**DNS budgets as the calibration target (1988--1993).** The first DNS of
channel flow was followed almost immediately by the term-by-term
Reynolds-stress and dissipation budgets \citep{MansourKimMoin1988}, and the
budgets were then used to fit closures term by term rather than to fit the
mean profile. \citet{RodiMansour1993} is the canonical case: $C_\mu$ and the
damping function $f_\mu$ are evaluated directly from the DNS of channel and
boundary-layer flow at two Reynolds numbers each, the $\epsilon$-budget is
computed and its terms scaled, and a new $f_\mu$ and $\epsilon$-source model
are fitted to the data. Two things from that paper carry over. First, they
found the *near-wall* form of every existing $f_\mu$ wrong against the DNS,
and fitting one to the data fixed the channel and not much else --- the same
in-sample-only result this project reproduced twenty-five years later with
a larger library. Second, they fitted a *budget*, not a profile: the
unclosed terms of the transport equation were the regression target, which
is closer to what `regress-pde-terms.py` does than to what
`discover-closure.py` does, and they too found the ranking of candidate
forms to depend on which term was targeted.

**Elliptic relaxation as the structural answer (1991).** Durbin's response
to the same DNS was not a better damping function but a different
structure: the $v^2$--$f$ model \citep{Durbin1991} replaces $f_\mu$ with a
transported wall-normal stress and an elliptic equation for the
pressure-strain, so that near-wall behavior comes from an equation rather
than from a fitted function of $y^+$. That is the strongest precedent for
this project's own conclusion that a *threshold* (a rectifier) rather than
a smooth fitted term is what the laminar state needs: in both cases the
data said "the form is wrong", not "the coefficients are wrong".
\citet{Kalitzin2005} later used DNS to characterize the near-wall behavior
of every common model and to build wall functions from the models' own
solutions rather than from a fit, which is the DNS-as-diagnostic role our
`analyze-flow-structure` stage plays.

**Scale-determining equations and the $\omega$ reassessment.** The choice
of $\omega$ over $\epsilon$ and the log-layer constraint
$\alpha = \beta/\beta^* - \kappa^2/(\sigma_\omega\sqrt{\beta^*})$ used in
`ClipKOmegaGamma` come from \citet{Wilcox1988}, whose coefficients were set
by analytical constraints (log law, decaying isotropic turbulence,
free-shear spreading rates) and *then* checked against DNS --- the
opposite order from a data fit. Spalart--Allmaras \citep{SpalartAllmaras1992}
was built the same way, with the calibration flows named and the DNS used
to reject rather than to fit. Spalart's own account of why this ordering
matters \citep{Spalart2015} --- that models are built from a small set of
canonical flows and a large set of constraints, and that fitting to one
flow buys nothing --- is the clearest statement of the position this
project's negative result supports.

**Bypass transition specifically.** The DNS lineage for our exact problem
runs from \citet{JacobsDurbin2001}, whose simulation established the
streak--lift-up--breakdown picture, through the review of
\citet{DurbinWu2007}, to the JHTDB dataset of \citet{Zaki2013} used here.
The deterministic RANS responses to it are three, and each is a different
bet on the closure variable. \citet{Lardeau2004} fitted a nonlinear
eddy-viscosity closure with low-Reynolds-number damping to the DNS of
bypass transition and reported that the pre-transitional streak energy
(their "laminar fluctuations") had to be represented explicitly for the
onset location to be right --- the argument for `ks` in `ClipTwoReservoir`.
\citet{Walters2008}, cited earlier, made that a transported laminar
kinetic energy. \citet{Ge2014}, building on the $\gamma$--$Re_\theta$
approach of \citet{LangtryMenter2009}, transported an intermittency
function with an onset criterion in the local vorticity Reynolds number
$Re_v$ and calibrated its threshold against DNS and experiment ---
structurally the closest published relative of our clip closure, and the
reason the fitted $\Lambda_c$ landing near $440$ is unsurprising rather than
a discovery. What none of the three did, and what this project's benchmark
now does, is score the calibrated model on flows other than the one it was
tuned to.

**Bayesian calibration to DNS (2011--2014).** The first careful attempts to
fit deterministic closures to DNS *with uncertainty* are
\citet{Cheung2011} and \citet{OliverMoser2011}, cited earlier, who
calibrated Spalart--Allmaras and $k$--$\epsilon$ coefficients against DNS
channel data and reported posteriors rather than optima; \citet{Ray2014}
did the same for $k$--$\epsilon$ on a jet in crossflow; and
\citet{Edeling2014} showed the posteriors move between boundary-layer
flows, which is the non-transferability result our leaderboard measures by
a different route. The lesson these papers drew --- that the *model-form
error* dominates coefficient uncertainty --- is the one the field then spent
a decade trying to address with learned corrections.

## 7c. The black-box end: what a decade of learned closures has shown

The reviews \citep{Duraisamy2019, Brunton2020, VinuesaBrunton2022,
SandbergZhao2022} agree on the taxonomy already used in \S2. The
perspective worth adding is what the *a-posteriori* record looks like once
one asks the transfer question this project asks.

**Tensor-basis and random-forest closures** \citep{Ling2016,
KaandorpDwight2020} learn the anisotropy as a function of local invariants
and are, by construction, Galilean- and frame-invariant. In sample they
reproduce DNS stresses well; propagated through a solver they are
ill-conditioned \citep{WuXiaoPaterson2018}, and their out-of-sample
behavior is rarely reported at all, since the standard datasets
\citep{McConkey2021} are built for a-priori scoring.

**Field inversion and machine learning** \citep{ParishDuraisamy2016,
DuraisamyZhangSingh2015} is the a-posteriori route: infer a correction
field by adjoint, then learn it as a function of local features. Applied to
transition, \citet{DuraisamyZhangSingh2015} and \citet{YangXiao2020} learned
corrections to intermittency-transport models from DNS and improved the
calibration cases. The generalization record is now explicit.
\citet{Nishi2024} trained on separated airfoil flows and found the
augmented model *worse than the baseline* on a different class of
separated flow (the NASA hump), recovering only by localizing the
correction with a sensor so it switches off where the training features
are not seen. \citet{RumseyColemanWang2022}, at NASA, set out to find *any*
data-driven improvement to Spalart--Allmaras for separated flows that would
not degrade the baseline elsewhere, trained across a wide array of cases,
and concluded that with the constraints of universality and no-harm
imposed, the improvements available were small. \citet{Volpiani2021}
reached a similar place for massively separated flows.

**Symbolic and sparse regression**, the branch this project belongs to,
has moved from single-case fits \citep{Weatheritt2016, Schmelzer2020} to
multi-case a-posteriori training precisely because the single-case models
did not transfer: \citet{Waschkowski2022} and \citet{Fang2023} train gene
expression programs against several flows at once in the solver, and report
that the resulting models are more general at the cost of being less
accurate on any one flow --- the transfer penalty made into an objective.
\citet{WuZhang2023} and \citet{WuZhangZhang2025} do the same with a
sparse-regression correction to SST, conditioning the field inversion so
that the correction is expressible in a small basis, and
\citet{ShanZhang2025} restrict the learned term to adverse-pressure-gradient
physics with a fixed functional form, which is the "hypothesis space
first" position. \citet{Hu2024} return to the 1993 problem ---
low-Reynolds-number corrections for two-equation models --- with modern
DNS and data-guided fitting, and their result reads much like Rodi and
Mansour's. The newest work keeps adding structure rather than removing it:
nonlocal features \citep{WuZhang2025}, an equation-learner embedded in the
inversion so the correction is symbolic from the start \citep{Li2026},
language-model transfer of symbolic models between geometries
\citep{Reissmann2025}, multi-objective training toward a single unified
model \citep{Liu2025}, and mixture-of-experts routing so that adding a
flow class does not degrade the ones already learned \citep{Ji2026}. Read
together, these are the field converging on the two design decisions the
RANS gym encodes: score in the solver, and separate in-sample from
out-of-sample by construction.

**The modeler's rebuttal.** \citet{Spalart2023} is the sharpest statement
of why purely data-driven closures keep failing to generalize, and it is
worth taking seriously because its author's models are the ones the
learned corrections are applied to. His argument is that several properties
a usable model must have --- behavior at the edge of the turbulent region,
where the model must go to zero in a specific way; free-stream sensitivity;
Galilean invariance; realizability; the log law --- are analytical
properties of the differential equations, out of reach of a regression on
fields, and that most published learned models violate at least one. He
separates the *mission* (which flows, which quantities) from hard and soft
*requirements*, and proposes that machine learning be confined to the space
of models that satisfy the requirements by construction. This project's
own findings sit comfortably inside that frame: the edge-of-turbulent-region
behavior is exactly where our mixing-length closures fail without a wall
(\S2.3 of `shear-layer-vortex-lessons.md`), and the log-layer constraint is
the one coefficient relation we chose to impose rather than fit.

## 7d. What this changes for the project

- The claim that a fitted closure does not transfer is not new; it was
  known in 1993 for $f_\mu$ and in 2014 for coefficient posteriors. What is
  new here is the demonstration by three independent routes plus a harness
  that makes the test routine. The paper's novelty statement should say so
  and cite \citet{RodiMansour1993} and \citet{Edeling2014}.
- The rectifier-versus-smooth-term finding has a structural precedent in
  \citet{Durbin1991}: both say the data reject the *form*. That is a
  stronger frame than "we found a better term".
- \citet{Ge2014} should be cited beside \citet{LangtryMenter2009} wherever
  the $Re_v$ threshold is discussed, and the near-$440$ result presented as
  agreement with prior calibration rather than discovery.
- The multi-case training of \citet{Waschkowski2022} and \citet{Fang2023}
  is the obvious next experiment for our closure: fit against several gym
  cases at once and measure what in-sample accuracy it costs.
- Spalart's hard requirements \citep{Spalart2023} are a checklist the gym
  could score deterministically --- decaying free-stream turbulence, the
  log law, edge-of-turbulent-region behavior, Galilean invariance --- as a
  cheap tier below any DNS case. That is the Falkner--Skan sanity tier of
  the roadmap, generalized.

## 7e. The universal RANS model: what has been claimed, achieved and conceded

> Added 2026-08-30 by Claude (Anthropic's Claude Code), prompted by Pete
> Bachant, when the project's goal was restated as one coefficient set
> across every reference case.

**The goal is old and its limits were conceded early.** The CFD Vision
2030 study \citep{Slotnick2014} named "the inability of RANS to predict
separated and transitional flows" as the single largest gap in aerospace
CFD and did not expect a universal RANS closure to close it; it asked for
hybrid RANS–LES and for better *use* of RANS instead. Spalart's
philosophies paper \citep{Spalart2015} makes the modeler's version of the
same concession: a model is a compromise across a chosen set of flows, the
choice is a judgement, and "universality" is not a property a closure can
have, only a property of how honestly its calibration set is reported. The
literature since has mostly been a search for how large that set can be
made before the compromise costs more than it buys.

**Two industrial answers.** The first is to stop pretending one coefficient
set serves every flow and expose the trade-off as knobs: GEKO
\citep{Menter2025}, the generalized $k$–$\omega$ model in a major
commercial code, has coefficients for near-wall behavior, separation,
mixing and jet spreading that are *designed* to be independent of one
another and to be tuned per application within stated ranges without
breaking the wall-bounded calibration. That is a universal *framework*
with per-case coefficients, the exact opposite of this project's target,
and it is where the practitioner community landed after thirty years of
$k$–$\epsilon$/$k$–$\omega$/SST. Bayesian optimization of those knobs per
case \citep{McConkey2024} is now routine, and constrained re-calibration
\citep{Bin2024} — re-fitting some coefficients while holding the log law,
decay and free-shear constraints fixed — is the disciplined way to do it.
The second answer is the Reynolds-stress model: seven equations, more
physics, and in the vortex study of \citet{Sansica2026} the only model
that behaved, at the cost of convergence and industrial uptake that the
Vision 2030 report already noted.

**What machine learning has added, in its own words.** The 2022 NASA
symposium on turbulence modeling asked developers of learned models to
run them on unseen aerospace flows, and its consensus was that improvement
on training-like cases did not carry to complex unseen ones
\citep{RumseyColemanWang2022}. \citet{MandlerWeigand2024} set out to measure
the *limits* of generalization for learned closures and found them at the
edge of the training distribution of the input features, which is a
statement about feature coverage rather than about physics; the practical
responses — localize the correction with a sensor \citep{Nishi2024}, blend
several regime-specific models with a learned gate \citep{Oulghelou2025},
route with a mixture of experts \citep{Ji2026}, train on many cases at
once with a multi-objective loss \citep{Liu2025, Fang2023} — all give up
on one coefficient set and manage the compromise instead. Girimaji's
foundational-physics perspective \citep{Girimaji2024} explains why the
compromise is not an artifact of poor training: the closure problem has
different dominant physics in different regimes (rapid distortion,
equilibrium cascade, near-wall), a single-point local model cannot carry
enough state to tell them apart, and any regression on local features
inherits that ceiling. Spalart's requirements paper \citep{Spalart2023}
reaches the same place from the other direction.

**What this project can claim in that landscape.** Not a universal model —
the leaderboard says the opposite, and so does everyone above. What it can
claim is a *measurement*: one coefficient set, held fixed, scored on eight
flows in five geometries under a rule that separates calibration from
test by construction, with the transfer penalty as the reported number.
That number is what GEKO's per-case knobs hide, what the NASA symposium
found qualitatively, and what \citet{MandlerWeigand2024} measured for
learned closures on one flow class. The Closure Challenge
\citep{McConkey2026} measures the same thing for field predictors on a
different case set; with its cases added here (roadmap §2.3), the two
benchmarks overlap enough to compare a closure and a surrogate on the same
flow, which no published result does.

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


Added 2026-08-29, in priority order for the questions this project asks:

1. \citet{Spalart2023} --- the requirements any closure must meet, and why
   regression on fields cannot supply them.
2. \citet{RodiMansour1993} --- the original deterministic fit to DNS, with
   the original in-sample-only result.
3. \citet{RumseyColemanWang2022} and \citet{Nishi2024} --- what happens when
   learned corrections are held to no-harm and out-of-class tests.
4. \citet{Fang2023} and \citet{Waschkowski2022} --- multi-case training,
   the experiment our benchmark makes possible.
5. \citet{Ge2014} and \citet{Lardeau2004} --- the deterministic bypass
   transition models closest to ours.
6. \citet{SandbergZhao2022} --- the review that maps the ML branch by
   physical phenomenon rather than by method.

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
