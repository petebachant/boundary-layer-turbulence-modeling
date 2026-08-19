# A clipping closure for transitional boundary layers

> **Note on authorship and provenance.** This document was written by Claude
> (Anthropic's Claude Code).
>
> **The central idea is Pete Bachant's, not Claude's.** The signal-clipping
> analogy — that a turbulent flow holds more energy than smooth waves can
> carry, so it "clips" and redistributes the excess into higher harmonics —
> came from Pete in the prompt that started this work, along with the goal of
> finding a closure built on a new conserved quantity. Claude's contribution
> is the part downstream of that: testing the analogy against the DNS, finding
> that it shows up as a saturating structure parameter and a two-reservoir
> energy split, and turning that into a closed set of PDEs.
>
> Every number quoted below was produced by the scripts in this repository and
> can be regenerated from the pipeline; nothing here is recalled from
> literature without being checked against our own DNS extraction.
> Interpretation and physical framing beyond Pete's original idea are Claude's
> and should be reviewed critically before being carried into the paper.

## 1. What we are trying to fix

The target case is the JHTDB `transition_bl` dataset: a flat-plate boundary
layer that starts laminar, undergoes **bypass transition** under freestream
turbulence, and ends fully turbulent. Extracting the mean field from
`data/jhtdb-transitional-bl/time-ave-profiles.h5` gives:

| x | c_f | H | Re_θ | state |
|---:|---:|---:|---:|---|
| 100 | 0.00241 | 3.70 | ~100 | laminar |
| 205 | 0.00200 | 2.77 | 216 | laminar, c_f at minimum |
| 310 | 0.00298 | 2.10 | ~330 | transition underway |
| 451 | 0.00480 | 1.57 | ~560 | c_f peak, transition ending |
| 941 | 0.00377 | 1.47 | 1450 | equilibrium turbulent |

The failure we care about is not a coefficient error. Running our own
parabolic solver over the same plate (`py_package/bl_solver.py`):

| model | c_f(100) | c_f(451) | c_f(941) | c_f rel. RMS |
|---|---:|---:|---:|---:|
| DNS | 0.00241 | 0.00480 | 0.00377 | — |
| laminar | 0.00225 | 0.00110 | 0.00075 | 0.67 |
| Launder–Sharma k–ε | 0.00487 | 0.00405 | 0.00360 | 0.56 |

Laminar is right early and never transitions. k–ε is turbulent from the
leading edge — **twice** the correct drag at x = 100 — and only becomes
reasonable far downstream. No choice of k–ε coefficients fixes this, and that
is a structural statement, not a tuning statement (see §5).

## 2. What the DNS actually says

Two measurements drive everything that follows.

### 2.1 The structure parameter saturates at a hard rail

Define a₁ = −⟨u'v'⟩ / 2k, integrated across the boundary layer:

| x | 100 | 205 | 310 | 380 | 451 | 556 | 731 | 941 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| a₁ | 0.017 | 0.031 | 0.069 | 0.098 | 0.118 | 0.132 | 0.138 | 0.136 |

a₁ climbs through transition and then **pins at 0.137 and stays there** for the
rest of the plate. It does not overshoot and drift back; it saturates. This is
the signature of a clipped quantity riding a rail, and it is what motivates the
whole model.

### 2.2 Energy is present long before it does any work

Pre-transition the boundary layer is *not* quiescent. At x = 205 the peak k is
7.2e-3, already 83 % of its equilibrium turbulent value (8.6e-3). But the
anisotropy is extreme:

| x | u'u'/2k | v'v'/2k | w'w'/2k | −u'v'/k |
|---:|---:|---:|---:|---:|
| 205 | 0.982 | 0.005 | 0.013 | 0.051 |
| 451 | 0.822 | 0.036 | 0.142 | 0.142 |
| 941 | 0.780 | 0.042 | 0.177 | 0.161 |

Essentially all of the pre-transitional energy is in streamwise fluctuations —
Klebanoff streaks. Since −⟨u'v'⟩ requires v', this energy carries almost no
momentum flux. Measured directly, the pre-transitional eddy viscosity is
**ν_t ≈ 0.44 ν**: negligible. The boundary layer is loud but laminar.

This is precisely Pete's signal-clipping analogy, made quantitative: energy
accumulates in the "fundamental" (streamwise streaks, the linear response to
freestream forcing) without distortion, until an amplitude threshold is
crossed; then it clips, and the excess is redistributed into "harmonics"
(v', w' — the three-dimensional breakdown) which are the components that
actually carry Reynolds shear stress.

### 2.3 Splitting the energy into two reservoirs

Using the saturated rail a₁∞ = 0.137, define an *active* (stress-bearing)
energy and a *streak* (inactive) energy:

    k_a ≡ −⟨u'v'⟩ / 2a₁∞ ,    k_s ≡ k − k_a

Integrated over the boundary layer this behaves exactly as the picture
demands:

| x | 100 | 205 | 310 | 380 | 451 | 556 | 661 | 801 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| k_a/k | 0.12 | 0.23 | 0.50 | 0.71 | 0.86 | 0.96 | 1.00 | 1.00 |
| ∫k_s | 7.9e-3 | 1.0e-2 | 1.4e-2 | 1.0e-2 | 5.1e-3 | 1.5e-3 | 1.1e-4 | ~0 |

The streak reservoir **fills** through the pre-transitional region (peaking at
x ≈ 310), then **drains to zero** as the active reservoir takes over, and
k_a/k saturates at 1.000. The split is not imposed; it falls out of the data.

### 2.4 An information-theoretic view: entropy, and a hysteresis

Pete also proposed reading transition as an entropy or information balance —
the inflow carries some information, the wall injects some, dissipation
removes some. That has a concrete and measurable form here. Define the
**component entropy** of the fluctuation energy partition,

    H = −Σ_i p_i ln p_i ,   p_i = ⟨u_i u_i⟩ / 2k

so H = ln3 ≈ 1.099 for isotropic turbulence and H = 0 for a perfectly ordered,
purely streamwise field. Measured from the DNS:

| x | 30 | 118 | 205 | **233** | 293 | 381 | 469 | 600 | 941 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H | 1.074 | 0.800 | 0.520 | **0.499** | 0.592 | 0.827 | 0.948 | 0.986 | 0.994 |

**H is not monotone.** It starts near its maximum — the incoming freestream
turbulence is nearly isotropic — then *falls* by a factor of two as the mean
shear organises the fluctuations into streaks, reaches a minimum at x = 233,
and rises back to near-isotropy once they break down. In other words the mean
shear does work to *reduce* the entropy of the fluctuation field, and
**transition onset coincides with the point of maximum order**. That is a
satisfying statement of the clipping idea in information terms: the flow can
only be compressed into so few degrees of freedom before it must break down.

The consequence for modelling is a warning. Fitting a₁ against H over the
whole plate fails badly (correlation 0.80, deviations up to 860 %), because
the system is **hysteretic** — the same H occurs once on the way down and once
on the way up, with very different stress. Split at the turning point and each
branch collapses almost perfectly:

| branch | fit | corr | max dev |
|---|---|---:|---:|
| ordering, x < 233 | a₁ = 0.0136·H^(−1.31) | 0.988 | 13 % |
| breakdown, x > 233 | a₁ = 0.1368·H^(1.61) | 0.996 | 15 % |

So H is an excellent *diagnostic* and a legitimate state variable, but it
cannot by itself close the model: a single-valued constitutive law a₁ = f(H)
does not exist. At least two variables are needed — which is exactly what the
activation γ supplies.

We implemented the entropy balance anyway as an independent model family
(`EntropyKOmegaH`), with H transported under a competition between an
entropy-reducing ordering term (∝ S, mean-flow work) and an entropy-producing
relaxation term (∝ ω, always ≥ 0 — a local second-law statement):

    DH/Dt = −C_ord·S·(H − H_floor) + C_mix·ω·(H_max − H) + diffusion

It reproduces the *shape* of H, including the minimum, from the competition
alone rather than by prescription. But it puts the minimum at x ≈ 33 instead
of 233 and scores 1.02 against the clipping model's 0.44. The reason is
structural and worth stating plainly: a smooth relaxation balance has no way
to *hold* the laminar state. Without a rectified threshold, the boundary layer
starts transitioning immediately. **Entropy is a good state variable; it is
not a substitute for the clip.** The promising synthesis is to keep H as the
state variable and give it a clipped source, which is what the structural
search in §7 is set up to find.

## 3. The model

Two transported quantities: total fluctuation energy k, and an **activation
fraction γ** — the fraction of k that bears Reynolds shear stress. γ is the
new quantity, and it is the one that clips.

    ν_t = C_μ γ √k ℓ                                  (activation gates the stress)
    ν_L = C_L √k_∞ ℓ_s                                (lift-up / streak forcing)

    Dk/Dt  = (ν_t + ν_L) S²  −  ε  +  ∂/∂y[(ν + ν_t/σ_k) ∂k/∂y]
    Dγ/Dt  = C_γ S (γ + γ₀)(1 − γ) · max(0, Λ − 1)^p
                              +  ∂/∂y[(ν + ν_t/σ_γ) ∂γ/∂y]

    Λ = Re_v / Λ_c ,   Re_v = y²|∂U/∂y| / ν
    ε = C_D γ k^{3/2}/ℓ  +  C_ν (1−γ) ν k / y²

Three ingredients matter, and each is forced by a measurement in §2:

1. **The eddy viscosity is gated by γ.** This is what lets the boundary layer
   carry large k while remaining laminar in the mean (§2.2). A conventional
   k–ε cannot do this: its ν_t is slaved to k.
2. **k is fed by a lift-up term built on the freestream amplitude √k_∞, not on
   k itself.** This is essential and was the single biggest structural
   discovery in this investigation. Our first attempt scaled streak production
   on the streak energy, which makes it self-amplifying and runs away (k came
   out 10× the DNS value). Scaling on the freestream forcing gives the
   algebraic, non-modal growth that transient-growth theory predicts, and it
   cannot trigger a spurious transition. Both ν_t and ν_L appear in the
   momentum equation, so mean-to-fluctuation energy transfer is exact.
3. **The γ source is rectified and saturating.** `max(0, Λ−1)` is the clip:
   identically zero below the rail, so a laminar boundary layer is a fixed
   point of the model rather than something it merely passes through slowly.
   `(γ+γ₀)(1−γ)` is logistic, so once triggered the transfer saturates at
   γ = 1 — reproducing the a₁ = 0.137 rail of §2.1 by construction.

The conservation content: k = k_a + k_s with k_a = γk, and the γ equation is a
statement about how energy is *partitioned* between a non-stress-bearing and a
stress-bearing mode, not about how much of it there is. The transfer moves
energy between reservoirs without creating or destroying it.

## 4. Results

Coefficients were fit by random search plus local refinement against DNS c_f
and the full velocity field (`py_package/search.py`). Scores are
`c_f rel. RMS + 10·U RMS + θ rel. RMS`, lower is better.

### 4.1 Which structure wins

Eight structural variants were searched, each with its own coefficient fit
(`results/closure-search.json`):

| variant | total | c_f rel RMS | U RMS | θ rel RMS |
|---|---:|---:|---:|---:|
| laminar (baseline) | 2.721 | 0.674 | 0.1428 | — |
| Launder–Sharma k–ε (baseline) | 1.462 | 0.560 | 0.0505 | — |
| **Re_v, p=1** | **0.441** | **0.083** | **0.0114** | 0.243 |
| Re_v, p=0.5 | 0.445 | 0.134 | 0.0125 | 0.187 |
| Re_v, p=2 | 0.464 | 0.162 | 0.0147 | 0.155 |
| Re_v, p=1, stress-limited | 0.478 | 0.152 | 0.0152 | 0.175 |
| Re_k, p=1, stress-limited | 0.504 | 0.189 | 0.0183 | 0.132 |
| Re_k, p=1 | 0.508 | 0.158 | 0.0197 | 0.154 |
| Re_ks, p=1 | 0.660 | 0.223 | 0.0259 | 0.179 |
| shear-weighted streak energy | 1.552 | 0.562 | 0.0766 | 0.224 |

Against k–ε the best variant cuts skin-friction error by **6.7×** and velocity
error by **4.4×**. Two things are worth noting beyond the headline number.
First, **all four Re_v variants beat every alternative driver** — the shear
(vorticity) Reynolds number is clearly the right threshold quantity, and this
was not assumed, it was searched. Second, the rectifier exponent p = 1 wins
over both a softer (0.5) and a sharper (2) clip.

The fitted threshold is Λ_c = 503, against the classical critical vorticity
Reynolds number of ~440 \cite{Menter2006}. That is a genuine consistency
check, though a weak one since the searched range contained it.

### 4.2 What the winning model does

| x | c_f DNS | clipping | k–ε | laminar | H DNS | H clip | γ_max |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 112 | 0.00232 | 0.00264 | 0.00482 | 0.00214 | 3.50 | 6.21 | 0.02 |
| 229 | 0.00204 | 0.00233 | 0.00447 | 0.00154 | 2.67 | 3.28 | 0.25 |
| 287 | 0.00259 | 0.00270 | 0.00433 | 0.00138 | 2.26 | 2.62 | 0.59 |
| 346 | 0.00363 | 0.00342 | 0.00421 | 0.00122 | 1.89 | 2.13 | 0.97 |
| 463 | 0.00479 | 0.00448 | 0.00403 | 0.00107 | 1.57 | 1.71 | 1.00 |
| 638 | 0.00429 | 0.00443 | 0.00383 | 0.00092 | 1.50 | 1.56 | 1.00 |
| 989 | 0.00372 | 0.00406 | 0.00357 | 0.00074 | 1.46 | 1.46 | 1.00 |

The model reproduces the **c_f minimum near x ≈ 210 and the subsequent rise
to a peak near x ≈ 460** — the defining feature of this flow, and one that
neither baseline produces in any form. k–ε has no minimum at all (it decays
monotonically from a value twice too high), and laminar has no rise. The
activation γ climbs from its freestream value 0.02 to 1.0 over x ≈ 170–400,
and the shape factor lands on the DNS value downstream.

Remaining discrepancies, stated plainly: c_f is ~10 % high through the
laminar region, the transition is slightly early and less sharp than the DNS,
and the turbulent plateau is ~8 % high at the end of the plate. The shape
factor is badly wrong very near the leading edge, where the parabolic solver
is started from the DNS profile and the boundary layer is only a few cells
thick. The fitted Cgam sat **at its upper bound (20)**, meaning the search
wanted a sharper transition than the bound allowed — that bound should be
raised before these coefficients are taken seriously.

## 5. Why k–ε cannot be rescued by coefficients

Worth stating explicitly because it justifies changing the PDE rather than the
constants. For the standard k-equation with ν_t = C_μ√k ℓ and ε = C_D k^{3/2}/ℓ,

    P/ε = (C_μ/C_D) · ℓ²S² / k

so k grows whenever k < (C_μ/C_D) ℓ²S². With our boundary-layer values
(ℓ ≈ 0.16, S ≈ 0.6) that equilibrium sits at k ≈ 0.026 — the turbulent
attractor — and it is reached from *any* nonzero seed. The laminar state is
not a fixed point of the model. Coefficient changes move the attractor; they
cannot create a second one. A gate (γ, or an intermittency, or an explicit
threshold) is structurally required. This is why the term-multiplier search in
`docs/pde-discovery.md` was always going to plateau.

## 7. Searching over PDE structure, not just coefficients

Pete raised the natural next question: rather than hand-proposing models, can
we evolve them — and what would an acquisition function even look like when
we are adding *operators* and *derived quantities*, not mutating constants?
Our take, now implemented in `py_package/grammar.py` and
`py_package/evolve.py`:

**Two nested loops.** The outer loop searches discrete structure; the inner
loop fits that structure's coefficients. The outer objective is the best
a-posteriori error a structure can reach *after* its inner fit. This matters:
without the inner fit you are comparing a good structure with bad constants
against a bad structure with tuned ones, and the comparison is meaningless.

**Constrain the grammar, not the search.** This is the highest-leverage
choice and it matters more than the acquisition function. Candidates are
assembled as

    S_γ = C·rate·shape(γ)·Π_j f_j(D_j)

where `rate` has dimensions of 1/time, `D_j` are dimensionless local drivers
(Re_v, Re_k, a streak Reynolds number, the normalised total-energy gradient
|∇k|y/k, S/ω, Re_t, P/ε, and — when the closure carries it — the normalised
entropy H/ln3 and the entropy deficit 1 − H/ln3), and `f_j` are response
functions (rectifier, softclip, tanh, plain power law, inverse). Every
candidate is then dimensionally consistent, Galilean invariant, and has
admissible wall and freestream limits *by construction*. That removes the
overwhelming majority of nonsense candidates before any of them is evaluated,
which is worth far more than a clever acquisition function.

**On the acquisition function.** Discrete structures have no natural metric,
so give them one: `Candidate.features()` returns a bag-of-terms descriptor —
indicators for the rate, the shape, each (driver, response) pair present, and
the term count. A GP with an ARD-RBF or Tanimoto kernel over that vector
supports ordinary expected improvement, so standard BO machinery applies with
no modification. Tree or string kernels over full expression trees are the
more general option; a learned embedding is available if the space grows, but
would be overkill at this size. We currently run mutation/crossover evolution
over the same descriptor space, which is cheaper to get right and is the
approach with precedent in turbulence modelling \cite{Weatheritt2016}.

**Parsimony is not optional.** Extra terms can only reduce training error, so
an unpenalised search always adds them. `evolve()` scores
`error + λ·(number of terms)`; the models worth looking at sit at the knee of
the error-versus-complexity Pareto front, not at its floor.

**Where the earlier circularity goes.** Regressing PDE terms on DNS data is a
fine way to *propose* candidates and a bad way to *decide* between them. The
resolution is to demote the regression to a proposal distribution: use the
a-priori residual — what the current best model gets wrong — to rank which
library terms are worth trying, then decide strictly on the a-posteriori error
of the closed model run forward. That is what makes this loop non-circular,
and it is why every score in this document comes from a forward solve rather
than from a fit.

**Multi-fidelity.** The parabolic solver is the cheap fidelity (≈1 s per
model), OpenFOAM is the expensive one. Screen on the cheap one, promote only
the Pareto front to the expensive one.

## 8. Three findings about the existing simulation setup

These came out of porting the closure to OpenFOAM and matter independently of
whether the clipping model is right.

**The custom solver adds hidden momentum source terms.**
`sim/newModel/solver/UEqn.H` includes

    + a*fvc::grad(0.5*magSqr(U))     with a = 0.004
    + b*(fvc::grad(U) & fvc::grad(p))  with b = 2.0 s

hard-coded into the momentum equation. These are not part of any turbulence
model and they are large near the elliptical leading edge, where ∇p is steep.
Any model run through `ransFromDnsSimpleFoam` is therefore being compared
under a modified momentum equation. This cost real debugging time here: the
clipping model diverged under the custom solver while laminar converged on the
same mesh, and the difference was these terms, not the closure. `clipKGamma`
is an ordinary library RAS model, so `run.py` now runs it under plain
`simpleFoam` with the library loaded from `controlDict`. If those terms are a
deliberate experiment they should be behind a switch that defaults to off.

**The mesh cannot resolve this boundary layer.**
`blockMeshDict.template` graded y by a factor of 8 over a 120-unit-tall block.
At ny = 40 that puts the first cell at ≈ 0.9, i.e. y⁺ ≈ 30, while δ₉₉ ≈ 1.8
at x = 100 — roughly two cells across the whole boundary layer, in
wall-function territory. That is workable for a high-Re k–ε run but useless
for a wall-resolved transition model, which needs y⁺ < 1. The grading is now a
`--y-grading` argument; ny = 80 with a ratio of 500 gives a first cell of
0.018 (y⁺ ≈ 0.6), and `checkMesh` passes on it (max aspect ratio 81, max
skewness 0.94, max non-orthogonality 52). Note this also means the existing
mesh-independence study spans only wall-function-resolution meshes.

**A latent compile error.** `ransFromDns.C` called `operator()` on
`dimensionedScalar` term multipliers, which does not exist; the model could
not build. Fixed by dropping the `()`.

## 6. Honest limitations

- **One case.** All coefficients are fit to a single DNS at one freestream
  turbulence level (Tu decays 2.65 % → 0.58 % along the plate). The threshold
  Λ_c is calibrated, not derived, and in reality the critical Reynolds number
  depends on Tu. Nothing here demonstrates generality.
- **Λ_c ≈ 460–520 landed close to the classical critical vorticity Reynolds
  number (~440).** That is a genuine consistency check and encouraging, but we
  searched a range that contained it, so it is weak evidence.
- The parabolic solver is the screening tool, not the deliverable. Results must
  be reproduced in the elliptic OpenFOAM solver before they mean anything for
  the paper.
- γ is wall-normal-diffused with a turbulent Prandtl number of order one; that
  choice is unexamined.
- The fitted Cgam sat at its search bound, so the reported coefficients are not
  a converged optimum.
- Re_v = y²Ω/ν grows with wall distance, so in a tall elliptic domain it can
  exceed the threshold far from any shear layer. The parabolic solver never
  exposed this because its domain stops at y = 26. If the OpenFOAM results
  show spurious free-stream activation, the γ source needs a shielding
  function (Langtry–Menter use the turbulence Reynolds number for exactly
  this \cite{Menter2006}).
