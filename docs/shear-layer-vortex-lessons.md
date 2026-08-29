# What the shear-layer/vortex DNS study teaches this project

> **Authorship.** Written by Claude (Anthropic's Claude Code), prompted by Pete
> Bachant, on 2026-08-29, after reading Part II of the study
> \citep{Sansica2026} and fetching the DNS data behind it. Numbers quoted for
> our own closures come from `results/benchmark.json` and the sensitivity runs
> recorded in `pypkg/cases/temporal_mixing_layer.py`.

---

## 1. The flow, the data, and what we did with it

\citet{Lusher2026} (Part I) simulated a temporally evolving tanh shear layer
between streams at ±ΔU/2, seeded with random noise and a weak soliton-like
wave, at ΔU L_x/ν = 250,000 on 12.7 billion points, ensemble-averaged over
five realizations. The layer is turbulent almost immediately, thickens as a
plane mixing layer, then rolls up through the Kelvin–Helmholtz instability
into two co-rotating vortices that merge into one by t̂ ≈ 3. There are no
walls. \citet{Sansica2026} (Part II) ran nine RANS variants on it — SA, SA-R95
with two values of C_rot, SA-R23, SA-R95-QCR2000, SA-RC, SST, SST-V and the
SSG/LRR Reynolds-stress model — in two independent solvers (JAXA's FaSTAR and
NASA's FUN3D), which agree with each other to under 1 %.

**What is public.** One file on the JAXA DNS database \citep{JAXADNSDatabase}:
301 instants, t̂ = 0–6, of peak and integral quantities over the (x, y) plane —
peak vorticity of both signs, minimum pressure, peak velocity, peak effective
eddy viscosity, momentum and vorticity thickness, peak shear stress and TKE,
integrated TKE, production and kinetic energy. These are the curves in Part
II's Figs. 3–4. The NASA TMR, also named in the data-availability statement,
does not list the case as of 2026-08-29.

**What is not public.** Fields and profiles. Everything in Part II §4.3 — the
effective-eddy-viscosity fields, the tensor-anisotropy match, the optimal
pointwise C_cr1 — is computed from the DNS fields and cannot be reproduced
from the published file.

**What we did.** The pre-roll-up phase, t̂ ≲ 0.4, is one-dimensional: the DNS
momentum thickness grows linearly at dδ_θ/dt = 0.015 ΔU (Rogers & Moser's
self-similar value is 0.014 \citep{RogersMoser1994}) and the peak shear
stress plateaus at 0.013 ΔU². That phase is now the `temporal-mixing-layer`
Tier-1 case, the suite's first flow with no wall. The vortex phase is a
two-dimensional unsteady problem; the same file supplies everything needed to
score it, and it is specified as a Tier-2 case in
[roadmap.md §2.3](roadmap.md).

---

## 2. Lessons for our closure

### 2.1 Wall distance is not a length scale

Three of our closures — `clip-gamma`, `clip-k-gamma`, `clip-two-reservoir` —
score *identically to laminar* on the mixing layer: peak eddy viscosity 9e-9,
peak shear stress 9e-7 ΔU². Nothing diverged; the models simply produce
nothing. Their algebraic length scale is a van Driest-damped κy, where y is
distance to the wall and y⁺ is built from the wall shear. With the wall
100 y₀ away and no wall shear, the damping factor is zero everywhere. Every
one of the four transition-gate drivers in `THRESHOLD_PARAMS` also contains
y. The one closure that survives, `clip-k-omega-gamma`, does so because its
length scale is k/ω, a transported quantity.

Spalart–Allmaras has the same dependence by design, and Part II's SA runs are
a warning about where it leads: in a free flow d → ∞ kills the destruction
term, so SA's eddy viscosity "peaks very high, of the order of 0.005 Γ" in the
vortex, and "the SA model has diffused the vortices, leading to a premature
merger". Our mixing-length closures fail in the opposite direction — they kill
production — but for the same reason.

**Recommendation.** Any closure we want to survive leaving the plate has to
build its length scale and its transition gate from transported quantities.
A turbulence Reynolds number k/(νω), or the Re_T that Langtry–Menter and
Walters–Cokljat already use, would carry the same information as
Re_v = y²Ω/ν on a plate and remain defined off it.

### 2.2 Eddy-viscosity models have no rotation sensitivity, and ours is one

Part II's central result: baseline EVMs "fail to capture the turbulence
depletion" as the vortex matures, because production S_ij S_ij does not
distinguish strain from rotation. Only the rotation/curvature corrections
\citep{SpalartShur1997} — which, as \citet{SpalartGarbaruk2019} showed for
a mature vortex, are the only tool that stops the baseline models from
creating opposite vorticity — which multiply production by f_r1(r*, r̃), where
r* = S/ω is the strain-to-vorticity ratio and r̃ measures how the strain
eigenframe rotates following a particle — or a Reynolds-stress model avoid
"the creation of opposite vorticity", and only by forcing ν_t to zero. The DNS
effective viscosity ν_t,eff = P_k / (2 S²) even goes negative locally.

Our closure has no such term. On a flat plate this is invisible; in the
Tier-2 vortex case it will decide the score. When that case exists the first
experiment is our model with and without an f_r1 multiplier on production.
Note the ingredient r̃ needs the Lagrangian derivative of the strain tensor,
which no parabolic solver can supply — another reason the vortex phase is
Tier 2 by mathematics rather than preference.

### 2.3 The front at the turbulent/non-turbulent interface

Part II: "two-equation and similar models also produce sharp boundaries
propagating into non-turbulent fluid, a consequence of the structure of the
partial differential equations", and EVMs "generate spurious regions of
opposite vorticity" just outside the turbulent region. We see the first of
these directly. In the mixing layer, Launder–Sharma's peak velocity gradient
sits at the *edge* of the layer, where ν_t collapses, not at the center where
the DNS peak is, and it sharpens without bound under grid refinement:
max|dU/dy| = 39, 84, 169 at n_y = 301, 601, 1201, while the momentum
thickness moves 1 % per refinement. That is why the vorticity thickness is
computed but not scored on this case.

A transported activation γ that is zero outside the turbulent region and
multiplies ν_t is one natural way to control that front — which is what our
closure already carries. Whether a *rectified* γ source softens the front or
just moves the discontinuity is exactly the toggling problem of ideas-log
§4.3, and the mixing layer is a cheaper place to study it than the plate.

### 2.4 Our misalignment diagnostic is their tensor-anisotropy match

Part II defines TAM ≡ A_ij B_ij / (|A||B|) between two traceless tensors, and
notes that in 2-D it equals cos 2α, where α is the angle between the
eigenframes. Our `analyze-flow-structure` stage reports α directly: 44° before
transition and 21–24° in equilibrium turbulence, i.e. TAM ≈ 0.03 and 0.67–0.74.
Part II finds TAM ≈ 0.85 in the core of a free shear layer and quotes 0.6–0.7
for the outer boundary layer and as low as 0.2 in the buffer layer — so our
equilibrium value sits where the literature puts it, and our pre-transitional
value is a stronger statement than any in Part II. We should report TAM
alongside α so the numbers are directly comparable, and we should note that
they, like us, treat the Boussinesq question as *how good*, not *whether*.

### 2.5 An optimal coefficient in one flow is not a model

Part II extracts the pointwise-optimal QCR constant from the DNS and finds
C_cr1 ≈ 0.21 in the mixing layer against the 0.3 calibrated in boundary
layers \citep{Spalart2000} — then says explicitly that "obtaining the
optimal C_cr1(x, y, t) in a single flow does not constitute a turbulence
model." That is our paper's thesis, from an independent group on a different
flow, and it should be cited as such: coefficients tuned in one flow are not
the optimum in another, and an in-sample fit is not a constitutive law.

### 2.6 Initial conditions decide the early phase, so score after it

Part II spends a page (§3.1, Table 1) on how each model had to be seeded —
C_νt lowered from 0.02 to 0.0084 to delay the eddy-viscosity build-up, a
different y₀ for the RSM — and concludes that "obtaining a high degree of
agreement between RANS solutions and a reference solution via proper initial
distributions of the turbulence variables is not trivial." Then, having done
it: "at t̂ = 0.4 ... there is no excuse based on initial conditions for a model
to be inaccurate later." The gym's convention — the case seeds every closure by
one stated rule and scoring starts after the transient — is the same policy.
The mixing-layer case uses Part II's own seed and scores t̂ ∈ [0.14, 0.40].

### 2.7 Peaks are revealing and fragile; integrals are what the data support

Part II uses peak vorticity, minimum pressure and peak ν_t as its primary
diagnostics and says so: "the peak vorticity ... is only one, but a very
revealing, measure", while cautioning that it "involves the peak value of a
derivative rather than an integral". We found the cost of that in §2.3. The
scoring rule that follows: integral quantities first, peaks reported, and any
peak that a model can make grid-dependent excluded from the score.

---

## 3. Lessons for the gym: how others will express a model

### 3.1 A model is a base plus named corrections plus coefficient overrides

Nine variants from three bases: SA, SA-R95, SA-R95(C_rot = 1), SA-R23,
SA-R95-QCR2000, SA-RC, SST, SST-V, SSG/LRR. That is how practitioners think
about models, and the registry should let them say it:

```python
register_closure("sa-r95-qcr", base="sa",
                 modifiers=[("r95", {"C_rot": 1.0}), "qcr2000"])
```

A modifier is one of two things: a **production multiplier** (the RC family,
a function of r*, r̃ that scales the production term of whichever base it is
attached to) or a **constitutive relation** (QCR, which replaces the Boussinesq
stress and, as Part II says, "can be combined with any eddy-viscosity model,
without adjusting the constant"). Part II also shows the second kind is
invisible in a thin shear layer — "the influence of QCR is very weak" — so the
parabolic tier cannot distinguish it and only Tier 2 can.

### 3.2 Separate the constitutive relation from the transport model

Our `Closure` returns a scalar ν_t. A `stress(gradU, state)` method with a
default Boussinesq implementation would let QCR, non-linear EVMs and
Reynolds-stress models register without touching the solver, and would let
the a-priori TAM diagnostic in §2.4 be evaluated for any registered model on
any case with DNS stresses.

### 3.3 Name the canonical description and verify against it

Part II's model list points to the NASA TMR for "complete and correct
descriptions of all models". Each registered closure should carry a bib key
and, for standard models, the TMR page it implements, and the harness should
carry a code-to-code check: Part II's two solvers agree to under 1 % on the
same model, which is what makes their model comparison a model comparison.
Our roadmap's cross-tier consistency check is the same idea.

### 3.4 Seeding must reach every transported variable

Part II needed a per-model recipe to map physical targets (peak stress, k,
dissipation) onto each model's variables (ν̃, k, ω, the six stresses). The gym's
`SeededClosure` maps the case's physical seed onto whatever standard keys a
closure carries — k, ω, ε, γ, ν_t — and leaves anything else at the closure's
own initial value. So `entropy-k-omega-h`'s coherence variable H, and
`clip-two-reservoir`'s streak/active split, start from their laminar-plate
values on a flow that is already turbulent. A closure with non-standard state
should be able to declare how to derive it from (k, ν_t), e.g. a
`seed_from(k, nut, nu, grid)` hook, so that "seeded by the same rule" is
actually true for it.

### 3.5 The interface conflates the marching coordinate with the mean velocity

`advance(grid, U, V, nu, dx, Ue, x)` uses U both as the shear that produces
turbulence and as the speed at which x is marched. A temporal flow has U of
both signs and no marching direction. The `TranslatingFrame` proxy works
around it by adding U_c ≫ ΔU to what the closure sees (results are unchanged
to four figures between U_c = 100 and 10⁴), but a clean interface would take
a `step(dt)` or a separate convection speed.

### 3.6 What the Tier-2 vortex case needs

Everything required to score the vortex phase is in the public file. Setup
from Part II §3: domain [−L_x/2, L_x/2] × [−2L_x, 2L_x], periodic in x,
symmetry top and bottom, baseline mesh 800 × 720 uniform in the vortex region,
Δt̂ = 3.33 × 10⁻⁴, run to t̂ = 6, initial condition Eq. (1) with the seeds of
Table 1. The authors state "incompressible simulations would be equally
acceptable", so OpenFOAM's `pimpleFoam` is sufficient — but note this would
be the project's first *unsteady* RANS. Every OpenFOAM stage so far is a
steady `simpleFoam` run judged by residuals; a URANS case needs a transient
setup, time-resolved sampling, and a time-step sensitivity study in place of
a convergence check (Part II, Appendix A.2, halves and quarters Δt̂ and
doubles the sub-iterations to show its histories are converged). The Tier-1
mixing-layer case is already time-marching, which the Python interface
absorbs because marching in x and in t are the same operation for a
parabolic solver; OpenFOAM makes the distinction explicit. Metrics: histories of
max and min ω_z, min p, max |u|, max ν_t and integrated kinetic energy, with
the DNS curves as reference and the nine Part II models as published
baselines. See [roadmap.md §2.3](roadmap.md).

---

## 4. Datasets as contributions

The reverse of §3 — how a group that produces a DNS gets it *into* the gym so
the leaderboard updates when their data arrive — is a pipeline and tooling
question rather than a modeling one. It is written up in
[roadmap.md §2.4](roadmap.md).
