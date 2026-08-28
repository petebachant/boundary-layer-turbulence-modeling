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
parabolic solver over the same plate (`pypkg/bl_solver.py`):

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
Klebanoff streaks. Since −⟨u'v'⟩ requires v', this energy carries very little
momentum flux. Measured directly, the pre-transitional eddy viscosity rises
from **ν_t ≈ 0.17 ν at x = 60 to 0.57 ν at x = 205**. The boundary layer is
loud but nearly laminar.

An earlier draft called that "negligible". It is not: a 40 % addition to the
molecular viscosity, in exactly the region that sets where transition happens,
is what raises the pre-transitional c_f about 15 % above Blasius and thickens
the layer. Reading it as zero is what let the model lose the streak reservoir
without the error showing up in any scored quantity (§4.5).

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

The consequence for modeling is a warning. Fitting a₁ against H over the
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
search in §6 is set up to find.

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
and the full velocity field (`pypkg/search.py`). Scores are
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
error by **4.4×**. That gap against the baselines is far larger than the search
noise and is not in doubt.

The ordering *within* the table is another matter. The four Re_v variants are
separated by 0.02 and the Re_v group from the Re_k group by 0.04, while
repeating a single fit moves the score by up to 1.1 (§4.7). So "all four Re_v
variants beat every alternative driver" and "p = 1 wins over 0.5 and 2" are
**not supported** by these numbers; the top eight rows of this table are one
sample from a distribution wide enough to reorder them. The only rankings that
survive are the ones against the laminar and k–ε baselines at the top.

The fitted threshold is Λ_c = 503, against the classical critical vorticity
Reynolds number of ~440 \cite{Menter2006}. This looked like a consistency
check at the time, and it is a weaker one than it appears: repeats of the same
fit put Λ_c anywhere between 398 and 519 (§4.7), so 441, 491, 503 and the
classical 440 are all one number to within the noise. The honest statement is
that our threshold is *consistent* with the classical critical vorticity
Reynolds number. It is not a recovery of it, and the range we searched
contained it anyway.

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

### 4.3 The elliptic test: OpenFOAM

The parabolic solver is only the screening fidelity. The closure was
implemented as an OpenFOAM RAS model (`sim/newModel/src/clipKGamma`) in its
portable k-omega-gamma form, where the length scale comes from a transported
omega rather than an algebraic mixing length, and run on a wall-resolved mesh
(ny = 80, grading 500, first cell y+ ~ 0.6).

**It converges.** With the corrected model (§4.5) the run holds velocity
residuals at ~3e-5, pressure at 3e-4 and gamma at 2e-4 over 3000 SIMPLE
iterations, with 27 bounding events on k, all of them in the first few hundred
iterations and all of order 1e-9. That is an order of magnitude better on
pressure than the previous version, which stalled near 5e-3 — the gated omega
production evidently makes the system less stiff as well as more accurate.

Running the two baselines on the *same* wall-resolved mesh makes the
comparison like for like:

| model | U rel RMS | c_f rel err mean | c_f rel err max |
|---|---:|---:|---:|
| **clip k-γ** | **0.022** | **0.141** | **0.448** |
| laminar | 0.055 | 0.535 | 0.809 |
| k-ε | 0.081 | 0.976 | 2.299 |

Velocity error is **3.7× lower than k-ε and 2.5× lower than laminar**, and
pressure error is ~5e-4 for all three, so pressure does not discriminate on a
flat plate.

The skin friction tells a more useful story than the mean, because the error
is not spread evenly. Before and after the fixes of §4.5:

| x | 100 | 150 | 205 | 260 | 310 | 380 | 450 | 600 | 800 | 980 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c_f error, before | −5 % | −10 % | −13 % | **+55 %** | **+96 %** | +33 % | +12 % | +10 % | +5 % | +2 % |
| c_f error, after | +5 % | +12 % | +12 % | **+2 %** | **+37 %** | +45 % | +21 % | +18 % | +9 % | +0 % |

Mean c_f error falls from 18.7 % to 14.1 % and the **worst station from 96 % to
45 %**. Two things changed qualitatively:

- **The pre-transitional error changed sign**, from −13 % to +12 %. With the
  streak reservoir switched back on, the model now produces pre-transitional
  momentum transport where before it had essentially none — which is right in
  kind (the DNS has ν_t ≈ 0.2–0.6 ν there) but currently somewhat too strong.
- **The transition overshoot is much reduced but has moved downstream.** The
  x = 260 station is now within 2 %, but the peak error has shifted to
  x = 310–380, and the early turbulent region (450–700) runs 14–21 % high
  where it was 8–12 % high before. The error has been redistributed rather
  than removed.

The eddy viscosity behaves as intended: it stays small through the
pre-transitional region, then rises by two orders of magnitude across
transition. The boundary layer is nearly laminar in the mean before the clip,
which was the whole point.

**The free-stream activation risk did not materialise.** Section 9 flagged
that Re_v = y²Ω/ν grows with wall distance and might exceed the rail far from
any shear layer in the 120-unit-tall elliptic domain. In the converged
solution gamma sits at its inlet value of 0.0200 in the free stream at every
station, because the free-stream shear is small enough that Re_v stays below
the rail despite the large y. The concern was reasonable but the data says no
shielding function is needed for this configuration. It may still be needed
for a case with a sheared or accelerating free stream.

**The screening solver is optimistic, and by a lot.** The same closure and the
same coefficients give c_f rel RMS of 4.3 % in the parabolic solver and a mean
c_f error of 14.1 % in OpenFOAM. Some of that is definitional — one is an RMS
over stations, the other a mean of absolute relative errors — but not most of
it. The parabolic solver is handed the DNS inlet profile, marches without an
elliptic pressure field, and does not see the elliptical leading edge. It is a
usable screen for *ranking* structures, which is what it is used for, and it
should not be quoted as the model's accuracy. Every headline number in this
document should come from the elliptic run.

### 4.4 Ablation: which terms actually earn their place

A model that fits is not evidence that each of its terms matters. Removing one
ingredient at a time from the fitted closure and re-scoring
(`scripts/ablate-closure.py`, `results/closure-ablation.json`):

| ablation | total | c_f rel RMS | k err (pre) | Δ total |
|---|---:|---:|---:|---:|
| fitted reference | 10.20 | 0.043 | 1.25 | — |
| drop viscous decay of streak energy (C_ν = 0) | 11.71 | 0.073 | 1.10 | +1.51 |
| lift-up on √(γk) instead of √k | 12.04 | 0.062 | 1.71 | +1.84 |
| standard Wilcox β = 0.072 instead of the fitted 0.047 | 12.58 | 0.101 | 0.72 | +2.38 |
| drop lift-up production (C_L = 0) | 12.71 | 0.074 | 1.74 | +2.52 |
| classical threshold Λ_c = 440 instead of the fitted 491 | 17.35 | 0.167 | 1.16 | +7.15 |
| **ungated ω production** (α·S² instead of the gated form) | 23.64 | 0.243 | 2.10 | **+13.44** |
| no activation gating at all (an ordinary k–ω model) | 48.71 | 0.675 | 0.25 | +38.51 |
| **remove the clip** (threshold so low the source is always on) | 49.86 | 0.695 | 0.20 | **+39.67** |

**This table replaces an earlier one whose numbers were wrong.** The ablation
script rebuilt the solver settings by hand and did not match the ones the
coefficients were fitted under: it omitted `freestream_decay` and left
`liftup_mode` at its default `"active"` rather than the fitted `"total"`.
Every row therefore measured a configuration difference on top of the term
being removed. The totals also now include the k error term (§4.5), so they
are not numerically comparable with the old ones. Two conclusions from the old
table are **withdrawn**:

- *"the lift-up production and the viscous decay of un-activated energy both
  earn nothing; the recommended model is simpler than the one we fitted."*
  They are worth **+2.52** and **+1.51**. The recommended model is not simpler.
- *"the classical threshold Λ_c = 440 is as good as the fitted one, so the
  model needs no bespoke constant here."* Forcing 440 costs **+7.15** at this
  coefficient set. Note that this ablation is a deterministic evaluation with
  the other coefficients held fixed, so it measures the local sensitivity to
  Λ_c honestly — but the fitted Λ_c it is measured against is itself uncertain
  by ±35 (§4.7), so the size of the penalty should not be read too precisely.

What survives, and more convincingly than before:

1. **The clip is load-bearing by a wide margin.** Removing the rectified
   threshold costs +39.7 and takes c_f error from 4.3 % to 69 %, worse than
   k–ε. The next most important structural choice is worth a third as much.
   The central claim of this investigation passes a considerably fairer test
   than the one it passed before.
2. **The ω gating is the second most important structure in the model** — see
   §4.5. It was not previously tested because it was not previously in doubt.
3. **β = 0.072 is still rejected.** The fit wants 0.047, and forcing the
   standard value more than doubles c_f error. This remains a single-case
   calibration compensating for something and should not be trusted until
   tested on a second flow.

### 4.5 The streak reservoir was not actually running, and why

The two-reservoir picture of §2.3 is the part of this model that is not
already in the literature. Until this was checked, the model did not
reproduce it. Measured against the DNS
(`scripts/analyze-streak-reservoir.py`, `results/streak-reservoir.json`),
peak k through the pre-transitional region came out **5–10× too low** in both
solvers:

| x | 60 | 100 | 150 | 205 |
|---|---:|---:|---:|---:|
| k_peak DNS | 3.2e-3 | 4.7e-3 | 6.0e-3 | 7.2e-3 |
| k_peak model (before) | 1.0e-3 | 7.9e-4 | 6.1e-4 | 2.3e-3 |

The model's k *decays* over the stretch where the DNS grows by a factor of
two. Without the reservoir the closure is a Re_v-threshold intermittency
model — close to existing practice \cite{Menter2006} — rather than a statement
about streak energy. Three separate causes, all now fixed.

**The objective could not see k.** It scored c_f, U, θ and the free-stream
decay. Losing the reservoir therefore cost nothing, and the fit bought c_f
accuracy through a compensating error: no streak energy gives a thin
pre-transitional boundary layer, which inflates Re_v, which fires the clip
early, which was then absorbed by raising Λ_c. `k_log_rms` is now a term in
`Case.score` with a target of 0.20 in log units.

**The ω equation was structurally wrong for a gated eddy viscosity.** The
strain-based production α·S² is the SST substitution for the textbook
α·(ω/k)·P, and the two agree **only when ν_t = k/ω**. This closure gates the
viscosity, ν_t = γk/ω, so the equivalent strain form carries a γ. Left
ungated, the mean shear drives ω up in a region that carries no turbulence,
the streak energy is dissipated at the turbulent rate, and the reservoir
empties. The comment in the code justifying the ungated form — that ω is a
frequency scale rather than an energy — was a numerical patch for a ν_t
blow-up, and it cost the physics.

Three variants were fitted, each with its own inner coefficient search
(`results/clip-k-gamma-coeffs.json`, `structure_ranking`):

| ω production | total | c_f rel RMS | c_f rel max | k err (pre) | Λ_c |
|---|---:|---:|---:|---:|---:|
| `exact`: α·min(ν_tω/k, 1)·S² | 10.78 | 0.043 | 0.088 | 1.25 | 491 |
| `gamma`: α·(γ + γ₀)·S² | 11.12 | 0.044 | — | 1.09 | 544 |
| `none`: α·S² (what we had) | 11.99 | 0.068 | 0.156 | 1.84 | 497 |

**These are single-seed fits and the ranking in this table is not
significant.** Repeating the identical fit with nothing changed but the random
seed moves the score by ±1.1 (§4.7), which is comparable to or larger than
every gap above. The table records what one run produced; it is not evidence
that `exact` beats `none`.

The case for the `exact` form does not rest on that table. It is a
*derivation*, not a fit result: α·S² is the SST substitution for the textbook
α·(ω/k)·P, the two agree only when ν_t = k/ω, and this model has
ν_t = γk/ω. Writing the textbook production with this model's own
P = (ν_t + ν_L)S² gives α·(γ + ν_Lω/k)·S² directly. The γ part is exact; the
lift-up part supplies a physically derived floor instead of a fitted seed; the
cap at 1 keeps it below the ungated form and removes the k → 0 wall
singularity. **It introduces no new fitted constant.** Its effect on the
elliptic solution (§4.3) is a single deterministic solve and is not subject to
the search noise: worst-station c_f error halves, and the activation γ = 0.5
moves from x = 140 to x = 301, against a DNS c_f minimum at x = 205.

**OpenFOAM was not running the model that was fitted.** `nuL` used the active
amplitude √(γk) while `scripts/fit-openfoam-coeffs.py` fitted with the total
√k. Pre-transition γ ≈ 0.02, so the elliptic solver ran the lift-up term about
seven times weaker than calibrated. This is the third time a fitted quantity
has failed to cross into the elliptic solver (§7), and the first two were
caught only by accident. √k is the right choice on its own merits:
dk/dt ∝ √k integrates to algebraic rather than exponential growth, which is
the non-modal behavior transient-growth theory predicts, and k = 0 remains a
fixed point, so a boundary layer with no free-stream turbulence stays laminar.

What the DNS says the model has to hit, measured directly from the
fluctuation-energy budget:

| x | 60 | 100 | 150 | 205 | 260 |
|---|---:|---:|---:|---:|---:|
| ν_t/ν at the production peak | 0.17 | 0.27 | 0.41 | 0.57 | 0.95 |
| a₁ there | 0.027 | 0.024 | 0.022 | 0.024 | 0.034 |
| advection / production | 0.33 | 0.27 | 0.25 | 0.23 | 0.24 |

The pre-transitional eddy viscosity really is small — a few tenths of ν — but
it is not zero, and only about a quarter of the production it drives goes into
growing k. That combination is what lets a nearly laminar mean profile carry a
streak energy comparable with the equilibrium turbulent value. An earlier
draft called ν_t ≈ 0.44ν "negligible"; it is not, it is a 40 % addition to the
viscosity in exactly the region that sets where transition happens.

After the fixes the reservoir is present but still thin: pre-transitional
k_peak is **0.29× the DNS**, up from 0.14×. That is the largest remaining
physical error in the model.

### 4.6 What length scale the streaks actually use

Chasing the remaining deficit produced the sharpest a-priori result in this
section. Write the pre-transitional eddy viscosity in mixing-length form,
ν_t = C_ℓ √k δ₉₉, and measure C_ℓ from the DNS at the point where the shear
production peaks:

| x | 60 | 100 | 150 | 205 |
|---|---:|---:|---:|---:|
| C_ℓ = ν_t / (√k δ₉₉) | 0.00250 | 0.00251 | 0.00262 | 0.00269 |
| y_Ppeak / δ₉₉ | 0.44 | 0.40 | 0.36 | 0.30 |

**C_ℓ = 0.0026 ± 0.0001** — constant to 4 % across the whole pre-transitional
region, while ν_t itself triples and δ₉₉ doubles. The streak eddy viscosity
follows an ordinary outer mixing-length scaling on a length that is a fixed
fraction of the boundary-layer thickness, and the production sits at
y ≈ 0.37 δ₉₉, in the middle of the layer. Downstream of transition the
production peak collapses to y ≈ 0.02 δ₉₉ and the outer scaling stops applying,
which is the expected signature of production migrating to the wall.

This is the target a local closure has to hit, and it is where our closure
fails. Its length scale is capped by C_s√k/ω, which sits at ≈ 0.57 and barely
moves along the plate, where the DNS wants 0.14 δ₉₉ growing from 0.20 to 0.44.
The consequence is a bootstrap: too little ν_L gives too little production
gives too little k gives too little ν_L. Measured term by term at x = 60, the
model's dissipation is **2.0× its production** where the DNS runs at 0.66×, so
the streak energy decays over the stretch where the DNS grows it threefold.

δ₉₉ is not available to a general-purpose CFD code, which is the whole reason
the portable form uses √k/ω. The two candidate local replacements — a
wall-distance-limited mixing length, giving algebraic streak growth, and
k/ω, giving exponential growth — are now both fitted as structural variants
rather than assumed. Neither is significantly better than the other (§4.7).

### 4.7 How much of a structure comparison is search noise

Everything in §4.1 and §4.5 ranks model structures by the best score their
inner coefficient fit reaches. That is only evidence if re-running the inner
fit does not move the score by more than the gaps being ranked. It does.

Fitting three structural variants four times each, changing nothing but the
random seed (`scripts/measure-fit-noise.py`, `results/fit-noise.json`):

| variant | mean total | sd | range | Λ_c | k err (pre) |
|---|---:|---:|---|---:|---:|
| mixing lift-up, ungated | 11.36 | **1.10** | 10.58 – 13.23 | 463 ± 38 | 1.09 ± 0.21 |
| mixing lift-up, (1−γ) gated | 11.16 | **0.35** | 10.87 – 11.73 | 463 ± 33 | 0.93 ± 0.05 |
| k/ω lift-up, (1−γ) gated | 10.72 | **0.37** | 10.35 – 11.22 | 489 ± 29 | 1.10 ± 0.04 |

The consequences are worth stating bluntly.

1. **None of the structural rankings in this document are significant.** The
   between-structure gaps are 0.2 to 0.6; the within-structure spread is up to
   1.1. Every conclusion of the form "structure A beats structure B" that
   rests on a single fit — including the ω-gating table in §4.5 and the
   eight-variant table in §4.1 — is unsupported. The same applies to the
   evolutionary search, whose reported Pareto front separated candidates by
   0.04.
2. **Λ_c is identified to about ±35 at best**, drifting between 398 and 519
   across repeats of the same structure. The fitted values of 441, 491 and 503
   quoted in earlier sections are all the same number to within the noise, and
   so is the classical 440 \cite{Menter2006}. The right statement is that our
   threshold is *consistent* with the classical critical vorticity Reynolds
   number, not that it recovers it and not that it contradicts it.
3. **The one difference that is real is in the variance, not the mean.**
   Gating the lift-up term by (1−γ) cuts the spread of the total from 1.10 to
   0.35 and the spread of the pre-transitional k error from 0.21 to 0.05. The
   ungated landscape has bad basins — one seed in four fell into a 13.2
   solution — and gating removes them. That is a reason to prefer the gated
   form, but it is a statement about how well-posed the calibration is, not
   about how well the model fits.

The methodological point generalizes beyond this closure. §6 proposes
searching over PDE structure with an outer loop over structure and an inner
loop over coefficients, and argues that the inner fit is what makes the
comparison meaningful. That is right, and it is exactly why the inner fit has
to be shown to be converged before any outer comparison is reported. A
structure search built on an unconverged inner fit is a random number
generator with extra steps.

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

## 6. Searching over PDE structure, not just coefficients

Pete raised the natural next question: rather than hand-proposing models, can
we evolve them — and what would an acquisition function even look like when
we are adding *operators* and *derived quantities*, not mutating constants?
Our take, now implemented in `pypkg/grammar.py` and
`pypkg/evolve.py`:

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
(Re_v, Re_k, a streak Reynolds number, the normalized total-energy gradient
|∇k|y/k, S/ω, Re_t, P/ε, and — when the closure carries it — the normalized
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
approach with precedent in turbulence modeling \cite{Weatheritt2016}.

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

## 7. Four findings about the existing simulation setup

These came out of porting the closure to OpenFOAM and matter independently of
whether the clipping model is right.

**Model coefficients were never being read.** OpenFOAM looks for a turbulence
model's coefficients in `RAS { <model>Coeffs { ... } }`. The template placed
`ransFromDnsCoeffs` — and, initially, our `clipKGammaCoeffs` — at the *top
level* of `constant/turbulenceProperties`, outside the `RAS` block, where they
are silently ignored: no warning is issued and every model simply falls back
to its built-in defaults.

This is easy to miss and expensive. It surfaced only because a coefficient
change that moved `alphaOmega` from 0.52 to 0.83 and `Cgam` from 0.6 to 165
produced a **byte-identical** OpenFOAM result, converging at the same
iteration. The solver's own `printCoeffs` output then showed the defaults.

The consequence for earlier work is worth stating plainly: `sim/evolve-model.py`
and the term-multiplier study in `docs/pde-discovery.md` wrote their
coefficients into that top-level block, so **none of them ever reached the
solver**. Every iteration of that search ran the identical default k-epsilon
model, and its loss differences were numerical noise rather than model
response. That is a sufficient explanation for a search that appears to
plateau, and it should be re-run before any of those conclusions are used.

Fixed by moving both coefficient dictionaries inside `RAS`, and verified by
`printCoeffs` echoing the fitted values back.

**The custom solver's extra momentum terms are a deliberate experiment, and
they are still a confound.** `sim/newModel/solver/UEqn.H` includes

    + a*fvc::grad(0.5*magSqr(U))     with a = 0.004
    + b*(fvc::grad(U) & fvc::grad(p))  with b = 2.0 s

hard-coded into the momentum equation. These are **not stray leftovers** — they
are Pete's candidate Reynolds-stress residual from `notebooks/main.ipynb`,

    −(1/ρ)∇·R = a∇K + b(∇U·∇P) + c(∇·U)∇P

with the first two terms implemented and the third omitted. An earlier draft of
this document mischaracterised them as accidental; that was wrong.

They remain a genuine confound for *model comparison*, though, because any
closure run through `ransFromDnsSimpleFoam` is being evaluated under a modified
momentum equation, with fixed coefficients that were never refitted. They are
also large near the elliptical leading edge where ∇p is steep: the clipping
model diverged under this solver while laminar converged on the same mesh, and
the difference was these terms rather than the closure. `clipKGamma` is an
ordinary library RAS model, so `run.py` now runs it under plain `simpleFoam`.
The right fix is to put these terms behind a switch defaulting to off, so the
experiment is preserved but does not silently contaminate other comparisons.

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

## 8. Keeping experiments reproducible as the model changes

Originally `sim/newModel` was a **dependency of the `blsim` Docker
environment**, and the Dockerfile compiled the models into the image. That
coupling is quietly destructive for a project whose whole point is trying many
models: every edit to a closure produced a new image and a new environment
lock, which invalidated *every* stage that used that environment — including
`mesh-independence` and `laminar-sim`, which never touch the custom library.
Old results could not be reproduced without rebuilding the exact image.

The models are now built **into the working tree at run time** instead:

- `sim/Dockerfile` provides only a stable toolchain (OpenFOAM + Python +
  foamPy) and no longer copies or compiles `newModel`.
- `sim/build-model.sh` compiles into
  `sim/newModel/platforms/$WM_OPTIONS/lib`, which is a DVC-tracked output of
  the `build-turbulence-lib` stage.
- `sim/foam-env.sh` points `FOAM_USER_LIBBIN` and `LD_LIBRARY_PATH` at that
  directory so solvers resolve the library from the tree.
- Stages that need a model list the built library as an **input**, so editing
  a closure invalidates only those stages.

The practical consequences: the environment lock is now stable across model
development; unrelated simulations are no longer invalidated; and because the
built library is content-addressed in the DVC cache, checking out an older
commit restores the exact model binary that produced those results.

### Where outputs live

Small text outputs are stored in **git** rather than DVC
(`storage: git` on the pipeline output), because they are what a human
actually reads and reviews: coefficient sets, ranked search results, per-station
error tables. Keeping them in git means they diff in a pull request, they are
present after a plain `git clone` with no `dvc pull`, and the history shows how
a model's fitted coefficients moved as the closure changed. Binary or bulky
outputs — figures, the compiled turbulence library, sampled `postProcessing`
directories, the DNS HDF5 — stay on DVC.

The rough rule used here: if it is text, under a few hundred kilobytes, and
someone might want to read the diff, put it in git.

A natural next step, if the collection of models grows, is to split
`src/Make` into one library per model (`libclipKGamma`, `libransFromDns`) so
that editing one closure does not invalidate simulations using another. Right
now they share a single library.

## 9. What this is for, and how it might relate to LES

The target is engineering prediction, not instantaneous flow: mean velocity,
mean pressure, and **wall shear stress**, for things like wings and bluff
bodies. Boundary layers are the deliberately small first scope. If the
attached boundary layer is right there is a reasonable hope that separation
follows, since separation is largely a question of how much momentum the
near-wall layer has when it meets an adverse pressure gradient — but that is a
hope, not a result, and nothing here tests it.

Skin friction is now scored explicitly rather than left implicit in the
velocity profile (`scripts/compare-openfoam-dns.py`). That matters because the
two can diverge badly: our best model currently has a **2 % mean velocity
error but a 29 % mean skin-friction error**, because c_f depends on the wall
gradient rather than on the profile as a whole. For drag prediction the c_f
number is the one that counts, and by that measure the model is not yet good
enough.

Mean pressure is also validated, as the wall-normal variation of p (pressure
is only defined up to a constant). Across all three models the pressure error
is ~5e-4, so pressure is currently not discriminating in this flow — the
boundary layer is thin and the wall-normal pressure variation is small. It
will discriminate on a curved or separating geometry, which is another reason
to move beyond a flat plate.

### Relation to LES and coarse unsteady RANS

The clipping idea may transfer, and there is a specific reason to think so.
What the activation γ encodes is not "how much energy is present" but "how
much of the fluctuation field is correlated enough to transport momentum" —
in §2.4 the coherence rises 3.3× through transition while the energy
partition moves only 1.3×. That distinction is exactly the one a subgrid model
faces on a coarse mesh: the resolved field may carry plenty of energy while
the unresolved motions are not yet organised into stress-bearing structures.

Three concrete connections, none of them tested here:

1. **Grey-zone / "terra incognita" modeling.** On meshes too coarse to
   resolve the energetic eddies but too fine for full RANS, standard subgrid
   models over-dissipate because they assume equilibrium — the same
   assumption that makes k-ε turbulent from the leading edge. A gated
   viscosity, active only where a local threshold is exceeded, is the same
   remedy in a different setting.
2. **Atmospheric boundary layers.** A stably stratified ABL suppresses
   vertical motion much as a laminar boundary layer does: energy present,
   correlation absent. The measured collapse of R_uv is a natural diagnostic,
   and the Λ_c threshold has an obvious analogue in a critical Richardson
   number. Coarse LES of the stable ABL suffers a well-known
   over-mixing problem which has the same shape as the failure we fixed here.
3. **Transition in unsteady RANS.** The γ equation is already unsteady-ready:
   it is a transport equation, not an algebraic correlation, so it carries
   history. That is the property intermittency-based transition models need
   and algebraic criteria lack.

The honest caveat is that a subgrid model must depend on filter width Δ, and
nothing in our formulation does. Re_v = y²Ω/ν is a wall-distance criterion,
not a resolution criterion. Making this an LES model would mean finding the
Δ-dependent analogue of the rail, which is a research question rather than a
port.

## 10. Honest limitations

- **One case.** All coefficients are fit to a single DNS at one freestream
  turbulence level (Tu decays 2.65 % → 0.58 % along the plate). The threshold
  Λ_c is calibrated, not derived, and in reality the critical Reynolds number
  depends on Tu. Nothing here demonstrates generality.
- **Λ_c is a fitted constant, not a recovered one, and it is not well
  identified.** Repeats of the same fit put it between 398 and 519 (§4.7). It
  is consistent with the classical ~440 and with our quoted 491; the data
  cannot presently distinguish them.
- **No structural conclusion in this document is statistically supported.**
  Structures are separated by less than the spread of their own coefficient
  fits (§4.7). The structural choices we have made are defended by derivation
  and by a-priori DNS measurement, not by their scores.
- **The streak reservoir is present but still thin.** Pre-transitional peak k
  is 0.29× the DNS after the fixes of §4.5, up from 0.14×. This is the largest
  remaining physical error, and it matters more than its effect on c_f
  suggests, because the streak energy is the part of this model that is not
  already in the literature.
- **The mean-field metrics could not see the biggest physical error in the
  model, and did not, for a long time.** That is a warning about the method,
  not just about this closure: adding k to the objective changed which
  structure wins, which terms ablate as load-bearing, and where the threshold
  fits.
- The parabolic solver is the screening tool, not the deliverable. Results must
  be reproduced in the elliptic OpenFOAM solver before they mean anything for
  the paper — and the gap between them is large (4.3 % against 14.1 % on skin
  friction), so screening numbers must never be quoted as accuracy.
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
