# Ideas log: quantities and closures to try

> **Authorship.** Written by Claude (Anthropic's Claude Code), prompted by and
> in dialogue with Pete Bachant. Ideas marked **[PB]** originated with Pete;
> the rest are Claude's elaborations or additions. Numbers come from scripts in
> this repository — see [clipping-closure.md](clipping-closure.md) for the full
> investigation.

Running log so we do not lose threads or re-try things that already failed.
Each entry says what it is, what it needs, and — if tried — what happened.

---

## 1. Tried, and they worked

### 1.1 Clipping / saturating activation **[PB]**
The founding idea: a flow holds more energy than smooth waves can carry, so it
"clips" and redistributes into higher harmonics.

**Result: this is the whole model.** Ablating the rectifier takes c_f error
from 4.3 % to 69 % (no better than k-ε) and is worth +39.7 in the objective,
against +13.4 for the next most important structural choice. Structure
parameter a₁ = −u'v'/2k saturates at **0.137** and pins there. See
[clipping-closure.md §4.4](clipping-closure.md).

The earlier version of this entry said every *other* term "changes the score
in the third decimal". That was an artifact of a broken ablation script; see
§4.11. The clip still dominates, but the other terms do earn their place.

### 1.2 Vorticity Reynolds number as the threshold variable
Re_v = y²Ω/ν. Beat every alternative driver tested (Re_k, streak Reynolds
number, shear-weighted streak energy).

**The claim that the classical value comes out for free is withdrawn** (§4.11).
In the portable k-ω-γ form the threshold fits to **Λ_c ≈ 491**, and forcing
the classical 440 costs +7.15 in the objective. The earlier agreement was
measured with a broken ablation and was in any case absorbing the Re_v error
caused by the missing streak reservoir (§4.9). Λ_c is a calibrated constant,
and its dependence on free-stream turbulence level is untested
\cite{Menter2006}.

### 1.3 Two-reservoir energy split
Splitting k into a stress-bearing and a non-stress-bearing part, using the
saturated a₁ as the divider. k_active/k rises 0.10 → 1.000 monotonically;
streak energy fills then drains to zero. Closest existing model is the laminar
kinetic energy of \cite{Walters2008}, but we measured the split rather than
assuming it.

### 1.4 Coherence vs energy partition **[PB — "the wall imposes structure"]**
Decomposing a₁ = R_uv × anisotropy, where R_uv = −u'v'/√(u'u'·v'v') is the
correlation coefficient:

| | pre-transitional | turbulent | factor |
|---|---:|---:|---:|
| R_uv (coherence) | 0.131 | 0.437 | **3.3×** |
| anisotropy (energy partition) | 0.242 | 0.314 | 1.3× |

**Transition is about creating correlation, not redistributing energy.**
R_uv saturates at 0.437 and is nearly uniform across the layer (0.43–0.47 over
y/δ = 0.05–0.7) — a genuinely wall-imposed, universal structure. This means the
transported γ in our closure is best understood as *normalised coherence*, not
an abstract "activation".

### 1.5 Meaning inside randomness: total correlation **[PB]**
Shannon entropy is the wrong tool — it is *maximised* by random letters. The
right family of measures score low for **both** pure order and pure randomness.
The computable one here is the **total correlation** (multi-information)
of the fluctuation vector,

    T = Σ_i h(u_i) − h(u) = ½ log₂( σ_u²σ_v²σ_w² / det C )

which is exactly zero for independent components (the "random letters"
surrogate) and positive when the wall imposes joint structure.

**Result: T rises 0.013 → 0.153 bits, an 11.4× increase, while the energy only
grows 2.4×.** So the wall adds ~0.15 bits of shared structure per point. For
this flow (spanwise symmetry kills u'w' and v'w') T reduces exactly to the u–v
mutual information −½log₂(1−R_uv²), which is why §1.4 and this are the same
statement seen twice.

### 1.6 Exergy injection vs entropy rejection, with turbulence as storage **[PB]**
The hypothesis: a steady, nearly uniform inflow is a low-entropy, high-exergy
stream; the wall degrades it; and if entropy cannot be *rejected* fast enough,
the excess must be **stored** in the turbulence. Confirmed, and the timing is
almost exact.

Two rejection routes: direct viscous dissipation of the mean field
(ν(∂U/∂y)², immediate) and the turbulent route (production → k → ε), which is
**buffered** — energy entering k is dissipated later and downstream. The
storage rate is then d/dx ∫U k dy.

| x | 214 | **246** | 273 | 331 | 448 | 600 | 800 |
|---|---:|---:|---:|---:|---:|---:|---:|
| stored / produced | 0.165 | **0.240** | 0.205 | 0.124 | 0.008 | 0.028 | 0.022 |

**Peak storage fraction 0.240 at x = 246, against a transition onset of
x ≈ 233.** During transition roughly a quarter of the exergy routed into
turbulence is *stored* rather than rejected. Once turbulence is established
storage collapses to 2–3 % and production balances dissipation. So transition
is precisely the interval in which rejection cannot keep up with injection.

The reservoir's residence time confirms this. τ_store/τ_flow = (k/ε)/(δ/U_e)
falls **79 → 4.2**: in the laminar region energy placed in the fluctuation
field has a residence time ~80× the flow transit time — it effectively cannot
be rejected at all — and the ratio saturates near 4 once the turbulent
dissipation channel is open.

**The Reynolds-number link is real.** In the fully turbulent region the
fraction of rejection carried by the turbulent route keeps climbing with Re:

| Re_θ | 831 | 1085 | 1268 | 1391 |
|---|---:|---:|---:|---:|
| turbulent fraction of rejection | 0.520 | 0.540 | 0.552 | 0.559 |

As Re rises, direct viscous rejection becomes progressively less able to
keep up and more of the load shifts to turbulence — exactly the expected
behaviour, and measured rather than assumed.

**The inflow really is very ordered.** Freestream fluctuation energy is
2.1e-3 of the mean kinetic energy at the inlet (Tu = 2.65 %), and even
boundary-layer-integrated the disordered fraction only rises from 0.0019 to
0.0134. Better than 98 % of the energy is in the mean everywhere; turbulence
is a thin veneer of disorder on a highly ordered stream.

The cumulative budget closes at 0.86–0.88 of the wall work at every station
(the residue is the y > 12 truncation and transport through the top boundary),
and cumulative storage is never more than 3 % of cumulative input — storage is
a **transient of transition**, not a standing inventory.

---

## 2. Tried, and they did not work (worth knowing)

### 2.1 Component entropy as a closure variable **[PB — entropy idea]**
H = −Σp_i ln p_i of the energy partition. **Non-monotone**: 1.074 → minimum
**0.499 at x = 233** → 0.994. The mean shear *reduces* the entropy of the
fluctuation field, and transition onset is the point of maximum order — a
lovely result. But it makes a₁ = f(H) **hysteretic** (correlation 0.80 over the
whole plate; each branch separately collapses at 0.99). Implemented as
`EntropyKOmegaH`: reproduces the shape of H from an ordering-vs-relaxation
competition, but puts the minimum at x ≈ 33 and scores 1.02 against the
clipping model's 0.44. **A smooth relaxation balance cannot hold the laminar
state.** Entropy is a good state variable, not a substitute for the clip.

### 2.2 Term-multiplier "structural discovery" on k-ε
The approach in [pde-discovery.md](pde-discovery.md). Two independent reasons it
could not have worked: (a) the coefficients were never reaching the solver at
all (see §4.1 below), and (b) even if they had, multiplying existing terms by
constants cannot create a second fixed point — the laminar state is not a fixed
point of k-ε for *any* coefficients. See
[clipping-closure.md §5](clipping-closure.md).

### 2.3 Self-amplifying streak production
Scaling streak production on streak energy makes it a runaway; k came out 10×
the DNS value.

The fix is to scale it on the **total** amplitude √k, not on the active
amplitude √(γk): dk/dt ∝ √k integrates to algebraic rather than exponential
growth, which is the correct non-modal behaviour, and k = 0 stays a fixed
point so a boundary layer with no free-stream turbulence stays laminar.
Scaling on √(γk) instead switches the term off in exactly the pre-transitional
region it exists to represent — which is what the OpenFOAM model was doing
(§4.9). The earlier note here that "the lift-up term earns nothing anyway"
came from the broken ablation and is withdrawn (§4.11): it is worth +2.52.

---

## 3. Not yet tried — differential and structural measures **[PB asked]**

Quantities involving derivatives that plausibly capture "meaning". Ordered by
expected value per unit of work.

### 3.1 Alignment of the anisotropy tensor with mean strain — **DONE, and it matters**
An eddy-viscosity closure is exact only if b_ij ∝ S_ij, i.e. their eigenframes
coincide. Measured angle between the leading eigenvectors of −b_ij and S_ij:

| x | 100 | 205 | 264 | 310 | 381 | 450 | 600 | 907 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| misalignment (deg), y/δ=0.2 | 44.4 | 44.0 | 42.9 | 41.4 | 36.1 | 29.7 | 24.3 | 23.7 |
| misalignment (deg), y/δ=0.5 | 43.1 | 42.6 | 40.7 | 37.3 | 31.9 | 27.6 | 22.0 | 20.7 |

The mean strain sits at 45° everywhere (pure shear). The anisotropy tensor sits
at **~89° pre-transition** — i.e. aligned with the streamwise axis, the
signature of one-component streaks — and rotates to ~66–69° once turbulent.

Two conclusions, and the first is a better justification for our model than the
one we had:

1. **Pre-transition the eddy-viscosity constitutive relation is not merely
   small, it is structurally wrong** — off by 44°, nearly the maximum possible.
   Gating ν_t on γ is therefore not just "the energy is not active yet"; it is
   *switching off a constitutive relation that does not hold there*. That is a
   much stronger argument for the gate.
2. **Even in equilibrium turbulence the misalignment is 21–24°**, so a linear
   eddy viscosity is imperfect everywhere, not just in transition. This bounds
   how good any ν_t-based closure — ours included — can ever be, and is a
   quantitative argument for a nonlinear/tensorial extension.

*Needed:* nothing new. Should become a pipeline stage and a figure.

### 3.2 Lumley invariants — partially done
Anisotropy shape independent of energy. Already computed: η goes 0.055
(isotropic) → **0.260 (one-component, x ≈ 205–264)** → 0.134, non-monotone and
peaking where the entropy is minimum. Worth turning into a proper figure and a
constraint — a closure should reproduce this trajectory, not just c_f.
*Needs:* nothing new.

### 3.3 Vortex stretching and strain–vorticity alignment
ω_i S_ij ω_j, and the alignment of ω with the intermediate strain eigenvector.
This is the actual cascade mechanism and a strong structural signature that
energy statistics miss entirely.
*Needs:* fluctuating velocity gradients — a new JHTDB pull.

### 3.4 Q-criterion / λ₂ occupancy
Fraction of volume where rotation dominates strain — the standard
coherent-structure identifier. The *statistics* of Q through transition would
quantify when organised vortices appear.
*Needs:* instantaneous fields from JHTDB.

### 3.5 Relative helicity cos θ = u·ω/(|u||ω|)
Alignment of velocity and vorticity; a pseudo-scalar measuring chiral
organisation. Near zero for unstructured flow.
*Needs:* instantaneous fields.

### 3.6 Pressure–strain redistribution Π_ij
This is *literally* the clipping operator: the term that moves energy between
components without changing k. Our whole model is a scalar surrogate for it.
Measuring it would let us check the surrogate directly.
*Needs:* Π_ij is not in the profiles file; we do have u'p', v'p', w'p'
correlations, which give part of the pressure transport. Partial win available
now, full version needs a new pull.

### 3.7 Predictive information / excess entropy I(past; future)
The formal answer to "meaning in randomness": zero for IID noise **and** zero
for a constant, maximal for structured processes. Related: statistical
complexity and ε-machines \cite{Crutchfield1989}, and effective complexity
\cite{GellMann1996}, which separates the algorithmic content of an object's
*regularities* from its random part.
*Needs:* time-resolved velocity at points — a JHTDB time-series pull. Very
doable and would be a genuinely novel diagnostic for transition.

### 3.8 Two-point correlation length / spatial mutual information
∫R(r)dr, and the mutual information between separated points. The true
coherent-structure scale, and the honest version of §1.5 (single-point
covariance is a weak notion of structure).
*Needs:* two-point data — new pull.

### 3.9 KL divergence of the velocity PDF from Gaussian
"Distance from featureless" at a point. Cheap once instantaneous data exists.
*Needs:* instantaneous fields.

---

## 4. Housekeeping and known problems

### 4.1 Coefficients were never read by the solver — **fixed, but re-run needed**
`<model>Coeffs` was written at the top level of `turbulenceProperties`, outside
the `RAS` block, where OpenFOAM silently ignores it. **Every earlier
`evolve-model.py` iteration therefore ran the identical default k-ε**, and its
loss differences were noise. Fixed; that study needs re-running before its
conclusions are used.

### 4.2 The custom solver has hidden momentum terms
`ransFromDnsSimpleFoam` adds `0.004·∇(½|U|²) + 2.0·(∇U·∇p)` to the momentum
equation. Any model compared through it is being compared under modified
physics. `clipKGamma` now runs under plain `simpleFoam`. These should be behind
a switch defaulting to off.

### 4.3 Convergence with a stiff rectifier
With the fitted Cgam = 165 the elliptic run reaches the iteration limit rather
than the convergence criterion (pressure stalls near 5e-3). A hard rectifier
makes cells toggle across the threshold. **Try the `softclip` response** already
in the grammar — likely converges cleanly at similar accuracy.

### 4.4 β = 0.1 is suspicious
The fit wants 0.1 against the standard 0.072, and forcing the standard value
doubles c_f error. Single-case calibration compensating for something, probably
the turbulent equilibrium. **Do not trust until tested on a second flow.**

### 4.5 Free-stream boundary conditions were never carried to OpenFOAM
`omega_fs_scale` was a fitted parameter but belongs in no coefficient
dictionary, because it is a boundary condition. The elliptic solver therefore
ran with free-stream omega 10-20x too small, the free stream barely decayed,
and by x=980 the boundary layer was fed 5.4x too much turbulence. This was the
whole reason OpenFOAM showed 33 percent skin-friction error where the
screening solver claimed 4 percent.

Worse, **the screening solver could not detect it**, because it imposed the
measured DNS k_inf(x) and was handed the right answer for free. The model now
generates its own free stream from its own decay law (`freestream_decay`) and
the mismatch is penalised in the objective, so the decay exponent
betaStar/beta became a real constraint. Inlet k and omega are templated and
written from the same law the model was fitted under.

### 4.6 The momentum-thickness metric was corrupted
Edge velocity was taken at the top of the domain, but in this DNS the velocity
peaks INSIDE the domain and falls slightly towards the upper boundary -- about
28 percent of nodes exceed the top-node value. The integrand of the momentum
thickness therefore went negative and **theta came out negative for the DNS
itself**, with shape factors of order 1e11 at the inlet.

`theta_rel_rms` was consequently ~32 percent for every model regardless of
quality, contributing over half the objective as pure noise, and a full
coefficient fit was carried out against it. Fixed by taking the edge velocity
from each profile's own maximum; the DNS now gives positive theta and H = 2.1
laminar falling to 1.46 turbulent.

Lesson worth keeping: a shape factor of 1.5e11 appeared in several diagnostic
tables before anyone chased it. An absurd number in a column you are not
currently looking at is still a bug.

### 4.7 The free-stream/boundary-layer conflict in beta was not real
When the free-stream constraint was first imposed, the boundary layer appeared
to want beta ~ 0.09 while the DNS decay wanted ~0.045, suggesting the classic
k-omega free-stream sensitivity and an SST-style blend. Blending was
implemented and made things **worse** (12.51 against 12.14). A direct scan then
showed beta = 0.045 is optimal both with and without the constraint: the
apparent preference for 0.09 was an artifact of leaving free-stream omega free,
which let the model compensate for excess free-stream turbulence rather than
decay it. Blending is kept as an option and documented as unnecessary here.

### 4.8 Far-field upper boundary: tried, much worse
The `upperWall` zeroGradient condition forces the flow parallel, so the
displaced boundary layer accelerates the free stream by blockage: measured
+0.75 percent along the plate where the DNS *decelerates* by 1.3 percent, a
~2 percent error in the streamwise pressure gradient. Since pressure gradient
sets transition location, this looked worth fixing.

Replacing it with a proper far field (`pressureInletOutletVelocity` on U,
fixed pressure) is **far worse**: the fixed far-field pressure drains the
domain, Ue collapses from 1.017 to 0.68 by the end of the plate, the boundary
layer grows to 2.6x the DNS thickness and mean skin-friction error goes from
19 percent to 114 percent. Reverted.

If the residual 2 percent matters, the right fix is a taller domain, not a
different boundary condition. Worth noting this affects **every** simulation
in the repo, including the mesh-independence study and the k-epsilon
baselines, which all carry the same mild favourable pressure gradient.

### 4.9 The streak reservoir was never actually running — **fixed**
The two-reservoir picture is the part of this model that is not already in the
literature, and until now the model did not reproduce it. Peak k through the
pre-transitional region came out **5–10x below the DNS** in both the parabolic
solver and OpenFOAM (x = 100: 8e-4 against 4.7e-3; x = 150: 6e-4 against
6.0e-3). The model's k *decays* over the stretch where the DNS grows. Without
the reservoir the closure is a Re_v-threshold intermittency model, which is
close to existing practice, rather than a statement about streak energy.

Three separate causes, all now fixed:

1. **The objective could not see k.** It scored c_f, U, theta and free-stream
   decay. Losing the reservoir therefore cost nothing, and the fit bought c_f
   accuracy with a compensating error: no streak energy gives a Blasius-thin
   boundary layer, which inflates Re_v (500 against the DNS 417 at x = 150),
   which fires the clip early, which was then absorbed by raising Lambda_c.
   `k_log_rms` is now a term in `Case.score`.
2. **The omega equation was structurally wrong for a gated model.** See 4.10.
3. **OpenFOAM was not running the model that was fitted.** `nuL` used the
   ACTIVE amplitude sqrt(gamma*k) while `fit-openfoam-coeffs.py` fitted with
   the total sqrt(k). Pre-transition gamma ~ 0.02, so the elliptic solver ran
   the lift-up term about 7x weaker than calibrated. Worth +1.84 in the
   ablation table.

### 4.10 alpha*S^2 in the omega equation is wrong when nu_t is gated
The strain-based omega production `alpha*S^2` is the SST substitution for the
textbook `alpha*(omega/k)*P`, and the two agree **only when nu_t = k/omega**.
This closure gates the eddy viscosity, nu_t = gamma*k/omega, so the equivalent
strain form carries a gamma. Ungated, mean shear drives omega up in a region
that carries no turbulence; the streak energy is then dissipated at the
turbulent rate and the reservoir empties. The code comment justifying the
ungated form ("omega is a frequency scale, not an energy") was a numerical
patch for a nut blow-up, and it cost the physics.

Three variants were fitted, each with its own inner coefficient search:

| omega production | total | c_f rel RMS | k err (pre) | Lambda_c |
|---|---:|---:|---:|---:|
| `exact`: alpha*min(nu_t*omega/k, 1)*S^2 | 10.78 | 0.043 | 1.25 | 491 |
| `gamma`: alpha*(gamma + gseed)*S^2 | 11.12 | 0.044 | 1.09 | 544 |
| `none`: alpha*S^2 (what we had) | 11.99 | 0.068 | 1.84 | 497 |

**This ranking is not significant** -- see §4.14. Re-running the same fit with
a different seed moves the total by up to 1.1, which swallows these gaps.

The case for `exact` is a derivation rather than a score: it is the textbook
production written with this model's own P = (nu_t + nu_L)S^2, the cap at 1
keeps it below the ungated form and removes the k -> 0 wall singularity, and
it needs no new fitted constant. Its effect on the *elliptic* solution is a
single deterministic solve and is not affected by the search noise:
worst-station c_f error halves, 15.6 % -> 8.8 %, and the pre-transitional
activation moves from x = 140 to x = 301 against a DNS c_f minimum at x = 205.

Implemented as `omegaGating none|gamma|exact` in `clipKGamma`, and as
`gate_omega` in `ClipKOmegaGamma`.

### 4.11 The ablation table was measuring the wrong model — **results retracted**
`scripts/ablate-closure.py` rebuilt the solver settings by hand and did not
match the ones the coefficients were fitted under: no `freestream_decay`, and
`liftup_mode` left at its default `"active"` rather than the fitted `"total"`.
Every number in the old table therefore measured a configuration difference on
top of the term being removed. **Two conclusions from it are withdrawn:**

- "the lift-up production and the viscous decay earn nothing, delete them" —
  they are worth **+2.52** and **+1.51** once ablated against the model that
  was actually fitted;
- "the classical threshold Lambda_c = 440 is as good as the fitted one" — it
  now costs **+7.15**. Lambda_c is a genuinely calibrated constant, and the
  agreement with the classical value was absorbing the Re_v error caused by
  the missing streak reservoir.

Corrected table (totals now include the k term, so they are not comparable
with the old ones):

| ablation | total | c_f rel RMS | delta |
|---|---:|---:|---:|
| fitted reference | 10.20 | 0.043 | — |
| drop viscous decay of streak energy | 11.71 | 0.073 | +1.51 |
| lift-up on sqrt(gamma*k) (the OpenFOAM bug) | 12.04 | 0.062 | +1.84 |
| standard Wilcox beta = 0.072 | 12.58 | 0.101 | +2.38 |
| drop lift-up production | 12.71 | 0.074 | +2.52 |
| classical threshold Lambda_c = 440 | 17.35 | 0.167 | +7.15 |
| **ungated omega production** | 23.64 | 0.243 | **+13.44** |
| **remove the clip** | 49.86 | 0.695 | **+39.67** |
| no activation gating at all (plain k-omega) | 48.71 | 0.675 | +38.51 |

The clip is still overwhelmingly the load-bearing ingredient, so the central
claim survives a much fairer test than the one it passed before. The omega
gating is now the second most important structural choice in the model.

### 4.13 Why the streak reservoir stays thin: nu_t and k are locked together
Chasing the residual k deficit (§4.9) produced a clean structural diagnosis and
falsified two of my own hypotheses along the way.

**What is NOT the problem.**

- *"The viscous decay term C_nu is draining the streak energy."* Wrong. Setting
  C_nu = 0 barely moves k (k error 1.079 → 0.944) and wrecks c_f
  (0.041 → 0.097).
- *"The shear-gated dissipation bound of 8 was too tight."* Wrong. Raised to
  60, the fit still chose C_d ≈ 1.6. There is a sharp optimum there, not a
  rail.

**What is.** Every route to more streak energy costs skin friction
(`results/closure-ablation.json`):

| change | c_f rel RMS | k err (pre) | Δ total |
|---|---:|---:|---:|
| fitted reference | 0.043 | 1.253 | — |
| C_d = 5 (weaker cascade) | 0.234 | 0.765 | +11.69 |
| C_L ×2 (stronger lift-up) | 0.131 | 0.803 | +4.79 |

Both buy real streak energy and both are heavily penalised for it.

**Correction.** An earlier version of this entry said the model's ν_t/k at
x = 60 was 0.067 against a DNS 0.066, and concluded that the ratio was already
right and only the amplitude was wrong. That came from a throwaway script,
used a different definition of ν_t (maximum over the layer rather than at the
production peak), and does not survive being computed properly. From
`results/streak-reservoir.json`:

| x | 61 | 100 | 149 | 206 | 259 |
|---|---:|---:|---:|---:|---:|
| ν_t/k model, at production peak | 0.096 | 0.121 | 0.129 | 0.159 | 0.139 |
| ν_t/k DNS | 0.067 | 0.074 | 0.086 | 0.099 | 0.136 |
| ratio | 1.44 | 1.64 | 1.49 | 1.60 | 1.02 |

So the model carries **too little k and simultaneously about 1.5× too much
ν_t per unit k**. It is not a pure amplitude error, and the tidy "the ratio is
right, only the level is wrong" story was wrong.

The dissipation side is also more localised than claimed. D/P is 1.77 against
the DNS 0.67 at x = 61 -- the model really does dissipate its streak energy
faster than it makes it right at the start -- but by x = 100 it has fallen to
0.81 and thereafter runs *below* the DNS. The deficit is set in the first
stretch of plate, not maintained along it.

What still stands is the mechanism: with ℓ_s = min(y, C_s√k/ω) and C_s bounded
below 2, the cap binds everywhere pre-transition, so
ν_L = C_L·√k·C_s√k/ω = C_L C_s k/ω -- proportional to k, not √k. The DNS law
ν_t = 0.0026·√k·δ₉₉ has no k in the length at all. But see §4.17: that turned
out to be a bound I imposed, not a property of the form.

Corroborating evidence for the degeneracy: two independent fits landed on
C_L = 0.019, C_s = 1.63 and C_L = 0.224, C_s = 0.137 — wildly different, but
**C_L·C_s = 0.031 in both**. Only the product is identified, which is exactly
what you expect if the cap always binds. The searches have been wasting a
dimension.

The open question is what k-independent local length to use, since δ₉₉ is not
available to a general-purpose CFD code — which is the whole reason the
portable form used √k/ω in the first place.

### 4.14 Most of a structure comparison here is search noise
Fitting three structural variants four times each, changing nothing but the
random seed (`scripts/measure-fit-noise.py`, `results/fit-noise.json`):

| variant | mean total | sd | range | Lambda_c |
|---|---:|---:|---|---:|
| mixing lift-up, ungated | 11.36 | **1.10** | 10.58 - 13.23 | 463 +- 38 |
| mixing lift-up, (1-gamma) gated | 11.16 | **0.35** | 10.87 - 11.73 | 463 +- 33 |
| k/omega lift-up, (1-gamma) gated | 10.72 | **0.37** | 10.35 - 11.22 | 489 +- 29 |

Between-structure gaps are 0.2 to 0.6. Within-structure spread is up to 1.1.
**No structural ranking in this log or in clipping-closure.md is supported by
its score**, including the omega-gating table above, the eight-variant table
in §4.1 of that document, and the evolutionary Pareto front in §6 whose
candidates were separated by 0.04.

Lambda_c wanders between 398 and 519 across repeats of one structure, so 441,
491, 503 and the classical 440 are all the same number here.

The one real difference is in the **variance**: gating the lift-up term by
(1-gamma) cuts the spread of the total from 1.10 to 0.35 and of the
pre-transitional k error from 0.21 to 0.05. The ungated landscape has bad
basins -- one seed in four landed at 13.2 -- and gating removes them. That is
a reason to prefer the gated form, but it is about how well-posed the
calibration is, not about fit quality.

What to do about it: report structures with repeats and error bars, not single
fits; raise the sample budget until the spread is below the gaps being ranked;
or replace random search plus local refinement with something that actually
converges. Until one of those happens, structural conclusions have to be
carried by derivation and a-priori DNS measurement rather than by score.

### 4.17 The streak-energy deficit was partly a search bound I imposed
`C_s` sets where the length cap `C_s√k/ω` takes over from the wall distance
`y` in `ℓ_s = min(y, C_s√k/ω)`. It was bounded at `(0.05, 2.0)`, which put the
cap near 0.5 -- so it bound **everywhere** in the pre-transitional layer and
`ν_L` was proportional to `k` by construction. The mixing-length form gives the
correct `√k·y` scaling on its wall-distance branch; the bound simply prevented
the fit from ever reaching it.

Widening to `(0.05, 50)` and refitting (four seeds per variant,
`results/fit-noise.json`):

| variant | C_s bound | mean total | sd | mean k err (pre) |
|---|---|---:|---:|---:|
| mixing | 0.05–2.0 | 11.36 | 1.10 | 1.088 |
| mixing | 0.05–50 | 11.56 | **1.64** | **0.778** |
| mixing+gate | 0.05–2.0 | 11.16 | 0.35 | 0.925 |
| mixing+gate | 0.05–50 | 11.34 | **2.09** | 1.010 |

Three things, and the second is annoying:

1. **The streak energy does improve.** For the ungated mixing form the
   pre-transitional k error drops from 1.088 to 0.778, and the fits that go to
   the wall-distance branch choose `C_s` between 8 and 42 -- far outside the
   old bound. The best single run of the whole investigation appeared here:
   total 9.293 with c_f rel RMS 0.030 and k error 0.693, at `C_s = 23`.
2. **The variance gets much worse** -- `mixing+gate` goes from sd 0.35 to 2.09.
   Random search over a wider range samples it more thinly, so it finds the
   good basin sometimes and a bad one otherwise: within one variant the seeds
   span 9.29 to 14.81. The mean does not improve; only the best case does.
3. **The mean is unchanged within noise**, so this is *not* evidence that the
   wide bound is better. It is evidence that the good solutions live outside
   the old bound and that random search cannot reliably find them.

The correct reading is that the inner optimiser, not the model form, is now
the binding constraint -- consistent with §4.14. Narrowing the bound again
would hide the problem rather than fix it.

### 4.15 The mesh snapshot is a stub that produces an empty file **[PB]**
`scripts/save-mesh-snapshot.sh` does not render anything. It touches
`case.foam`, prints "Manual step: generate ...", and then **touches the output
PNG**, so `figures/rans-mesh-snapshot-isometric.png` is a 0-byte file that the
pipeline reports as successfully produced. That is worse than having no stage:
`calkit run` goes green and the artifact is empty. The stage also runs in
`_system`, so even once it does render, it would depend on whatever ParaView
happens to be on the host.

Pete's suggestion, to do later: make it a real stage with a ParaView Docker
environment (`pvpython` in a container, driven by a checked-in Python script
that opens `case.foam`, sets the isometric camera and writes the PNG), so the
figure is generated rather than pasted in. Two things to settle when we get
there:

- the mesh is now a declared output of `mesh-independence` (§4.16), so the
  input side is already reproducible;
- the render needs the mesh only, not a solution, so it can depend on
  `constant/polyMesh` alone and stay cheap.

Until then the stage should probably **fail loudly** rather than emit an empty
PNG, so that nobody mistakes the placeholder for a figure.

### 4.16 Two pipeline dependency bugs found by `calkit status`
Both were warnings about stage inputs containing Git-ignored files.

**`build-turbulence-lib` invalidated itself.** It takes `sim/newModel/src` as
an input, and wmake writes `lnInclude/` and `Make/<platform>/` into that same
directory while compiling, so every build changed the hash of its own
dependency. The platform directory names are machine-specific
(`darwin64Clang...` against `linuxARM64Gcc...`), so the stage also showed stale
on any machine that had not built it. Fixed with `.dvcignore` patterns that
exclude the build artifacts while keeping `Make/files` and `Make/options`,
which are real sources.

**`save-mesh-snapshot-isometric` depended on a file no stage produced.** It
reads `sim/cases/k-epsilon-ny-40/constant/polyMesh`, but `mesh-independence`
declared only `postProcessing` as an output, and `sim/cases` is Git-ignored --
so on a fresh clone the mesh does not exist and the stage cannot run. The mesh
is a genuine product of blockMesh, so it is now a declared output of the
parameterised stage for every (turbulence, ny) combination.

Worth noting that `.dvcignore` was the right tool for the first and the wrong
one for the second: blanket-ignoring `sim/cases` would have silenced the
warning while breaking every stage whose *outputs* live there.

### 4.12 Split the model library per closure
`src/Make` builds one `libransFromDns.so` containing both models, so editing
one invalidates simulations using the other.

---

## 5. Bigger open directions

- **A second flow.** Everything here is one DNS at one freestream turbulence
  level. The threshold Λ_c should depend on Tu; we cannot see that from one
  case. This is the single most valuable next step for credibility.
- **Transition length, not just onset.** Cgam railed at its bound, meaning the
  optimiser wants a near-instant switch. Onset is set by where Re_v crosses
  Λ_c; the *length* is then whatever Re_v growth gives. Real bypass transition
  has a breakdown length set by streak dynamics. A γ-destruction term or a
  smoother response is the natural fix.
- **Reformulate γ as coherence.** Given §1.4, γ is normalised R_uv. Writing the
  model in those terms would make it physically interpretable and give a direct
  a-priori target from DNS.
- **Evolutionary structure search over the entropy/coherence drivers.** The
  grammar already exposes `Hn` and `Hdef`; the search has not yet been run with
  a closure that carries H.

---

## 6. Evolutionary structure search: first run

53 structures evaluated (6 generations, population 14). Pareto front:

| terms | total | structure |
|---:|---:|---|
| 1 | 0.5508 | `sqrtKOverY \| linear \| rectify(Re_ks)` |
| 2 | 0.5128 | `sqrtKOverY \| linear \| rectify(Re_k)*rectify(Re_ks)` |

**Every one of the top eight structures uses `rectify`** — the search could
have chosen `power`, `inverse`, `tanh` or `softclip` for any term, and chose
the hard clip every time. That is independent support for the central
hypothesis from a procedure that was free to reject it, and it is the part of
this run worth keeping: it is a statement about which operators survive at all,
not about the ordering of scores.

The **ordering** is not usable. The Pareto entries differ by 0.04 while a
repeat of the same fit moves by up to 1.1 (§4.14), so this front cannot rank
its own candidates, and the comparison against the hand-designed model
(0.5128 against 0.5555) is meaningless.

Two caveats, both real:

- **The search stagnated immediately.** The best structure was in the initial
  random population and five generations of mutation and crossover improved
  nothing. With 53 structures and a population of 14 this is under-powered;
  the result is a sanity check, not a thorough exploration.
- **The comparison to our hand-designed model is not fair.** Each evolved
  structure got ~35 coefficient samples plus a short refinement, against 500
  plus four refinements for the k-ω-γ fit. The evolved 0.5128 versus our
  0.5555 is therefore suggestive, not conclusive.

Worth noting that the search prefers `Re_k` and a streak Reynolds number over
the `Re_v` we selected, and prefers a *linear* (1−γ) shape over a logistic one,
meaning no self-excitation is needed. Both are worth following up with a
properly powered run.
