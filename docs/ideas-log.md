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
transported γ in our closure is best understood as *normalized coherence*, not
an abstract "activation".

### 1.5 Meaning inside randomness: total correlation **[PB]**
Shannon entropy is the wrong tool — it is *maximized* by random letters. The
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
behavior, and measured rather than assumed.

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
growth, which is the correct non-modal behavior, and k = 0 stays a fixed
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
the mismatch is penalized in the objective, so the decay exponent
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
baselines, which all carry the same mild favorable pressure gradient.

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

Both buy real streak energy and both are heavily penalized for it.

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

The correct reading is that the inner optimizer, not the model form, is now
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
parameterized stage for every (turbulence, ny) combination.

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
  optimizer wants a near-instant switch. Onset is set by where Re_v crosses
  Λ_c; the *length* is then whatever Re_v growth gives. Real bypass transition
  has a breakdown length set by streak dynamics. A γ-destruction term or a
  smoother response is the natural fix.
- **Reformulate γ as coherence.** Given §1.4, γ is normalized R_uv. Writing the
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

### 6.1 The evolutionary search does not reproduce — **bug, 2026-08-28**

Running `evolve-closure` twice on the same machine, in the same environment,
with the same seed and the same arguments gives different answers:

| run | structures evaluated | best total |
|---|---:|---:|
| 2 | 47 | 9.121 |
| 3 | 50 | 6.718 |

A **26 % swing in the reported best**, and a different number of structures
explored. This is a stage of a pipeline whose entire purpose is
reproducibility, so it is a defect rather than a curiosity.

It is not an unseeded random number generator. Every stochastic step is
explicitly seeded — `evolve` draws from `default_rng(seed)`, the per-candidate
seed is drawn from that stream, `_fit_worker` is seeded and sequential, and
`Candidate.key()` sorts its terms rather than relying on set iteration order.
`ex.map` preserves result order.

What is left is the loop's sensitivity to the last bits of a float. The
archive is ranked by fitness each generation and the top quarter become elites;
a difference of 1e-15 between two near-tied candidates flips the elite set,
which changes the next population, which changes how many structures are new,
which advances the shared RNG stream by a different number of draws. From there
the two runs have nothing in common. Candidate sources for that initial
difference, in order of likelihood: thread-count-dependent reductions inside
the worker processes (`workers` is derived from `os.cpu_count()`), and the
bare `except Exception: return math.inf` in `_score_one`, which converts any
transient failure into a score that reorders the ranking.

**This is the fourth independent demonstration that this search cannot rank its
own candidates**, after the seed-to-seed spread of §4.14, the 0.04-vs-1.1 gap
in §6 above, and the fact that migrating numpy 1.20 → 2.5 (a ~1e-8 change in
the objective) moved the reported best from 12.28 to 9.12. The conclusion in
§6 — that the Pareto *ordering* is not usable and only the operator preference
is — is not merely still true, it is now the only defensible reading.

**To fix, in order:** pin `workers` and the BLAS thread count rather than
deriving them from the machine; break ranking ties on `cand.key()` so equal
fitness cannot reorder; narrow the bare `except`; and report the whole archive
rather than a single "best", since the best is the least stable thing in it.

Worth noting that the search prefers `Re_k` and a streak Reynolds number over
the `Re_v` we selected, and prefers a *linear* (1−γ) shape over a logistic one,
meaning no self-excitation is needed. Both are worth following up with a
properly powered run.

---

## 7. Fitting methodology: what we search, and how

### 7.1 A term library for the *momentum* equation, dimensions unconstrained **[PB]**

**Status (2026-08-29): tried, in the fast tier, as `pypkg/momentum_library.py`
and the `fit-momentum-library` / `fit-momentum-library-multi` stages.** Six
streamwise force terms built from y-derivatives of U, k and nu_t, on top of
Launder--Sharma, with dimensionless coefficients fitted by Bayesian
optimization (`pypkg/bayesopt.py`) of the a-posteriori bench score, once on
the plate alone and once on four cases jointly. Two deliberate departures
from the idea as first posed: coefficients are *not* left dimensional
(second paragraph below says why), and every term is Galilean invariant and
vanishes with k, which the two terms once hard-coded into
`sim/newModel/solver/UEqn.H` were not -- they are now off by default. The
results are in `results/momentum-library*.json` and the findings keys
`momentum_library_*`. The original status note follows for the record.

**Status as of 2026-08-28: not tried. It is not what §4-adjacent work did,
and the distinction matters.** Three things in this repo look like it and
are not:

| what exists | target | dimensional freedom |
|---|---|---|
| `scripts/regress-pde-terms.py` | closure quantities `-<u'v'>/k` and the ε-budget residual | dimensionless groups only, 8 columns for stress and 6 for ε |
| `pypkg/grammar.py` | the γ activation source | "dimensionally-consistent ... by construction", stated as a design virtue |
| `sim/evolve_pde_structure.py` term multipliers | existing k-ε terms | multiplies terms already present; cannot add one that is not |

So the actual idea — assemble a library of candidate terms for the **mean
momentum equation itself**, let the coefficients carry whatever dimensions they
need, and regress them (PDE-FIND / SINDy applied to RANS momentum rather than
to a closure variable) — is genuinely unexplored. It is strictly more general
than the eddy-viscosity ansatz, which it contains as one column.

**Two things to know before spending time on it.**

*It will fit beautifully and almost certainly not transfer.* We already have
the controlled version of this experiment. The closure-level library, which is
far more constrained, reaches R² = 0.953 in-sample and **−472** predicting the
Jiménez ZPG boundary layer, with upstream and downstream fits disagreeing on
the sign of every coefficient (`coeff_disagreement = 1.0`). A momentum-equation
library has *more* freedom, so it will score higher in-sample and transfer
worse. That is not a reason to skip it — it is a reason to make out-of-sample
transfer the primary reported metric from the first run, never in-sample R².

*"Regardless of dimensions" has a sharp consequence worth exploiting.* If the
coefficients carry dimensions, they cannot be universal across cases with
different ν, U_e and δ — a coefficient fitted at ν = 1.25 × 10⁻³ on the JHTDB
plate is not transferable to a channel at ν = 3.5 × 10⁻⁴ *by construction*,
before any physics enters. That is a checkable prediction, and the multi-case
harness ([roadmap §2](roadmap.md)) would show it in one run. The useful version is therefore to fit
dimensional coefficients **and** their non-dimensionalisation with explicit
local scales, and report how much of the non-transfer is trivial units and how
much is real. Skipping that decomposition means a guaranteed negative result
for an uninteresting reason.

### 7.2 Bayesian optimization of coefficients in the forward model **[PB]**

**PB's own framing, and the answer to it.** The question was whether BO on the
forward model beats least squares on the inverse model when we already have
forward-model results, with the suspicion that it may not unless new transport
equations are being added. The suspicion is right about the optimization and
wrong about where the value is.

*As an optimizer, the gain is modest and we can bound it.* What we do now is
`random_search` + `refine` in `pypkg/search.py` — no surrogate, no
acquisition function. BO would find the same basin in fewer forward solves. But
§4.14 measured the objective noise: repeating the same fit moves the score by
up to **1.1**, while the structures being ranked differ by **0.04**. BO reduces
the number of samples needed; it does not resolve a degeneracy that is 25×
wider than the differences being resolved. Faster convergence to an
unidentified optimum is not progress.

*The value is the posterior, not the optimum.* A Bayesian treatment returns a
**credible interval per coefficient**, which is precisely the quantity this
project keeps needing and does not have. It converts the paper's qualitative
claim — "the coefficients do not transfer" — into a quantitative one: *this
coefficient is not identified by this case, to within N decades*. Λ_c fitting
to 491 against a classical 440 (§1.2) is a much weaker statement than Λ_c
having a posterior spanning 300–900, and the second is the one a reader can
act on. Cgam railing at its bound (§5) is the same story: a posterior would say
whether the data constrains it at all.

*Where it compounds: one posterior per case.* Run the same Bayesian fit
independently on each case in the harness ([roadmap §2](roadmap.md)) and the **overlap of the
per-case posteriors is a direct, calibrated measure of transferability** —
disjoint posteriors mean the constant is a per-case fit, and that is a
measurement rather than an assertion. This is the strongest version of the
paper's central argument, and it needs BO only as machinery.

*PB is right that new transport equations are where it becomes necessary
rather than merely nice.* Random search scales badly in dimension; adding an
equation adds coefficients, and at that point the sample efficiency stops being
a convenience.

*Not either/or.* The inverse (a-priori) least-squares fit is a cheap prior for
the forward (a-posteriori) BO rather than a competitor to it. That the two
disagree is itself one of this project's results and should be preserved, not
optimized away.

**turbo-RANS [PB, 2026-08-30].** McConkey's `turbo-rans`
\citep{McConkey2024} is the ready-made version of the OpenFOAM-tier half
of this: Bayesian optimization of a model's coefficients with the solver
in the loop, per case. When the Tier-2 cases are running routinely, the
multi-case objective of `fit-momentum-library` could be handed to it
rather than to `pypkg/bayesopt.py`, and its per-case optima are exactly
the "one posterior per case" measurement above.

### 7.3 One closure definition for both tiers **[PB, 2026-08-30]**

**Status: idea.** Today a closure exists twice: a Python class implementing
`initialize`/`eddy_viscosity`/`advance` for the screening solver, and an
OpenFOAM `RASModel` in C++ for the confirmation tier, and the cross-tier
consistency check exists because the two can disagree. A contributor who
wants to try a model in the bench has to write it twice, in two idioms, and
the fork model (roadmap §2.4) inherits that cost. The ask is a single
definition both solvers read.

**What the definition has to carry.** Looking at every closure in
`pypkg/closures.py` and `sim/newModel/src`, a transport-equation closure is
fully specified by: the transported scalars and their wall and free-stream
conditions; for each, a diffusivity (`nu + nu_t/sigma`), an explicit source
and an implicit sink (which is exactly the `source_ex`/`source_im` split
`march_scalar` takes, and the `fvm::Su`/`fvm::Sp` split OpenFOAM's
`fvScalarMatrix` takes); the eddy-viscosity expression; and a small set of
non-smooth operators — `max`, `min`, a rectifier, a Heaviside gate — plus
wall distance, `|grad U|`, the strain and rotation invariants. That is a
small algebra, and it is the same algebra on both sides.

**Three ways to write it, in order of how much cleverness they need.**

1. *A declarative spec, e.g. YAML with expressions as strings*
   (`nut: "gamma*k/omega"`, `k: {diffusivity: "nu + nut/sigma_k", source:
   "P_k", sink: "betaStar*omega"}`). Parsed once with SymPy, checked for
   dimensional consistency (free, given units on the fields), then emitted
   as (a) a numpy `Closure` subclass by direct evaluation of the
   expressions on the grid and (b) an OpenFOAM `RASModel` by a template:
   each expression becomes a `volScalarField` line, each transport equation
   a `fvScalarMatrix` with `fvm::div(phi, k) - fvm::laplacian(D, k) ==
   Su - fvm::Sp(Sp, k)`. This is the **OpenSBLI** pattern
   \citep{Lusher2021} — governing equations in a compact symbolic form,
   expanded and discretized by SymPy into generated code — and the
   **UFL** pattern \citep{Alnaes2014} for weak forms; both exist and work
   at scale, which is the evidence this is tractable.
2. *SymPy directly as the interchange.* Same as (1) but the source is
   Python; less friendly to non-programmers, no parsing.
3. *TeX as the source.* Attractive because the paper already has to
   contain the equations, and a reader could see exactly what runs. But
   parsing TeX into an AST is fragile (macros, implicit multiplication,
   ambiguous sub/superscripts), and every existing system that tried it
   ended up defining a restricted grammar that is a DSL wearing TeX
   clothing. The **inverse** is cheap and gives the same benefit: generate
   the TeX *from* the spec, and inject it into the paper through the same
   provenance machinery as the numbers (`\ckinput{generated-closure.tex}`).
   Then the model equations in the paper cannot differ from the model that
   ran, which is the cohesion we actually want.

**Recommendation.** (1), with (3)'s inverse as the paper-facing view. The
cross-tier consistency check becomes a test of the code generator rather
than of two hand-written implementations; the `calibrated_on`,
coefficients, and bounds live in the same spec so the bench can register a
closure from the file alone; and a fork "changes the model under test" by
changing one file. Restrictions to accept up front: local closures only
(no non-local free-stream lookups, which `ClipKOmegaGamma`'s lift-up term
has in one mode — that mode would not be expressible, which is a feature),
and eddy-viscosity or momentum-source form only until the tensor
`stress()` interface (§7.1, roadmap §2.3b) exists on the Python side.
Prior art for the discovery side of this — models proposed as algebraic
expressions and then run — includes CFD-driven symbolic identification
\citep{Zhao2020} and the LLM-driven variant AutoTurb
\citep{ZhangAutoTurb2024}; both would produce exactly the kind of spec (1)
consumes.

**ML closures in the same spec [PB, 2026-08-30].** A learned closure is
the same object with one expression node replaced by a model call, and the
spec should say so directly:

```yaml
nut: "gamma*k/omega"
anisotropy:
  model: closures/tbnn.onnx          # exported from any framework
  inputs: [S_norm, R_norm, Re_d, y_plus]   # invariants the spec computes
  outputs: [g1, g2, g3, g4]          # tensor-basis coefficients
```

ONNX is the interchange for the same reason the spec is: one artifact,
consumed by both tiers. On the Python side `onnxruntime` evaluates it on
the grid like any other expression. On the OpenFOAM side ONNX Runtime has a
C API that links into a `RASModel` the way the TensorFlow C API was linked
by \citet{Maulik2021} (the earliest working pattern for deploying a
trained network inside an OpenFOAM turbulence model) and the way the
ML-driven Reynolds-stress models of the `MachineLearningTurbulenceModels`
repository (Macedo, github.com/mthsmcd) inject learned stresses as source
terms. Either way the generated C++ is a fixed template — gather the input
fields, run the session cell by cell (or in one batched call per
iteration), scatter the outputs — so the codegen does not change with the
network. The lower-effort route on the OpenFOAM side is PythonFOAM
\citep{Maulik2022}, which embeds a Python interpreter in the solver, so
the *same* Python evaluation used by the screening tier runs in-situ; the
ONNX C API is the deployment-grade version of the same call. Two rules the spec should enforce because the literature keeps
relearning them: the network's inputs must be the invariants the spec
already defines (no raw velocities, so Galilean and rotational invariance
hold by construction, cf. \citet{Ling2016}); and its outputs must enter
through a form the solver can stabilize — coefficients of a tensor basis,
or a bounded multiplier on an existing term — rather than a bare stress
substituted into the momentum equation, which is the ill-conditioning of
\citet{WuXiaoPaterson2018}. With that, an ML closure is registered in the
bench from the file alone, is scored a posteriori on every case like any
other, and its `calibrated_on` is the training set it declares, so the
in-/out-of-sample split applies to learned models exactly as it does to
fitted coefficients.

**Two kinds of entrant, not one [PB, 2026-08-30].** PB's objection: an ML
model that only supplies a term in a PDE, with the rest solved
numerically, gives up the thing ML is good at — predicting the whole flow
field in one go. That is true, and the bench should accept both rather than
force a surrogate into a closure's clothing:

| entrant | what it supplies | what the case supplies | what "a posteriori" means |
|---|---|---|---|
| *closure* | a term (ν_t, a source, a stress) | the solver, the mesh, the BCs | the PDE is solved with the term in it; stability is part of the score |
| *predictor* | the fields, from a case description | the description (geometry, Re, inlet profile, BCs) and the evaluation grid | there is no solve; consistency of the predicted fields with the equations is what stands in for stability |

Both are scored the same way — the solution against the case's declared
targets — and both declare `calibrated_on`, which for a predictor is its
training set, so the in-/out-of-sample split applies unchanged. What
differs is the interface: `run(closure)` for the first, `run(predictor)`
handing over the case description for the second, with the case's own
inputs (inlet profile, `Ue(x)`, seed) being exactly the description a
predictor needs. The Closure Challenge \citep{McConkey2026} and ML4CFD
\citep{Yagoubi2024} score only the second kind; the bench today scores only
the first. Scoring both on the same cases is the comparison nobody has
published: a surrogate that has seen a hundred airfoils against a closure
with eight coefficients, on a flow neither was fitted to. Two consequences
to state in the paper if this is built: a predictor cannot be run on a
case whose geometry it has never seen unless its inputs are the local
description (which is a closure again, at the level of a point), so the
geometry-generalization test is *harder* for predictors, not easier; and a
predictor's fields can be checked against the momentum equation on the
grid — the residual of the equations it did not solve is a diagnostic the
bench can compute deterministically for every predictor entry, and it is
the surrogate analogue of "did the solve converge".

### 7.4 Multi-fidelity optimization: what the tiers are actually for **[PB, 2026-08-30]**

PB's reframing: the fast tier's value is not benchmarking, it is
optimization and evolution at low expense. That is a cleaner statement of
what this project already does implicitly — every coefficient search,
ablation and evolutionary run happens in the fast tier, and the OpenFOAM
tier is where a result is confirmed — and it has a name in the
optimization literature: **multi-fidelity Bayesian optimization**, where
fidelity is a controllable input, the surrogate models the correlation
between fidelities, and a cost-aware acquisition (knowledge gradient in
BoTorch's MFKG, or the cheaper multi-task GP) decides whether the next
evaluation is worth a cheap solve or an expensive one.

Three things follow.

**Measured (2026-08-30, `measure-tier-discrepancy` stage):** on the three
closures that ran on the plate in both tiers, the tier-2/tier-1
skin-friction-error ratio is 3.7 for the fitted clipping closure, 1.8 for
Launder--Sharma and 0.75 for laminar — geometric mean 1.7 with a log-sd of
0.65 — and the two tiers *do not agree on the ranking* (tier 1 orders
clip < LS < laminar, tier 2 orders clip < laminar < LS). Three points is
not a surrogate, but the sign is already the important one: the screening
bias is model-dependent, largest for the model whose coefficients were
fitted in the screening tier, which is precisely the situation
multi-fidelity optimization exists to handle and a warning against
trusting a tier-1 optimum without a tier-2 check.

**Confirmed on eight cases (2026-08-30, `run-benchmark-openfoam`):** with
the Closure Challenge hills and ducts run in OpenFOAM, the tier rankings
of the closures common to both are different — the fast tier's best model
out of sample, Launder–Sharma, diverges on two separated hills and is
unstable on a third and on the widest duct, while the transition models
the fast tier could not run lead. The clip closure's fast-tier score on
the plate (its calibration case) says nothing about its ducts, where it
relaminarizes. See `results/benchmark-openfoam.json` and Q16.

**The tier correlation is already measured.** The paper quantifies the
parabolic solver as ~3x optimistic on skin friction relative to the
elliptic solve for the same closure and coefficients, with the bias
systematic. That is the fidelity discrepancy model an MFBO needs, and the
first experiment is to fit it properly: every closure that has run in both
tiers on the plate gives a (cheap score, expensive score) pair, and the
regression of one on the other — with its spread — says how far a tier-1
optimum can be trusted before a tier-2 evaluation is worth its cost. Note
the nuance: the channel and temporal-mixing-layer cases are *exact*
formulations solved in the fast tier, so for those flows there is no
fidelity gap and no reason ever to prefer an OpenFOAM evaluation; the
parabolic cases (plate, ZPG, wing section) are the genuinely low fidelity.

**The optimization loop should be tier-aware, not tier-blind.** The
momentum-library fit (§7.1) ran entirely in the fast tier. The upgrade:
optimize in tier 1 with the discrepancy model correcting the objective,
spend tier-2 evaluations only where the surrogate's tier-2 posterior is
uncertain enough to change the ranking, and report the final number from
tier 2 always. `pypkg/bayesopt.py` stays the dependency-free single-
fidelity tool; MFBO needs BoTorch (a torch stack), which belongs in its
own environment so it cannot invalidate the compute lock — same isolation
argument as `py-jhtdb`.

**The user-facing use case.** Combined with the declarative closure spec
(§7.3), this becomes a service: a user writes the functional form — extra
transport equations, momentum source terms, coefficient bounds — and the
bench (1) generates the tier-1 implementation, (2) pre-optimizes the
coefficients against the chosen cases at fast-tier cost, (3) generates
the OpenFOAM model with the optimized coefficients as the starting point,
and (4) benchmarks in tier 2, which is the leaderboard of record. Nobody
hand-tunes coefficients in a solver that costs minutes per evaluation
when a second-per-evaluation solver, with a measured bias, can do the
first 95 % of the search. turbo-RANS (§7.2 note) is the tier-2-only
version of step (4)'s inner loop; this is the two-fidelity version.

### 7.5 Discovering the k and omega transport equations from the DNS **[PB, 2026-09-24]**

**Status (2026-09-25): tried a priori; no sparse shared equation.** Stages
`build-transport-targets` and `fit-transport-equations`; numbers in
`results/transport-targets-summary.json` and
`results/transport-equation-fit.json`, and the answers in `calkit.yaml`.
Three things came out, in the order they were found:

1. *The two omegas are not one.* Even in the log layer the effective C_mu
   is well below 0.09 and falls with Reynolds number in the channel, so
   eps/(beta* k) and k/nu_t agree at a minority of points.
2. *Regressing the omega equation as written fits noise.* In every family
   it is a near-local balance: advection and molecular diffusion, over
   omega^2, are two orders smaller than the sources, and SST's own
   coefficients score R^2 from -0.1 to -4000 on them. The fit was redone in
   the implicit (SINDy-PI) form with the destruction fixed, scored as the
   fraction of destruction left unexplained.
3. *The stress-based omega is a trap.* S/omega_frozen = -uv/k exactly, so
   the library's S^2/omega^2 and S/omega columns are the structure
   parameter. Fits to that target keep every term and transfer no better
   than those two columns alone, which means they found that -uv/k is
   nearly constant, not a transport equation; on the transitional plate,
   where it is not constant, the extra terms make transfer worse. A guard
   column set (`A1_TERMS`) now makes this visible in every run. Only the
   dissipation-based omega leaves the other terms anything to explain,
   and there they help modestly while keeping 10 of 11 terms.

What would change the conclusion: an omega target on the transitional
plate with a real dissipation (the same JHTDB gradient pull as §7.6), and a
fit scored inside a solver rather than on DNS fields.

**The idea.** Keep the eddy viscosity, nu_t = k/omega, but stop assuming
the transport equations for k and omega. Build their fields from every DNS
case, build a library of candidate terms starting from SST's (production,
destruction, diffusion, cross-diffusion, blending) and adding more, and
solve for the coefficients by sparse regression. This is SINDy
\cite{Brunton2016} with control \cite{BruntonProctorKutz2016c}: k and omega
are the state, and the mean-flow quantities (S, Omega, Re_v, wall distance,
free-stream turbulence) are the inputs acting on it.

**Prior art, and what would be new.** The k-corrective-frozen approach of
SpaRTA \cite{Schmelzer2020} adds a regressed correction to the k equation
on three separated flows, and \cite{BeethamCapecelatro2020} embed form
invariance; field inversion has corrected transition models too. Our own
`regress-pde-terms` stage fits the shear stress and the k-budget residual on
one flow; the stress fit reaches R^2 = 0.953 in sample and -472 on the
Jiménez layer. What is new here: every
DNS family at once, term *stability* rather than fit as the output
(ensemble-SINDy inclusion probabilities per left-out family), and every
nominated equation scored a posteriori on held-out flows in the bench.

**Two things that decide whether it works.**
1. *omega is not in the DNS.* Two targets, both fitted: omega_eps =
   eps/(beta* k), where eps is available (NACA 4412 budgets, Jiménez
   pseudo-dissipation from vorticity rms), and the frozen omega = k/nu_t
   with nu_t the Boussinesq projection of the DNS stress, which exists on
   every case and makes k/omega consistent with the momentum equation. Their
   ratio is C_mu,eff/0.09, itself a diagnostic of where one omega cannot
   serve both roles.
2. *Langtry-Menter's gamma and Re_theta_t are not observable.* The a-priori
   library starts from SST's k-omega terms and adds transition terms built
   from observables: the rectified Re_v activation (the ablation showed it
   load-bearing), S/omega, Omega/omega, free-stream intensity, and spectral
   entropy once §7.6 has it.

**Success criterion, fixed before the fit.** Beat SST-LM out of sample on
the OpenFOAM tier, or at least beat SST in both the boundary-layer and the
separated/secondary-flow families. Otherwise the answer is that no sparse
transport equation is shared across these flows.

**Stages.** `build-transport-targets` (fields, both omega targets, and what
each case can support), then the ensemble fit, then promotion to the bench.
Questions in `calkit.yaml` track each.

### 7.6 Spectral entropy as a transported variable **[PB, 2026-09-24]**

**Status (2026-09-25): measured on the JHTDB plate; the gate as set fails,
the scale-consistent version is still interesting.** Stage
`analyze-spanwise-lines`, from 17 stations x 6 heights x 2048 spanwise
nodes x 8 snapshots pulled with `fetch-jhtdb-lines.py`; answers in
`calkit.yaml`. The pre-registered entropy, over the fixed span, is
monotone at no height and rises then falls, but it is confounded: the span
holds fewer energetic modes as delta_99 grows. Over bands of wavelength
scaled by delta_99 (exploratory) it rises through transition and then
levels off, a state-like shape, but it leads the dissipation coefficient
by well over a hundred units of x rather than moving with it. Harmonics of
the streak wavenumber peak after the broadening, not before, and the
entropy does not collapse on the turbulence Reynolds number downstream. So
a transported spectral width would carry something, the spectrum
broadening ahead of the cascade reaching equilibrium, but not in the form
the S*(Re_t) relaxation model assumed. The same pull gives the plate its
true dissipation: the effective C_mu is near zero before transition and
near the log-layer range after.

**The idea.** k carries no wavenumber. The normalized energy spectrum
p(kappa) = E(kappa)/k has an entropy S = -int p ln p dkappa, which is low when
energy sits in a few modes (pre-transitional streaks) and high when the
cascade has spread it (developed turbulence). A transported S would be an
omega-like variable carrying the *width* of the spectrum rather than one
scale: the non-equilibrium between spectral transfer and dissipation that
split-spectrum models carry with extra equations
\cite{HanjalicLaunderSchiestel1980}, and that our C_eps varies through
transition by a factor of several (§ What transfers, paper) says is there.

**Not the entropies already tried.** Thermodynamic entropy production at
constant temperature is dissipation over temperature, eps in other units
\cite{KockHerwig2004}. The component entropy of the energy partition is
non-monotone and hysteretic through transition, which is why
`EntropyKOmegaH` failed. Enstrophy models \cite{RobinsonHassan1998} carry
what omega already does.

**What it needs.** Instantaneous (y, z) planes from the JHTDB transitional
boundary layer at stations through transition, spanwise FFTs of u', and
S(x, y). A new token-gated pull, alongside the one the open question on a
structure-variable (Q, lambda_2) equation already needs.

**The gate.** S must rise monotonically through transition, unlike the
component entropy, and its rise should coincide with the swing in C_eps. If
it is non-monotone, the entropy route is closed.

### 7.7 Turbulence as an overdriven amplifier **[PB, 2026-09-25]**

**The picture.** A laminar layer driven past what it can carry smoothly
clips, and the clipped energy is redistributed into other wavenumbers,
the way an overdriven amplifier puts a sine wave's power into harmonics
and raises its spectral entropy.

**Where it is exact.** The Navier-Stokes nonlinearity is quadratic, so in
Fourier space modes interact only in triads, k = p + q: sum and difference
wavenumbers, i.e., intermodulation, including the second harmonic and the
k = 0 (DC) term, which is the mean-flow distortion. The nonlinear term
conserves energy, so the mixer is lossless.

**Where it needs amending.** Nothing in Navier-Stokes clips amplitude like a
hard limiter, and a square-law mixer makes even harmonics and DC rather
than a symmetric clipper's odd ones. Saturation comes from feedback: the
DC output flattens the mean profile, which lowers the shear that feeds the
fluctuations, so it is a mixer under automatic gain control. And laminar
flow has no Reynolds-number ceiling of its own; what is limited is the
disturbance amplitude, so onset should be input times gain reaching a
fixed headroom -- free-stream intensity times transient growth reaching
the streak amplitude at which secondary instability sets in
\cite{Andersson2001} -- which is why the classical threshold depends on
free-stream turbulence \cite{VanDriestBlumer1963}. "The flow holds itself at
the limit" is Malkus's marginal-stability idea \cite{Malkus1956}.

**A first look (not yet a stage).** On the JHTDB plate, Re_v,max =
max(y^2 S/nu) grows with the laminar layer, then holds near the transition
threshold from x ~ 200 to 350 while C_f starts to rise, and resumes growing
from the flattened turbulent profile; the streak rms overshoots to about
0.16 U_e at x = 350 and settles near 0.12. That is a clip and a saturation.

**The spectral-entropy equation.** From the spectral energy equation,
dE/dt = P + T - 2 nu k^2 E, with p = E/K and S = -int p ln p,

    dS/dt = -(1/K) int (ln p + S) (P + T - 2 nu k^2 E) dk

exactly. Energy put where p is below its typical share raises S: transfer
raises it, production at the streak scale and dissipation at high k lower
it. The transfer correlation is the unclosed term. In equilibrium the
spectrum's shape, and so S, is a function of Re_t = k^2/(nu eps) alone,
S*(Re_t), so a transported S carries new information only as a departure
from S*; the simplest closure is relaxation toward it over an eddy-turnover
time.

**Questions**, in `calkit.yaml`: whether the flow clips at a fixed local
Reynolds number; whether onset is input times gain reaching a fixed
headroom; whether saturating streaks put energy into harmonics first;
whether the mean-flow distortion acts as gain control; whether S collapses
on S*(Re_t) and transition is a departure from it; how C_mu,eff evolves
through transition; where the integral dissipation coefficient departs from
laminar; why kkL-omega never transitions on the plate; and whether there
is a sustaining threshold distinct from the trippable onset.

**Tripping, and an amendment to the amplifier.** An amplifier clips when
input times gain passes its rails, so a trip is a larger input that clips at
lower gain: the onset Reynolds number is not a property of the flow alone.
Where turbulence differs is that its instabilities are the gain and, once
running, it regenerates its own input (streaks, their instability,
vortices, lift-up, streaks again), which makes it an oscillator rather than
an amplifier. That gives two thresholds: onset, which tripping lowers, and
sustain, the minimum loop gain, which it cannot. Pipe flow cannot keep
turbulence below Re of about 2000 however it is started. The gap between
them is hysteresis, consistent with the hysteretic entropy-state relation
found through transition, and a single activation threshold cannot
represent it.

### 7.8 A transition threshold that knows the free stream **[2026-09-25]**

**Status: tried; it transfers.** Wu et al.'s bypass-transition DNS at five
inlet intensities \cite{Wu2026} (`data/wu-bypass-transition`) show onset set
by a disturbance reaching a fixed headroom, Re_x,t roughly proportional to
Tu^(-n) with n near 2, and the onset Re_v falling from about 800 at 1.5
percent to about 260 at 6 (`analyze-bypass-onset`). The classical 440, and
our fitted 418, are values for the JHTDB plate's intensity of about 2.5-3
percent, not constants.

A law in the *local* intensity at onset, fitted on Wu et al., failed to
predict the plate (`fit-threshold-law`); one in the *inlet* intensity,
tried after, came within 13 percent. Onset depends on how long the streaks
have been forced, which is the history Langtry-Menter carry by transporting
Re_theta_t in from the free stream.

The five flows are now bench cases (`pypkg/cases/wu_bypass.py`). With its
JHTDB threshold scaled as (Tu_in / Tu_in,JHTDB)^(-m), m fitted leaving each
flow out, the clipping closure's mean score on them falls from about 24 to
about 4, winning on four of five (`test-threshold-closure`); the plate is
unchanged by construction.

**Next.** A closure-native version: the inlet intensity is not a local
quantity, so carry it, e.g., as a transported free-stream intensity or as
the streak energy already in the model, and check it reproduces the
scaling without being told the inlet. Then Tier 2.
