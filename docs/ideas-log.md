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
from 4.2 % to 62 % (no better than k-ε), while removing every other term
changes the score in the third decimal. Structure parameter
a₁ = −u'v'/2k saturates at **0.137** and pins there. See
[clipping-closure.md §4.4](clipping-closure.md).

### 1.2 Vorticity Reynolds number as the threshold variable
Re_v = y²Ω/ν. Beat every alternative driver tested (Re_k, streak Reynolds
number, shear-weighted streak energy). Fitted threshold **Λ_c = 441**, against
the classical ~440 — and the classical value scores within 0.001 of the fitted
one, so no bespoke constant is needed \cite{Menter2006}.

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
the DNS value. Fixed by scaling on the wall-normal (active) amplitude instead.
Later ablation showed the lift-up term earns nothing anyway and can be deleted.

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

### 4.8 Split the model library per closure
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
hypothesis from a procedure that was free to reject it.

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
