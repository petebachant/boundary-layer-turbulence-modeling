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

### 3.1 Alignment of the anisotropy tensor with mean strain — **highest value**
An eddy-viscosity closure is exact only if b_ij ∝ S_ij. Measuring the angle
between the eigenframes of b_ij and S_ij quantifies **exactly how much
structure a scalar ν_t cannot represent**, and where. Directly actionable for
closure design: it says whether we need a nonlinear/tensorial model and at
which x.
*Needs:* only data we already have. **Do this next.**

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

### 4.5 Split the model library per closure
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
