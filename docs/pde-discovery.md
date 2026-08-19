# PDE Discovery via Turbulence Model Evolution

## Overview

The `evolve-model.py` optimizer implements **structural PDE discovery** through term multipliers. Rather than just tuning numerical coefficients, it can discover which physical terms in the k-ε equations are essential to match DNS data.

## Term Multiplier System

### Standard Coefficients
Traditional k-ε model coefficients that are optimized:
- **Cmu** (0.05–0.2): Eddy viscosity coefficient
- **C1** (1.0–2.0): Production coefficient in ε equation
- **C2** (1.5–2.5): Dissipation coefficient in ε equation  
- **C3** (−1.0–1.0): Compressibility correction in ε equation
- **sigmak** (0.5–2.0): Prandtl number for k diffusion
- **sigmaEps** (0.8–2.0): Prandtl number for ε diffusion

### Term Multipliers
Enable **structural discovery** by weighting major physical terms:
- **f1ProductionK** (0.0–2.0): Production term in k equation: `α·ρ·G`
- **f2DissipationK** (0.0–2.0): Dissipation term in k equation: `−α·ρ·ε²/k`
- **f1ProductionEps** (0.0–2.0): Production term in ε equation: `C1·α·ρ·G·Cmu·k`
- **f2DissipationEps** (0.0–2.0): Dissipation term in ε equation: `−C2·α·ρ·ε²/k`

## Discovery Mechanism

Each term multiplier ranges from **0 to 2**:
- **f ≈ 0**: Term is "turned off" → equation reduces
- **f ≈ 1**: Term is standard → standard k-ε model
- **f > 1**: Term is enhanced → stronger physical effect

### Modified Equations

**k equation:**
```
∂(αρk)/∂t + ∇·(αρUk) = f1ProductionK·αρG - f2DissipationK·αρε²/k + diffusion
```

**ε equation:**
```
∂(αρε)/∂t + ∇·(αρUε) = f1ProductionEps·C1αρG·Cmu·k - f2DissipationEps·C2·αρε²/k + diffusion
```

## Example Discoveries

The optimizer can discover:
1. **Suppressed production** → f1ProductionK ≈ 0 (mean flow doesn't generate turbulence in certain regions)
2. **Altered dissipation** → f2DissipationEps ≠ 1 (non-standard turbulence decay)
3. **Decoupled equations** → f1ProductionK ≠ f1ProductionEps (asymmetric interaction between k and ε)
4. **Required additions** → All f_i ≠ 0 (standard k-ε structure is necessary)

## Running Optimizer

```bash
cd sim
python evolve-model.py --iterations 20 --ny 40 --case-name evolved-model
```

Outputs:
- `results/model-evolution.json` — History of all 20 iterations with coefficients and loss
- `results/model-params.json` — Best-fit parameters (structure + coefficients)

## Interpretation

After optimization, inspect `model-params.json`:
```json
{
  "iteration": 15,
  "coeffs": {
    "Cmu": 0.085,
    "f1ProductionK": 0.95,
    "f2DissipationK": 1.1,
    "f1ProductionEps": 0.88,
    "f2DissipationEps": 1.25,
    ...
  },
  "loss": 0.00342
}
```

**What this tells you:**
- f1ProductionK ≈ 1 + small correction: k production is nearly standard
- f2DissipationEps > 1: ε dissipates faster than k-ε predicts
- f1ProductionEps < 1: ε production is suppressed relative to k production
- **Implication**: The ε transport is decoupled from k more than standard k-ε assumes

## Next Extensions

Possible enhancements to discover even more:
1. **Higher-order terms**: Add f_i for cubic nonlinearities in stress tensor
2. **Anisotropy**: Introduce coefficients for Reynolds stress anisotropy
3. **Rotation/stratification**: Term multipliers for rotation or density stratification effects
4. **Field-dependent coefficients**: Let f_i(S, Ω) depend on local strain/rotation

## Files Modified

- `sim/newModel/src/ransFromDns/ransFromDns.C` — Solver with term multipliers
- `sim/newModel/src/ransFromDns/ransFromDns.H` — Declares multiplier members
- `sim/constant/turbulenceProperties.template` — Template placeholders
- `sim/evolve-model.py` — Optimizer with multiplier bounds
- `sim/run.py` — Coefficient injection including multipliers

## References

The mechanism implements a form of **sparse identification** where coefficients approaching zero eliminate terms, allowing the optimizer to discover minimal models matching DNS data.
