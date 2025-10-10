# Complementarity Reduction for Sparse Trade Patterns

## Problem Statement

**Original Issue**: All firms exporting to ALL countries (0% sparsity)
- Unrealistic: Real firms typically export to 5-25 countries
- Caused by strong complementarities in the QuadSupermodular framework

## Key Insight (User)

> "You need to reduce the complementarities if they export to all countries"

This was **exactly correct**! 🎯

## Solution

### Understanding Complementarities in QuadSupermodular

The utility function is:
```
U(bundle) = Σ_j θ_mod · x_j + Σ_j Σ_j' θ_quad · x_j · x_j' + ε
```

Where:
- **θ_modular**: Country-specific effects (GDP, trade openness, etc.)
- **θ_quadratic**: Pairwise effects creating **complementarities**

**Positive θ_quadratic** means: "If I export to country A, then country B becomes more attractive"

This creates a **snowball effect**:
1. Start exporting to high-value countries
2. This makes other countries more attractive (complementarity)
3. Export to more countries
4. Repeat → Export to ALL countries

### The Fix

**Reduce θ_quadratic to near-zero** → Minimal complementarities → Sparse solutions

## Experiments

### Experiment 1: Original (Dense)
```python
θ_quadratic = [0.5, 0.5, 0.5]  # Strong complementarities
```
**Result**: 
- 0% sparsity
- 50/50 partners average
- All firms → all countries ❌

### Experiment 2: Weak Complementarities
```python
θ_quadratic = [0.0, 0.05, 0.05]
```
**Result**:
- 5% sparsity
- 47/50 partners average
- Still too dense ⚠️

### Experiment 3: Minimal Complementarities ✅
```python
θ_quadratic = [0.0, 0.001, 0.001]  # Near zero!
θ_modular = [3.0, 1.0, 0.5, 0.5]     # Stronger modular effects
σ = 0.8                               # Lower noise
```
**Result**:
- **67.2% sparsity** ✅
- **16.4 partners average** ✅
- Range: 8-26 (realistic!) ✅
- 100% firms in realistic range ✅

## Key Learnings

### 1. QuadSupermodular Constraint
- **Must have θ_quadratic ≥ 0** (solver requirement)
- Cannot use negative values to create substitutes
- Solution: Make θ_quad **very close to zero**

### 2. Sparsity-Complementarity Tradeoff
| Complementarity | Pattern | Sparsity | Realism |
|----------------|---------|----------|---------|
| Strong (θ=0.5) | Dense, predictable | 0% | ❌ Unrealistic |
| Weak (θ=0.05) | Semi-dense | 5% | ⚠️ Still too dense |
| Minimal (θ≈0) | Sparse, selective | 67% | ✅ Realistic |

### 3. Balance Modular vs Quadratic
For realistic sparse patterns:
- **Weak quadratic effects** (θ ≈ 0): Prevent snowball
- **Strong modular effects** (θ > 1): Drive systematic selection
- **Low error variance** (σ < 1): Reduce noise

## Mechanism

### With Strong Complementarities (θ_quad = 0.5):
```
Export to USA (high GDP) → +utility
  → Common language with UK → +utility (complementarity!)
    → Common region with Canada → +utility (complementarity!)
      → Continue exporting everywhere...
Result: ALL destinations selected
```

### With Minimal Complementarities (θ_quad ≈ 0):
```
Export to USA (high GDP) → +utility
  → Common language with UK → +0.001 utility (negligible!)
    → Common region with Canada → +0.001 utility (negligible!)
      → Selection based ONLY on country characteristics
Result: Selective exports (SPARSE!)
```

## Optimal Parameters

For **realistic sparse trade patterns**:

```bash
mpirun -n 4 python simulate.py \
  --num_firms 5000 \
  --sigma 0.8 \
  --theta_agent 0.3 \
  --theta_modular 3.0 1.0 0.5 0.5 \
  --theta_quadratic 0.0 0.001 0.001 \
  --exclude_home
```

**Produces**:
- 67% sparsity
- 16 partners average
- 8-26 partner range
- Realistic trade patterns

## Remaining Issue

**Weak GDP correlation (0.15, p=0.31)**

While sparsity is excellent, the gravity effect is weak. This is because:
- Minimal complementarities → mostly independent decisions
- High variance → noise dominates systematic patterns
- Standardized features → compressed differences

**Options**:
1. Accept tradeoff (sparsity achieved, some gravity present)
2. Increase θ_GDP further (risk: reduce sparsity)
3. Use log(GDP) instead of GDP (better captures size distribution)

## Conclusion

✅ **User insight confirmed**: Reducing complementarities solves the "export to all" problem

✅ **Solution validated**: θ_quadratic ≈ 0 creates realistic sparse patterns

✅ **Ready for next steps**: 
- Parameter estimation
- Counterfactual analysis
- Policy evaluation

## Visualization

See `datasets/complementarity_reduction_impact.png` for full comparison:
- Partner distribution before/after
- Sparsity improvement (5% → 67%)
- Bilateral flow patterns
- Summary statistics

---

**Key Takeaway**: In QuadSupermodular models, small changes in θ_quadratic have HUGE effects on sparsity. For realistic patterns, keep θ_quad ≈ 0 and drive selection via strong modular effects.

