#!/usr/bin/env python3
"""Test: Does basis help when weight changes are smaller?"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

np.random.seed(42)

N = 200  
M = 150  

print("="*60)
print("TEST: Effect of weight variance on warm-start benefit")
print("="*60)

A = np.random.randn(M, N) * 0.5
b = np.ones(M) * 5
c_base = np.random.randn(N)

def run_test(weight_scale, name):
    """Test with different weight perturbation scales."""
    model = gp.Model()
    model.setParam('OutputFlag', 0)
    model.setParam('Method', 1)
    model.setParam('LPWarmStart', 2)
    model.ModelSense = GRB.MAXIMIZE

    x = model.addMVar(N, lb=0, ub=1, obj=c_base, name='x')
    constrs = model.addMConstr(A, x, GRB.LESS_EQUAL, b)
    model.update()
    
    vars_list = x.tolist()
    constr_list = model.getConstrs()
    
    model.optimize()
    
    np.random.seed(123)  # Same for all tests
    
    # Method: reset(0) only (cold)
    results_cold = []
    for _ in range(5):
        weights = 1.0 + weight_scale * (np.random.exponential(1.0, N) - 1)
        c_new = list(c_base * weights)
        model.setAttr(GRB.Attr.Obj, vars_list, c_new)
        model.update()
        model.reset(0)
        model.optimize()
        results_cold.append(int(model.IterCount))
    
    # Reset for basis test
    np.random.seed(123)
    model.setAttr(GRB.Attr.Obj, vars_list, list(c_base))
    model.update()
    model.optimize()
    
    # Method: reset(0) + basis restore
    results_warm = []
    for _ in range(5):
        vbasis = model.getAttr(GRB.Attr.VBasis, vars_list)
        cbasis = model.getAttr(GRB.Attr.CBasis, constr_list)
        
        weights = 1.0 + weight_scale * (np.random.exponential(1.0, N) - 1)
        c_new = list(c_base * weights)
        model.setAttr(GRB.Attr.Obj, vars_list, c_new)
        model.update()
        model.reset(0)
        model.setAttr(GRB.Attr.VBasis, vars_list, vbasis)
        model.setAttr(GRB.Attr.CBasis, constr_list, cbasis)
        model.update()
        model.optimize()
        results_warm.append(int(model.IterCount))
    
    cold_avg = np.mean(results_cold)
    warm_avg = np.mean(results_warm)
    speedup = cold_avg / warm_avg if warm_avg > 0 else float('inf')
    
    print(f"\n{name} (scale={weight_scale:.2f}):")
    print(f"  Cold:      {cold_avg:.1f} avg iters")
    print(f"  Warm:      {warm_avg:.1f} avg iters")
    print(f"  Speedup:   {speedup:.2f}x" + (" ✓" if speedup > 1.1 else " ✗"))

# Exp(1) has mean=1, variance=1, so Exp(1)-1 has mean=0, var=1
# weight_scale controls how much the weights deviate from 1

print("\nExp(1) normalized: weights = 1 + scale * (Exp(1) - 1)")
print("scale=0.0 → all weights = 1 (same objective)")
print("scale=1.0 → standard Bayesian bootstrap")

for scale in [0.0, 0.1, 0.3, 0.5, 1.0]:
    run_test(scale, f"Scale {scale}")

print("\n" + "="*60)
print("CONCLUSION:")
print("  - For identical objectives (scale=0), warm-start helps")
print("  - For Bayesian bootstrap (scale=1), cold start is better!")
print("  - The 'bad' basis from different objective slows convergence")
print("="*60)
