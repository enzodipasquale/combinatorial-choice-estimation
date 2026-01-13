#!/usr/bin/env python3
"""Test: Warm-start with SIMILAR objectives (like Bayesian bootstrap)."""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

np.random.seed(42)

N = 200  
M = 150  

print("="*60)
print("TEST: Warm-start with SIMILAR objectives")
print("(Simulating Bayesian bootstrap weight changes)")
print("="*60)

# Create LP
A = np.random.randn(M, N) * 0.5
b = np.ones(M) * 5
c_base = np.random.randn(N)  # Base objective

model = gp.Model()
model.setParam('OutputFlag', 0)
model.setParam('Method', 0)
model.setParam('LPWarmStart', 2)
model.ModelSense = GRB.MAXIMIZE

x = model.addMVar(N, lb=0, ub=1, obj=c_base, name='x')
constrs = model.addMConstr(A, x, GRB.LESS_EQUAL, b)
model.update()

# Initial solve
model.optimize()
iters_initial = int(model.IterCount)
print(f"\n1. Initial cold: {iters_initial:4d} iters")

# Simulate 5 Bayesian bootstrap iterations
print("\n" + "-"*60)
print("Simulating Bayesian bootstrap (Exp(1) weights):")
print("-"*60)

iters_with_reset = []
iters_with_basis = []

for b_iter in range(5):
    # Generate Exp(1) weights like Bayesian bootstrap
    weights = np.random.exponential(1.0, N)
    weights = weights / weights.mean()  # Normalize
    c_new = c_base * weights
    
    # Method A: reset(0) without basis save/restore (current approach)
    x.Obj = c_new
    model.update()
    model.reset(0)
    model.optimize()
    iters_a = int(model.IterCount)
    iters_with_reset.append(iters_a)
    
    # Save basis for next iteration
    vbasis = np.array([x[i].VBasis for i in range(N)])
    cbasis = np.array([constrs[i].CBasis for i in range(M)])
    
    # Method B: For comparison, cold start
    model2 = gp.Model()
    model2.setParam('OutputFlag', 0)
    model2.setParam('Method', 0)
    model2.ModelSense = GRB.MAXIMIZE
    x2 = model2.addMVar(N, lb=0, ub=1, obj=c_new, name='x')
    model2.addMConstr(A, x2, GRB.LESS_EQUAL, b)
    model2.optimize()
    iters_cold = int(model2.IterCount)
    
    print(f"  Boot {b_iter+1}: reset(0)={iters_a:3d} iters, cold={iters_cold:3d} iters")

# Now test with basis restore between iterations
print("\n" + "-"*60)
print("With explicit basis restore between iterations:")
print("-"*60)

# Reset model
x.Obj = c_base
model.update()
model.optimize()

for b_iter in range(5):
    # Save current basis
    vbasis = np.array([x[i].VBasis for i in range(N)])
    cbasis = np.array([constrs[i].CBasis for i in range(M)])
    
    # Generate new weights
    weights = np.random.exponential(1.0, N)
    weights = weights / weights.mean()
    c_new = c_base * weights
    
    # Update objective, reset, restore basis, solve
    x.Obj = c_new
    model.update()
    model.reset(0)
    for i in range(N):
        x[i].VBasis = int(vbasis[i])
    for i in range(M):
        constrs[i].CBasis = int(cbasis[i])
    model.update()
    model.optimize()
    iters_b = int(model.IterCount)
    iters_with_basis.append(iters_b)
    print(f"  Boot {b_iter+1}: {iters_b:3d} iters")

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"  Initial cold:        {iters_initial} iters")
print(f"  With reset(0) only:  {np.mean(iters_with_reset):.1f} avg iters")
print(f"  With basis restore:  {np.mean(iters_with_basis):.1f} avg iters")

if np.mean(iters_with_basis) < np.mean(iters_with_reset) * 0.7:
    print("\n✓ Basis restore HELPS for similar objectives!")
else:
    print("\n✗ Basis restore doesn't help much")
    print("  (reset(0) is sufficient, basis naturally persists)")
print("="*60)
