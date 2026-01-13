#!/usr/bin/env python3
"""Test: No reset - just update Obj and re-optimize (recommended by Gurobi)."""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

np.random.seed(42)

N = 200  
M = 150  

print("="*60)
print("TEST: No reset() - automatic warm-start")
print("="*60)

A = np.random.randn(M, N) * 0.5
b = np.ones(M) * 5
c_base = np.random.randn(N)

model = gp.Model()
model.setParam('OutputFlag', 0)
model.setParam('Method', 1)      # Dual simplex (better for reopt)
model.setParam('LPWarmStart', 2) # Warm start even with presolve
model.ModelSense = GRB.MAXIMIZE

x = model.addMVar(N, lb=0, ub=1, obj=c_base, name='x')
constrs = model.addMConstr(A, x, GRB.LESS_EQUAL, b)
model.update()

# Get variable list for bulk setAttr (need actual Var objects)
vars_list = x.tolist()

# Initial solve
model.optimize()
obj_init = model.ObjVal
sol_init = x.X.copy()
iters_init = int(model.IterCount)
print(f"\n1. Initial:      {iters_init:4d} iters, obj = {obj_init:.4f}")

# Test: Just update Obj and re-optimize (NO reset)
print("\n" + "-"*60)
print("Method A: Just update Obj, no reset (Gurobi recommended)")
print("-"*60)

results_no_reset = []
for b_iter in range(5):
    weights = np.random.exponential(1.0, N)
    weights = weights / weights.mean()
    c_new = list(c_base * weights)  # List for setAttr
    
    # Bulk update (fast)
    model.setAttr(GRB.Attr.Obj, vars_list, c_new)
    model.update()
    
    # Re-optimize - should warm-start automatically
    model.optimize()
    
    iters = int(model.IterCount)
    obj = model.ObjVal
    results_no_reset.append(iters)
    print(f"  Boot {b_iter+1}: {iters:3d} iters, obj = {obj:.4f}")

# Compare: Cold start (new model each time)
print("\n" + "-"*60)
print("Method B: Cold start (new model)")
print("-"*60)

np.random.seed(42)  # Same seed for same weights
results_cold = []
for b_iter in range(5):
    weights = np.random.exponential(1.0, N)
    weights = weights / weights.mean()
    c_new = c_base * weights
    
    m2 = gp.Model()
    m2.setParam('OutputFlag', 0)
    m2.setParam('Method', 1)
    m2.ModelSense = GRB.MAXIMIZE
    x2 = m2.addMVar(N, lb=0, ub=1, obj=c_new, name='x')
    m2.addMConstr(A, x2, GRB.LESS_EQUAL, b)
    m2.optimize()
    
    iters = int(m2.IterCount)
    results_cold.append(iters)
    print(f"  Boot {b_iter+1}: {iters:3d} iters")

# Compare: reset(0) approach (current code)
print("\n" + "-"*60)
print("Method C: reset(0) before optimize")
print("-"*60)

np.random.seed(42)
# Reset model to initial state
model.setAttr(GRB.Attr.Obj, vars_list, list(c_base))
model.update()
model.optimize()

results_reset = []
for b_iter in range(5):
    weights = np.random.exponential(1.0, N)
    weights = weights / weights.mean()
    c_new = list(c_base * weights)
    
    model.setAttr(GRB.Attr.Obj, vars_list, c_new)
    model.update()
    model.reset(0)  # Current approach
    model.optimize()
    
    iters = int(model.IterCount)
    results_reset.append(iters)
    print(f"  Boot {b_iter+1}: {iters:3d} iters")

# Compare: reset(0) WITH basis restore
print("\n" + "-"*60)
print("Method D: reset(0) + manual basis restore")
print("-"*60)

np.random.seed(42)
# Reset model to initial state
model.setAttr(GRB.Attr.Obj, vars_list, list(c_base))
model.update()
model.optimize()

constr_list = model.getConstrs()
results_basis = []
for b_iter in range(5):
    # Save basis BEFORE changing objective
    vbasis = model.getAttr(GRB.Attr.VBasis, vars_list)
    cbasis = model.getAttr(GRB.Attr.CBasis, constr_list)
    
    weights = np.random.exponential(1.0, N)
    weights = weights / weights.mean()
    c_new = list(c_base * weights)
    
    model.setAttr(GRB.Attr.Obj, vars_list, c_new)
    model.update()
    model.reset(0)
    
    # Restore basis AFTER reset
    model.setAttr(GRB.Attr.VBasis, vars_list, vbasis)
    model.setAttr(GRB.Attr.CBasis, constr_list, cbasis)
    model.update()
    model.optimize()
    
    iters = int(model.IterCount)
    results_basis.append(iters)
    print(f"  Boot {b_iter+1}: {iters:3d} iters")

print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"  No reset:                {np.mean(results_no_reset):.1f} avg iters")
print(f"  Cold start:              {np.mean(results_cold):.1f} avg iters")
print(f"  reset(0):                {np.mean(results_reset):.1f} avg iters")
print(f"  reset(0) + basis:        {np.mean(results_basis):.1f} avg iters")

best = min(np.mean(results_no_reset), np.mean(results_cold), 
           np.mean(results_reset), np.mean(results_basis))
print(f"\n  Best method: {best:.1f} avg iters")
print("="*60)
