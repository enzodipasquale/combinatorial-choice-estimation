#!/usr/bin/env python3
"""Test: Verify basis restore gives correct solutions with non-trivial LP."""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

np.random.seed(42)

N = 200  
M = 150  

print("="*60)
print("TEST: Verify warm-start gives CORRECT solutions")
print("="*60)

# Create a non-trivial LP: maximize c'x s.t. Ax <= b, 0 <= x <= 1
A = np.random.randn(M, N) * 0.5
b = np.ones(M) * 5  # Loose constraints
c1 = np.random.randn(N)  # Mix of positive/negative (interesting solution)
c2 = np.random.randn(N)  # Different objective

model = gp.Model()
model.setParam('OutputFlag', 0)
model.setParam('Method', 0)  # Primal simplex
model.setParam('LPWarmStart', 2)
model.ModelSense = GRB.MAXIMIZE

x = model.addMVar(N, lb=0, ub=1, obj=c1, name='x')
constrs = model.addMConstr(A, x, GRB.LESS_EQUAL, b)
model.update()

# Solve with c1 (cold)
model.optimize()
assert model.Status == GRB.OPTIMAL, f"Not optimal: {model.Status}"
obj_c1_cold = model.ObjVal
sol_c1_cold = x.X.copy()
iters_c1_cold = int(model.IterCount)
print(f"\n1. c1 cold:    {iters_c1_cold:4d} iters, obj = {obj_c1_cold:.4f}")
print(f"   Solution: min={sol_c1_cold.min():.3f}, max={sol_c1_cold.max():.3f}, sum={sol_c1_cold.sum():.1f}")

# Save basis
vbasis = np.array([x[i].VBasis for i in range(N)])
cbasis = np.array([constrs[i].CBasis for i in range(M)])
print(f"   Basis: {sum(vbasis == 0)} basic vars")

# Solve with c2 (cold - new model)
model2 = gp.Model()
model2.setParam('OutputFlag', 0)
model2.setParam('Method', 0)
model2.ModelSense = GRB.MAXIMIZE
x2 = model2.addMVar(N, lb=0, ub=1, obj=c2, name='x')
model2.addMConstr(A, x2, GRB.LESS_EQUAL, b)
model2.optimize()
obj_c2_cold = model2.ObjVal
sol_c2_cold = x2.X.copy()
iters_c2_cold = int(model2.IterCount)
print(f"\n2. c2 cold:    {iters_c2_cold:4d} iters, obj = {obj_c2_cold:.4f}")
print(f"   Solution: min={sol_c2_cold.min():.3f}, max={sol_c2_cold.max():.3f}, sum={sol_c2_cold.sum():.1f}")

# Now try c2 with basis restore from c1 solution
x.Obj = c2
model.update()
model.reset(0)
# Restore basis
for i in range(N):
    x[i].VBasis = int(vbasis[i])
for i in range(M):
    constrs[i].CBasis = int(cbasis[i])
model.update()
model.optimize()
obj_c2_warm = model.ObjVal
sol_c2_warm = x.X.copy()
iters_c2_warm = int(model.IterCount)
print(f"\n3. c2 warm:    {iters_c2_warm:4d} iters, obj = {obj_c2_warm:.4f}")
print(f"   Solution: min={sol_c2_warm.min():.3f}, max={sol_c2_warm.max():.3f}, sum={sol_c2_warm.sum():.1f}")

# Verify correctness
print("\n" + "-"*60)
print("VERIFICATION:")
obj_diff = abs(obj_c2_cold - obj_c2_warm)
sol_diff = np.max(np.abs(sol_c2_cold - sol_c2_warm))
print(f"  Objective diff: {obj_diff:.2e} (should be ~0)")
print(f"  Solution diff:  {sol_diff:.2e} (should be ~0)")

if obj_diff < 1e-4 and sol_diff < 1e-4:
    print("\n✓ Warm-start solution is CORRECT!")
else:
    print("\n✗ Solutions DIFFER - potential issue!")

# Summary
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)
speedup = iters_c2_cold / max(iters_c2_warm, 1)
print(f"  Cold start: {iters_c2_cold} iterations")
print(f"  Warm start: {iters_c2_warm} iterations")
print(f"  Speedup:    {speedup:.1f}x")

if iters_c2_warm < iters_c2_cold:
    print("\n✓ Basis restore provides warm-start benefit!")
else:
    print("\n✗ Basis restore did NOT help")
print("="*60)
