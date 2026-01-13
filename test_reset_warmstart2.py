#!/usr/bin/env python3
"""Test: Manual basis save/restore for true LP warm-start."""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

np.random.seed(42)

N = 500  
M = 300  

print("="*60)
print("TEST: Manual basis preservation for LP warm-start")
print("="*60)

A = np.random.randn(M, N)
b = np.abs(np.random.randn(M)) * 10
c1 = np.random.randn(N)
c2 = np.random.randn(N)

model = gp.Model()
model.setParam('OutputFlag', 0)
model.setParam('Method', 0)
model.setParam('LPWarmStart', 2)

x = model.addMVar(N, lb=0, obj=c1, name='x')
constrs = model.addMConstr(A, x, GRB.LESS_EQUAL, b)
model.update()

# Solve 1: Cold start
model.optimize()
iters_cold = int(model.IterCount)
print(f"\n1. Cold start:           {iters_cold:4d} iterations")

# Save basis
vbasis = np.array([x[i].VBasis for i in range(N)])
cbasis = np.array([constrs[i].CBasis for i in range(M)])
print(f"   Saved basis: {sum(vbasis == 0)} basic vars, {sum(cbasis == 0)} basic constrs")

# Solve 2: Change obj, reset(0), NO basis restore
x.Obj = c2
model.update()
model.reset(0)
model.optimize()
iters_reset0 = int(model.IterCount)
print(f"\n2. reset(0) no restore:  {iters_reset0:4d} iterations")

# Solve 3: Change obj, reset(0), WITH basis restore
x.Obj = c1
model.update()
model.reset(0)
# Restore basis before optimize
for i in range(N):
    x[i].VBasis = int(vbasis[i])
for i in range(M):
    constrs[i].CBasis = int(cbasis[i])
model.update()
model.optimize()
iters_restore = int(model.IterCount)
print(f"\n3. reset(0) + restore:   {iters_restore:4d} iterations")

# Solve 4: Change obj to c2, reset(0), restore basis
x.Obj = c2
model.update()
model.reset(0)
for i in range(N):
    x[i].VBasis = int(vbasis[i])
for i in range(M):
    constrs[i].CBasis = int(cbasis[i])
model.update()
model.optimize()
iters_c2_restore = int(model.IterCount)
print(f"\n4. c2 + reset + restore: {iters_c2_restore:4d} iterations")

# Solve 5: Change obj, reset(0), restore basis (should be fast)
x.Obj = c1
model.update()
model.reset(0)
for i in range(N):
    x[i].VBasis = int(vbasis[i])
for i in range(M):
    constrs[i].CBasis = int(cbasis[i])
model.update()
model.optimize()
iters_restore2 = int(model.IterCount)
print(f"\n5. reset(0) + restore:   {iters_restore2:4d} iterations (confirm)")

print("\n" + "="*60)
print("CONCLUSION:")
if iters_restore < iters_cold * 0.3:
    print("✓ Manual basis restore enables true warm-start!")
    print(f"  Cold: {iters_cold} iters → Same obj warm: {iters_restore} iters")
    print(f"  Different obj warm: {iters_c2_restore} iters ({100*iters_c2_restore/iters_cold:.0f}% of cold)")
else:
    print("✗ Manual basis restore didn't help much")
print("="*60)
