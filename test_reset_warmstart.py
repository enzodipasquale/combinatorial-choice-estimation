#!/usr/bin/env python3
"""Test: Does Gurobi reset(0) preserve LP basis for warm-start?"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB

np.random.seed(42)

# Create a non-trivial LP
N = 500  # variables
M = 300  # constraints

print("="*60)
print("TEST: Does reset(0) preserve LP warm-start basis?")
print("="*60)

# Random LP: min c'x s.t. Ax <= b, x >= 0
A = np.random.randn(M, N)
b = np.abs(np.random.randn(M)) * 10
c1 = np.random.randn(N)  # First objective
c2 = np.random.randn(N)  # Second objective (different)

model = gp.Model()
model.setParam('OutputFlag', 0)
model.setParam('Method', 0)  # Primal simplex (shows iterations clearly)
model.setParam('LPWarmStart', 2)  # Use advanced basis

x = model.addMVar(N, lb=0, obj=c1, name='x')
model.addMConstr(A, x, GRB.LESS_EQUAL, b)
model.update()

# Solve 1: Initial solve (cold start)
model.optimize()
iters_cold = int(model.IterCount)
print(f"\n1. Cold start:           {iters_cold:4d} iterations")

# Check basis is set
basis_before = [x[i].VBasis for i in range(min(10, N))]
print(f"   Basis sample (first 10): {basis_before}")

# Solve 2: Change objective, use reset(0)
x.Obj = c2
model.update()
model.reset(0)
model.optimize()
iters_reset0 = int(model.IterCount)
basis_after_reset0 = [x[i].VBasis for i in range(min(10, N))]
print(f"\n2. After reset(0):       {iters_reset0:4d} iterations")
print(f"   Basis sample (first 10): {basis_after_reset0}")

# Solve 3: Change objective back, NO reset
x.Obj = c1
model.update()
# NO reset - just optimize
model.optimize()
iters_no_reset = int(model.IterCount)
print(f"\n3. No reset (just obj change): {iters_no_reset:4d} iterations")

# Solve 4: Change objective, use reset(1) - should lose basis
x.Obj = c2
model.update()
model.reset(1)  # Clear everything including hints/priorities
model.optimize()
iters_reset1 = int(model.IterCount)
basis_after_reset1 = [x[i].VBasis for i in range(min(10, N))]
print(f"\n4. After reset(1):       {iters_reset1:4d} iterations")
print(f"   Basis sample (first 10): {basis_after_reset1}")

# Solve 5: True cold start - rebuild model
model2 = gp.Model()
model2.setParam('OutputFlag', 0)
model2.setParam('Method', 0)
x2 = model2.addMVar(N, lb=0, obj=c1, name='x')
model2.addMConstr(A, x2, GRB.LESS_EQUAL, b)
model2.optimize()
iters_true_cold = int(model2.IterCount)
print(f"\n5. True cold (new model): {iters_true_cold:4d} iterations")

print("\n" + "="*60)
print("ANALYSIS:")
print("="*60)

if iters_reset0 < iters_cold * 0.5:
    print("✓ reset(0) PRESERVES warm-start (few iterations)")
else:
    print("✗ reset(0) LOSES warm-start (many iterations)")

if iters_no_reset < iters_cold * 0.5:
    print("✓ No-reset also works for warm-start")
elif iters_no_reset == 0:
    print("⚠ No-reset returns cached solution (0 iterations) - need reset!")
else:
    print("? No-reset: unclear")

if iters_reset1 >= iters_cold * 0.8:
    print("✓ reset(1) correctly loses warm-start info")
else:
    print("? reset(1) still has some warm-start effect")

print("\n" + "="*60)
print("CONCLUSION:")
if iters_reset0 < iters_cold * 0.3 and iters_no_reset > 0:
    print("reset(0) is SAFE - preserves LP basis warm-start")
    print("You can also try without reset if iterations > 0")
elif iters_no_reset == 0:
    print("reset(0) is NEEDED - without it, Gurobi returns cached solution")
    print("But reset(0) preserves the basis for warm-start")
print("="*60)
