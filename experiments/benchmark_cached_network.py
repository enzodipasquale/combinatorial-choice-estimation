"""Benchmark cached vs original NetworkX implementations - measuring only subproblem solving time."""
import numpy as np
import time
from bundlechoice.subproblems.registry.quad_supermodular import MinCutSubmodularSolver, MinCutSubmodularSolverCached

# Test parameters
num_items = 200
num_iterations = 1000
random_seed = 42
np.random.seed(random_seed)

# Generate random quadratic matrices (pre-generate to exclude from timing)
b_j_j = np.random.randn(num_items, num_items)
b_j_j = np.triu(b_j_j, k=1)
b_j_j = -np.abs(b_j_j)
diagonal = np.random.rand(num_items) * 2 + 1
np.fill_diagonal(b_j_j, diagonal)

# Pre-generate all test matrices
np.random.seed(random_seed)
test_matrices = []
for _ in range(num_iterations):
    b_j_j_new = b_j_j.copy()
    perturbation = np.random.randn(num_items, num_items)
    perturbation = np.triu(perturbation, k=1)
    perturbation = -np.abs(perturbation) * 0.01
    b_j_j_new += perturbation
    np.fill_diagonal(b_j_j_new, np.diag(b_j_j) + np.random.randn(num_items) * 0.01)
    test_matrices.append(b_j_j_new)

print(f"Benchmarking with {num_items} items, {num_iterations} iterations")
print("=" * 60)

# Original: time spent in __init__ + solve_QSM()
start_time = time.time()
for b_j_j_test in test_matrices:
    solver = MinCutSubmodularSolver(b_j_j_test, np.arange(num_items))
    result = solver.solve_QSM()
original_time = time.time() - start_time
print(f"Original (__init__ + solve_QSM): {original_time:.4f}s")

# Cached: time spent in update_weights + solve_QSM()
solver_cached = MinCutSubmodularSolverCached(b_j_j, np.arange(num_items))
start_time = time.time()
for b_j_j_test in test_matrices:
    solver_cached.update_weights(b_j_j_test)
    result = solver_cached.solve_QSM()
cached_time = time.time() - start_time
print(f"Cached (update_weights + solve_QSM): {cached_time:.4f}s")

# Breakdown: Just solve_QSM() for cached (excluding update_weights)
solver_cached2 = MinCutSubmodularSolverCached(b_j_j, np.arange(num_items))
solver_cached2.update_weights(b_j_j)  # Pre-update
start_time = time.time()
for _ in range(num_iterations):
    result = solver_cached2.solve_QSM()
solve_only_time = time.time() - start_time
print(f"Cached (solve_QSM only): {solve_only_time:.4f}s")

# Breakdown: Just update_weights for cached
solver_cached3 = MinCutSubmodularSolverCached(b_j_j, np.arange(num_items))
start_time = time.time()
for b_j_j_test in test_matrices:
    solver_cached3.update_weights(b_j_j_test)
update_only_time = time.time() - start_time
print(f"Cached (update_weights only): {update_only_time:.4f}s")

print("=" * 60)
speedup = original_time / cached_time
print(f"\nOverall speedup: {speedup:.2f}x faster")
print(f"Solve time breakdown: original={original_time:.4f}s, cached={solve_only_time:.4f}s, update={update_only_time:.4f}s")

# Verify correctness
print("\nVerifying correctness...")
solver_orig_check = MinCutSubmodularSolver(test_matrices[-1], np.arange(num_items))
result_orig_check = solver_orig_check.solve_QSM()
solver_cached_check = MinCutSubmodularSolverCached(b_j_j, np.arange(num_items))
solver_cached_check.update_weights(test_matrices[-1])
result_cached_check = solver_cached_check.solve_QSM()
print(f"Results match: {np.array_equal(result_orig_check, result_cached_check)}")

