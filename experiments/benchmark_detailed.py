"""Detailed benchmark separating graph building from min-cut solving."""
import numpy as np
import time
from bundlechoice.subproblems.registry.quad_supermodular import MinCutSubmodularSolver, MinCutSubmodularSolverCached

# Test parameters
num_items = 200
num_iterations = 1000
random_seed = 42
np.random.seed(random_seed)

# Generate random quadratic matrices
b_j_j = np.random.randn(num_items, num_items)
b_j_j = np.triu(b_j_j, k=1)
b_j_j = -np.abs(b_j_j)
diagonal = np.random.rand(num_items) * 2 + 1
np.fill_diagonal(b_j_j, diagonal)

# Pre-generate test matrices
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

print(f"Detailed benchmark with {num_items} items, {num_iterations} iterations")
print("=" * 70)

# Time just the __init__ part of original (excluding solve)
init_times = []
for b_j_j_test in test_matrices[:100]:  # Sample 100 for init timing
    start = time.time()
    solver = MinCutSubmodularSolver(b_j_j_test, np.arange(num_items))
    init_times.append(time.time() - start)
avg_init_time = np.mean(init_times)
print(f"Original __init__ time (avg): {avg_init_time*1000:.3f}ms")

# Time build_posiform + build_graph separately
solver = MinCutSubmodularSolver(b_j_j, np.arange(num_items))
posiform_times = []
graph_times = []
for b_j_j_test in test_matrices[:100]:
    solver.b_j_j = b_j_j_test
    start = time.time()
    a_j_j, a_j = solver.build_posiform()
    posiform_times.append(time.time() - start)
    
    start = time.time()
    G = solver.build_graph(a_j_j, a_j, solver.choice_set)
    graph_times.append(time.time() - start)
avg_posiform = np.mean(posiform_times) * 1000
avg_graph = np.mean(graph_times) * 1000
print(f"Original build_posiform time (avg): {avg_posiform:.3f}ms")
print(f"Original build_graph time (avg): {avg_graph:.3f}ms")

# Time min-cut solving separately
mincut_times = []
for b_j_j_test in test_matrices[:100]:
    solver.b_j_j = b_j_j_test
    a_j_j, a_j = solver.build_posiform()
    G = solver.build_graph(a_j_j, a_j, solver.choice_set)
    start = time.time()
    S, cut_value = solver.solve_mincut(G)
    mincut_times.append(time.time() - start)
avg_mincut = np.mean(mincut_times) * 1000
print(f"Original solve_mincut time (avg): {avg_mincut:.3f}ms")
print(f"Total estimated per iteration: {(avg_init_time + np.mean(posiform_times) + np.mean(graph_times) + np.mean(mincut_times))*1000:.3f}ms")
print()

# Cached version: update_weights breakdown
solver_cached = MinCutSubmodularSolverCached(b_j_j, np.arange(num_items))

# Time update_weights
update_times = []
for b_j_j_test in test_matrices[:100]:
    start = time.time()
    solver_cached.update_weights(b_j_j_test)
    update_times.append(time.time() - start)
avg_update = np.mean(update_times) * 1000
print(f"Cached update_weights time (avg): {avg_update:.3f}ms")

# Time cached solve_mincut
cached_mincut_times = []
solver_cached.update_weights(b_j_j)  # Pre-update
for _ in range(100):
    start = time.time()
    S, cut_value = solver_cached.solve_mincut(solver_cached.G)
    cached_mincut_times.append(time.time() - start)
avg_cached_mincut = np.mean(cached_mincut_times) * 1000
print(f"Cached solve_mincut time (avg): {avg_cached_mincut:.3f}ms")
print()

print("=" * 70)
print("BREAKDOWN:")
print(f"Original per iteration:")
print(f"  - __init__: {avg_init_time*1000:.3f}ms ({avg_init_time/(avg_init_time+np.mean(posiform_times)+np.mean(graph_times)+np.mean(mincut_times))*100:.1f}%)")
print(f"  - build_posiform: {avg_posiform:.3f}ms ({np.mean(posiform_times)/(avg_init_time+np.mean(posiform_times)+np.mean(graph_times)+np.mean(mincut_times))*100:.1f}%)")
print(f"  - build_graph: {avg_graph:.3f}ms ({np.mean(graph_times)/(avg_init_time+np.mean(posiform_times)+np.mean(graph_times)+np.mean(mincut_times))*100:.1f}%)")
print(f"  - solve_mincut: {avg_mincut:.3f}ms ({np.mean(mincut_times)/(avg_init_time+np.mean(posiform_times)+np.mean(graph_times)+np.mean(mincut_times))*100:.1f}%)")
print(f"  - TOTAL: {(avg_init_time + np.mean(posiform_times) + np.mean(graph_times) + np.mean(mincut_times))*1000:.3f}ms")
print()
print(f"Cached per iteration:")
print(f"  - update_weights: {avg_update:.3f}ms ({np.mean(update_times)/(np.mean(update_times)+np.mean(cached_mincut_times))*100:.1f}%)")
print(f"  - solve_mincut: {avg_cached_mincut:.3f}ms ({np.mean(cached_mincut_times)/(np.mean(update_times)+np.mean(cached_mincut_times))*100:.1f}%)")
print(f"  - TOTAL: {(np.mean(update_times) + np.mean(cached_mincut_times))*1000:.3f}ms")
print()

savings = (avg_init_time + np.mean(posiform_times) + np.mean(graph_times)) * 1000
print(f"Savings from caching (posiform + graph): ~{savings:.3f}ms per iteration ({savings/((avg_init_time + np.mean(posiform_times) + np.mean(graph_times) + np.mean(mincut_times))*1000)*100:.1f}% of original time)")

