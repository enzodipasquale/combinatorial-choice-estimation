import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'bundlechoice'))

from bundlechoice.v2.subproblems.registry.quad_supermod_network import QuadSubmodularMinimization

def brute_force_minimize(P_j_j, choice_set):
    """Brute force search for minimum of quadratic form."""
    n = len(choice_set)
    min_value = float('inf')
    min_bundle = None
    
    for i in range(2**n):
        bundle = np.zeros(P_j_j.shape[0], dtype=bool)
        for j in range(n):
            if i & (1 << j):
                bundle[choice_set[j]] = True
        
        value = bundle @ P_j_j @ bundle
        if value < min_value:
            min_value = value
            min_bundle = bundle.copy()
    
    return min_bundle, min_value

def test_comprehensive():
    np.random.seed(42)
    num_items = 4
    choice_set = list(range(num_items))
    
    test_cases = [
        # Case 1: All negative diagonals (should prefer empty bundle)
        {
            "name": "All negative diagonals",
            "diagonals": [-2, -3, -1, -4],
            "off_diagonals": [[0, -0.5, -0.3, -0.2],
                              [0, 0, -0.4, -0.1],
                              [0, 0, 0, -0.6],
                              [0, 0, 0, 0]]
        },
        # Case 2: Mixed positive/negative diagonals
        {
            "name": "Mixed diagonals",
            "diagonals": [2, -1, 3, -2],
            "off_diagonals": [[0, -0.5, -0.3, -0.2],
                              [0, 0, -0.4, -0.1],
                              [0, 0, 0, -0.6],
                              [0, 0, 0, 0]]
        },
        # Case 3: All positive diagonals with strong negative off-diagonals
        {
            "name": "Positive diagonals, strong negative off-diagonals",
            "diagonals": [1, 1, 1, 1],
            "off_diagonals": [[0, -2, -2, -2],
                              [0, 0, -2, -2],
                              [0, 0, 0, -2],
                              [0, 0, 0, 0]]
        },
        # Case 4: Random submodular matrix (all off-diagonals negative)
        {
            "name": "Random submodular (all negative off-diagonals)",
            "diagonals": np.random.uniform(-2, 2, num_items),
            "off_diagonals": None  # Will generate random negative
        }
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n=== Test Case {i+1}: {case['name']} ===")
        
        # Build matrix
        P_j_j = np.zeros((num_items, num_items))
        
        if case["off_diagonals"] is None:
            # Generate random negative off-diagonals
            off_diag = -np.abs(np.random.uniform(0, 2, (num_items, num_items)))
            off_diag = (off_diag + off_diag.T) / 2
            np.fill_diagonal(off_diag, 0)
        else:
            off_diag = np.array(case["off_diagonals"])
            # Make symmetric and negative
            off_diag = -np.abs(off_diag + off_diag.T)
            np.fill_diagonal(off_diag, 0)
        
        # Set diagonals
        np.fill_diagonal(P_j_j, case["diagonals"])
        P_j_j += off_diag
        
        print("P_j_j:")
        print(P_j_j)
        print(f"Diagonal entries: {np.diag(P_j_j)}")
        print(f"Off-diagonal entries <= 0: {np.all(P_j_j - np.diag(np.diag(P_j_j)) <= 0)}")
        
        # Test solver
        solver = QuadSubmodularMinimization(P_j_j)
        quad_bundle = solver.solve_QSM()
        quad_value = quad_bundle @ P_j_j @ quad_bundle
        
        # Brute force
        bf_bundle, bf_value = brute_force_minimize(P_j_j, choice_set)
        
        print(f"Quadratic solver bundle: {quad_bundle}")
        print(f"Quadratic solver value: {quad_value:.6f}")
        print(f"Brute force bundle: {bf_bundle}")
        print(f"Brute force value: {bf_value:.6f}")
        print(f"Match: {np.array_equal(quad_bundle, bf_bundle)}")
        print(f"Value difference: {abs(quad_value - bf_value):.6f}")
        
        # Check if solution is all True or all False
        if np.all(quad_bundle):
            print("Solution: All items selected")
        elif np.all(~quad_bundle):
            print("Solution: No items selected")
        else:
            print(f"Solution: Mixed selection ({np.sum(quad_bundle)} items)")

    # Additional: 20 random submodular matrices
    print("\n=== 20 Random Submodular Matrices (all negative off-diagonals, positive diagonals) ===")
    n_random = 20
    n_fail = 0
    n_all_true = 0
    n_all_false = 0
    n_mixed = 0
    for idx in range(n_random):
        diagonals = np.random.uniform(0, 10, num_items)  # More positive diagonals
        off_diag = -np.abs(np.random.uniform(0, 2, (num_items, num_items)))
        off_diag = (off_diag + off_diag.T) / 2
        np.fill_diagonal(off_diag, 0)
        P_j_j = np.zeros((num_items, num_items))
        np.fill_diagonal(P_j_j, diagonals)
        P_j_j += off_diag
        solver = QuadSubmodularMinimization(P_j_j)
        quad_bundle = solver.solve_QSM()
        quad_value = quad_bundle @ P_j_j @ quad_bundle
        bf_bundle, bf_value = brute_force_minimize(P_j_j, choice_set)
        match = np.array_equal(quad_bundle, bf_bundle)
        value_diff = abs(quad_value - bf_value)
        if not match or value_diff > 1e-8:
            n_fail += 1
            print(f"Random Test {idx+1}: FAIL")
            print(f"P_j_j:\n{P_j_j}")
            print(f"Quad bundle: {quad_bundle}, value: {quad_value:.6f}")
            print(f"BF bundle: {bf_bundle}, value: {bf_value:.6f}")
            print(f"Value diff: {value_diff:.6e}")
        else:
            print(f"Random Test {idx+1}: PASS (value diff: {value_diff:.2e})")
        if np.all(quad_bundle):
            n_all_true += 1
        elif np.all(~quad_bundle):
            n_all_false += 1
        else:
            n_mixed += 1
    print(f"\nSummary: {n_random-n_fail} passed, {n_fail} failed.")
    print(f"All True: {n_all_true}, All False: {n_all_false}, Mixed: {n_mixed}")

if __name__ == "__main__":
    test_comprehensive() 