import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'bundlechoice'))

from bundlechoice.v2.subproblems.registry.quad_supermod_network import QuadSubmodularMinimization

def debug_case4():
    """Debug the failing case 4 from comprehensive test."""
    np.random.seed(42)
    num_items = 4
    choice_set = list(range(num_items))
    
    # Recreate the failing matrix directly
    P_j_j = np.array([
        [-0.50183952, -0.37855477, -0.44526313, -0.58520919],
        [-0.37855477,  1.80285723, -0.1164618,  -0.74733314],
        [-0.44526313, -0.1164618,   0.92797577, -0.30767476],
        [-0.58520919, -0.74733314, -0.30767476,  0.39463394]
    ])
    
    print("P_j_j:")
    print(P_j_j)
    print(f"Diagonal entries: {np.diag(P_j_j)}")
    
    # Test all possible bundles
    print("\nAll possible bundles and their values:")
    for i in range(2**num_items):
        bundle = np.zeros(num_items, dtype=bool)
        for j in range(num_items):
            if i & (1 << j):
                bundle[j] = True
        
        value = bundle @ P_j_j @ bundle
        print(f"Bundle {bundle}: {value:.6f}")
    
    # Test solver
    solver = QuadSubmodularMinimization(P_j_j)
    quad_bundle = solver.solve_QSM()
    quad_value = quad_bundle @ P_j_j @ quad_bundle
    
    print(f"\nQuadratic solver bundle: {quad_bundle}")
    print(f"Quadratic solver value: {quad_value:.6f}")
    
    # Debug posiform construction
    a_j_j, a_j, positive = solver.build_posiform(P_j_j)
    print(f"\nPosiform debug:")
    print(f"a_j_j:\n{a_j_j}")
    print(f"a_j: {a_j}")
    print(f"positive: {positive}")

if __name__ == "__main__":
    debug_case4() 