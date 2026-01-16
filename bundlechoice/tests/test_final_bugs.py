#!/usr/bin/env python
"""Final bug verification tests"""
import sys
import os
import importlib.util
import numpy as np
from mpi4py import MPI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_row_generation_create_result():
    """Test row_generation.py _create_result call"""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rg_path = os.path.join(base_path, 'estimation', 'row_generation.py')
    
    bugs = []
    with open(rg_path, 'r') as f:
        lines = f.readlines()
    
    # Check _create_result call
    for i, line in enumerate(lines, 1):
        if '_create_result(' in line and i > 100:
            # Get context
            context_start = max(0, i-10)
            context = '\n'.join(lines[context_start:i+1])
            
            # Check method signature
            base_path_file = os.path.join(base_path, 'estimation', 'base.py')
            with open(base_path_file, 'r') as f2:
                base_lines = f2.readlines()
                for j, bline in enumerate(base_lines):
                    if 'def _create_result' in bline:
                        # Extract signature
                        sig_end = j
                        for k in range(j, min(j+3, len(base_lines))):
                            if '):' in base_lines[k]:
                                sig_end = k
                                break
                        sig = '\n'.join(base_lines[j:sig_end+1])
                        # Check required args
                        required = ['theta', 'converged', 'num_iterations']
                        for req in required:
                            if req not in line:
                                bugs.append(f"Line {i}: _create_result missing required argument '{req}'")
                        break
    
    # Check for f-string syntax errors (only in uncommented code)
    for i, line in enumerate(lines, 1):
        if 'f"' in line and not line.strip().startswith('#'):
            # Check for nested f-strings with same quotes
            if 'f"' in line and line.count('f"') > 1:
                # More sophisticated check
                quote_count = line.count('"') - line.count('\\"')
                if quote_count % 2 != 0:
                    bugs.append(f"Line {i}: Potential f-string syntax error - unmatched quotes")
    
    if bugs and MPI.COMM_WORLD.Get_rank() == 0:
        print(f"✗ row_generation.py: {len(bugs)} issues")
        for bug in bugs:
            print(f"  - {bug}")
        return bugs
    elif MPI.COMM_WORLD.Get_rank() == 0:
        print("✓ row_generation.py: OK")
    return []

if __name__ == '__main__':
    rank = MPI.COMM_WORLD.Get_rank()
    all_bugs = []
    
    all_bugs.extend(test_row_generation_create_result())
    
    if all_bugs and rank == 0:
        print(f"\n✗ TOTAL: {len(all_bugs)} bugs found")
        sys.exit(1)
    elif rank == 0:
        print("\n✓ All checks passed!")
