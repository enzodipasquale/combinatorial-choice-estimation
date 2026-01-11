"""
Test SE computation with different step sizes and 200 agents.
Investigates convergence of full sandwich SE as step size decreases.
"""
import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice
import sys
sys.path.insert(0, '/Users/enzo-macbookpro/MyProjects/combinatorial-choice-estimation/experiments/BLP')
from experiment_se_comparison_no_fe import run_greedy_experiment, _greedy_oracle, _install_greedy_find_best

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    num_agents = 200
    num_se_sims = 200
    num_bootstrap = 200
    step_sizes = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    
    if rank == 0:
        print("\n" + "#" * 70)
        print(f"  TESTING STEP SIZE CONVERGENCE (N={num_agents})")
        print("#" * 70)
        print(f"  Step sizes to test: {step_sizes}")
        print(f"  SE simulations: {num_se_sims}, Bootstrap: {num_bootstrap}")
        print("#" * 70)
    
    results = {}
    
    for step_size in step_sizes:
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Testing step_size = {step_size:.0e}")
            print(f"{'='*70}")
        
        res = run_greedy_experiment(
            num_se_simulations=num_se_sims,
            num_bootstrap=num_bootstrap,
            step_size=step_size,
            num_agents=num_agents
        )
        
        if rank == 0 and res:
            results[step_size] = res
            print(f"\nStep size {step_size:.0e}:")
            print(f"  Full SE: {res['se_full']}")
            print(f"  Boot SE: {res['se_boot']}")
            print(f"  Subs SE: {res['se_subs']}")
            print(f"  Boot/Full: {res['se_boot'] / res['se_full']}")
            print(f"  Subs/Full: {res['se_subs'] / res['se_full']}")
        
        comm.Barrier()
    
    if rank == 0:
        print("\n" + "#" * 70)
        print("SUMMARY TABLE")
        print("#" * 70)
        print(f"\n{'Step Size':<12} {'Full SE (Agent)':<18} {'Boot SE':<12} {'Subs SE':<12} {'Boot/Full':<12} {'Subs/Full':<12}")
        print("-" * 90)
        for step_size in step_sizes:
            if step_size in results:
                res = results[step_size]
                print(f"{step_size:>10.0e}  {res['se_full'][0]:>16.6f}  {res['se_boot'][0]:>10.6f}  {res['se_subs'][0]:>10.6f}  {(res['se_boot']/res['se_full'])[0]:>10.4f}  {(res['se_subs']/res['se_full'])[0]:>10.4f}")

if __name__ == "__main__":
    main()
