"""
Test SE computation with different step sizes using 200 agents.
"""
import sys
sys.path.insert(0, 'experiments/BLP')
from experiment_se_comparison_no_fe import run_greedy_experiment
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    num_agents = 200
    step_sizes = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]
    
    if rank == 0:
        print("\n" + "=" * 70)
        print(f"TESTING STEP SIZES WITH N={num_agents}")
        print("=" * 70)
    
    for step_size in step_sizes:
        if rank == 0:
            print(f"\n{'='*70}")
            print(f"Step size = {step_size:.0e}")
            print(f"{'='*70}")
        
        res = run_greedy_experiment(
            num_se_simulations=150,
            num_bootstrap=100,
            step_size=step_size,
            num_agents=num_agents
        )
        
        if rank == 0 and res:
            print(f"\nResults for step_size={step_size:.0e}:")
            print(f"  Full SE (Agent): {res['se_full'][0]:.6f}")
            print(f"  Boot SE (Agent): {res['se_boot'][0]:.6f}")
            print(f"  Subs SE (Agent): {res['se_subs'][0]:.6f}")
            print(f"  Boot/Full ratio: {res['se_boot'][0]/res['se_full'][0]:.4f}")
            print(f"  Subs/Full ratio: {res['se_subs'][0]/res['se_full'][0]:.4f}")
        
        comm.Barrier()
    
    if rank == 0:
        print("\n" + "=" * 70)
        print("DONE")
        print("=" * 70)

if __name__ == "__main__":
    main()
