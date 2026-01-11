"""
Benchmark Bayesian bootstrap warm-start strategies:
- none: No warm-start (baseline)
- constraints: Reuse constraints from previous solve
- theta: Use theta from previous solve as initial point
"""
import time
import numpy as np
from mpi4py import MPI
from bundlechoice.core import BundleChoice


def run_benchmark(subproblem_name, num_agents, num_bootstrap=30):
    """Run benchmark for a given subproblem type and agent count."""
    num_items = 10
    num_features = 2
    sigma = 1.5
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"{subproblem_name.upper()}: {num_agents} agents, {num_bootstrap} bootstrap samples")
        print("="*70)
    
    theta_0 = np.array([3.0, 0.3])
    
    if rank == 0:
        np.random.seed(42)
        modular_agent = np.abs(np.random.randn(num_agents, num_items, 1))
        gen_errors = sigma * np.random.randn(num_agents, num_items)
        np.random.seed(2024)
        est_errors = sigma * np.random.randn(num_agents, num_items)
        if subproblem_name == "LinearKnapsack":
            item_weights = np.ones(num_items)
            agent_capacity = np.full(num_agents, 5, dtype=np.float64)
        else:
            item_weights = agent_capacity = None
    else:
        modular_agent = gen_errors = est_errors = item_weights = agent_capacity = None
    
    modular_agent = comm.bcast(modular_agent, root=0)
    gen_errors = comm.bcast(gen_errors, root=0)
    est_errors = comm.bcast(est_errors, root=0)
    if subproblem_name == "LinearKnapsack":
        item_weights = comm.bcast(item_weights, root=0)
        agent_capacity = comm.bcast(agent_capacity, root=0)
    
    config = {
        "dimensions": {"num_agents": num_agents, "num_items": num_items, 
                      "num_features": num_features, "num_simulations": 1},
        "subproblem": {"name": subproblem_name},
        "row_generation": {"max_iters": 200, "tolerance_optimality": 0.001},
    }
    if subproblem_name == "LinearKnapsack":
        config["subproblem"]["settings"] = {"capacity": 5}
    
    def _oracle(agent_idx, bundles, data):
        modular = data["agent_data"]["modular"][agent_idx]
        if bundles.ndim == 1:
            return np.array([(modular[:, 0] * bundles).sum(), -bundles.sum() ** 2])
        else:
            sizes = bundles.sum(axis=1)
            return np.column_stack([(modular[:, 0] * bundles).sum(axis=1), -sizes ** 2])
    
    # Generate observed bundles
    gen_data = {"agent_data": {"modular": modular_agent}, "item_data": {}, "errors": gen_errors}
    if subproblem_name == "LinearKnapsack":
        gen_data["agent_data"]["capacity"] = agent_capacity
        gen_data["item_data"]["weights"] = item_weights
    
    bc_gen = BundleChoice()
    bc_gen.load_config(config)
    bc_gen.data.load_and_scatter(gen_data if rank == 0 else None)
    bc_gen.features.set_oracle(_oracle)
    bc_gen.subproblems.load()
    obs_bundles = bc_gen.subproblems.init_and_solve(theta_0)
    obs_bundles = comm.bcast(obs_bundles, root=0)
    
    # Setup estimation
    est_data = {"agent_data": {"modular": modular_agent}, "item_data": {}, 
                "errors": est_errors, "obs_bundle": obs_bundles}
    if subproblem_name == "LinearKnapsack":
        est_data["agent_data"]["capacity"] = agent_capacity
        est_data["item_data"]["weights"] = item_weights
    
    bc = BundleChoice()
    bc.load_config(config)
    bc.data.load_and_scatter(est_data if rank == 0 else None)
    bc.features.set_oracle(_oracle)
    bc.subproblems.load()
    
    result = bc.row_generation.solve()
    theta_hat = result.theta_hat
    
    if rank == 0:
        print(f"  θ_hat: {theta_hat}")
    
    results = {}
    
    # Test each warm-start strategy
    for strategy in ["none", "constraints", "theta"]:
        comm.Barrier()
        t0 = time.perf_counter()
        se = bc.standard_errors.compute_bayesian_bootstrap(
            theta_hat, bc.row_generation, num_bootstrap=num_bootstrap, 
            seed=777, warmstart=strategy
        )
        comm.Barrier()
        elapsed = time.perf_counter() - t0
        
        if rank == 0:
            results[strategy] = elapsed
            print(f"  {strategy:12s}: {elapsed:.2f}s")
    
    if rank == 0:
        # Compute speedups relative to baseline
        baseline = results["none"]
        print(f"\n  Speedups vs baseline:")
        for s in ["constraints", "theta"]:
            speedup = baseline / results[s] if results[s] > 0 else 0
            print(f"    {s:12s}: {speedup:.2f}x")
        
        return results
    return None


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    all_results = {}
    
    # Greedy benchmarks
    for n in [100, 250]:
        res = run_benchmark("Greedy", n, num_bootstrap=30)
        if rank == 0 and res:
            all_results[f"Greedy_{n}"] = res
    
    # Knapsack benchmarks  
    for n in [100, 250]:
        res = run_benchmark("LinearKnapsack", n, num_bootstrap=30)
        if rank == 0 and res:
            all_results[f"Knapsack_{n}"] = res
    
    # Summary table
    if rank == 0:
        print("\n" + "=" * 80)
        print("SUMMARY: Warm-start Comparison")
        print("=" * 80)
        print(f"{'Setting':<20} {'None (s)':<12} {'Constr (s)':<12} {'Theta (s)':<12} {'Best':<10}")
        print("-" * 70)
        for name, r in all_results.items():
            best = min(r, key=r.get)
            speedup = r["none"] / r[best] if r[best] > 0 else 0
            print(f"{name:<20} {r['none']:>10.2f}  {r['constraints']:>10.2f}  {r['theta']:>10.2f}  {best} ({speedup:.2f}x)")


if __name__ == "__main__":
    main()
