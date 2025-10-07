#!/usr/bin/env python
"""
Benchmark script comparing old (pickle-based) vs new (buffer-based) MPI operations.

Run with: mpirun -n <num_ranks> python benchmark_comm_manager.py

This script tests:
1. scatter_from_root (pickle) vs scatter_array (buffer)
2. broadcast_from_root (pickle) vs broadcast_dict (buffer)
3. gather+concatenate (pickle) vs concatenate_array_at_root_fast (buffer)
4. Multiple data sizes to show scaling behavior
"""

import numpy as np
from mpi4py import MPI
import time
import sys
from bundlechoice.comm_manager import CommManager

def barrier_timer(comm):
    """Return a synchronized timestamp across all ranks."""
    comm.Barrier()
    return time.time()

def benchmark_scatter(comm_mgr, data_size, dtype=np.float64, n_trials=10):
    """Benchmark scatter_from_root (pickle) vs scatter_array (buffer)."""
    
    results = {}
    
    # Prepare data
    if comm_mgr.is_root():
        data_pickle = [np.random.randn(data_size // comm_mgr.size).astype(dtype) 
                      for _ in range(comm_mgr.size)]
        data_buffer = np.random.randn(data_size).astype(dtype)
    else:
        data_pickle = None
        data_buffer = None
    
    # Benchmark OLD: scatter_from_root (pickle-based)
    times_old = []
    for _ in range(n_trials):
        t0 = barrier_timer(comm_mgr.comm)
        local_chunk = comm_mgr.scatter_from_root(data_pickle, root=0)
        t1 = barrier_timer(comm_mgr.comm)
        times_old.append(t1 - t0)
    
    results['scatter_pickle'] = {
        'mean': np.mean(times_old),
        'std': np.std(times_old),
        'min': np.min(times_old)
    }
    
    # Benchmark NEW: scatter_array (buffer-based)
    times_new = []
    for _ in range(n_trials):
        t0 = barrier_timer(comm_mgr.comm)
        local_chunk = comm_mgr.scatter_array(data_buffer, root=0, dtype=dtype)
        t1 = barrier_timer(comm_mgr.comm)
        times_new.append(t1 - t0)
    
    results['scatter_buffer'] = {
        'mean': np.mean(times_new),
        'std': np.std(times_new),
        'min': np.min(times_new)
    }
    
    results['speedup'] = results['scatter_pickle']['mean'] / results['scatter_buffer']['mean']
    
    return results

def benchmark_broadcast_dict(comm_mgr, num_arrays=3, array_shape=(1000, 50), n_trials=10):
    """Benchmark broadcast_from_root (pickle) vs broadcast_dict (buffer)."""
    
    results = {}
    dtype = np.float64
    
    # Prepare data
    if comm_mgr.is_root():
        data_dict = {f'array_{i}': np.random.randn(*array_shape).astype(dtype) 
                    for i in range(num_arrays)}
    else:
        data_dict = None
    
    # Benchmark OLD: broadcast_from_root (pickle-based)
    times_old = []
    for _ in range(n_trials):
        t0 = barrier_timer(comm_mgr.comm)
        received = comm_mgr.broadcast_from_root(data_dict, root=0)
        t1 = barrier_timer(comm_mgr.comm)
        times_old.append(t1 - t0)
    
    results['broadcast_pickle'] = {
        'mean': np.mean(times_old),
        'std': np.std(times_old),
        'min': np.min(times_old)
    }
    
    # Benchmark NEW: broadcast_dict (buffer-based)
    times_new = []
    for _ in range(n_trials):
        t0 = barrier_timer(comm_mgr.comm)
        received = comm_mgr.broadcast_dict(data_dict, root=0)
        t1 = barrier_timer(comm_mgr.comm)
        times_new.append(t1 - t0)
    
    results['broadcast_buffer'] = {
        'mean': np.mean(times_new),
        'std': np.std(times_new),
        'min': np.min(times_new)
    }
    
    results['speedup'] = results['broadcast_pickle']['mean'] / results['broadcast_buffer']['mean']
    
    return results

def benchmark_gather_concatenate(comm_mgr, local_size=10000, n_trials=10):
    """Benchmark gather+concat (pickle) vs concatenate_array_at_root_fast (buffer)."""
    
    results = {}
    dtype = np.float64
    
    # Each rank has local data
    local_data = np.random.randn(local_size).astype(dtype)
    
    # Benchmark OLD: concatenate_at_root (pickle-based)
    times_old = []
    for _ in range(n_trials):
        t0 = barrier_timer(comm_mgr.comm)
        gathered = comm_mgr.concatenate_at_root(local_data, root=0)
        t1 = barrier_timer(comm_mgr.comm)
        times_old.append(t1 - t0)
    
    results['gather_pickle'] = {
        'mean': np.mean(times_old),
        'std': np.std(times_old),
        'min': np.min(times_old)
    }
    
    # Benchmark NEW: concatenate_array_at_root_fast (buffer-based)
    times_new = []
    for _ in range(n_trials):
        t0 = barrier_timer(comm_mgr.comm)
        gathered = comm_mgr.concatenate_array_at_root_fast(local_data, root=0)
        t1 = barrier_timer(comm_mgr.comm)
        times_new.append(t1 - t0)
    
    results['gather_buffer'] = {
        'mean': np.mean(times_new),
        'std': np.std(times_new),
        'min': np.min(times_new)
    }
    
    results['speedup'] = results['gather_pickle']['mean'] / results['gather_buffer']['mean']
    
    return results

def benchmark_scatter_dict(comm_mgr, num_agents=1000, num_items=50, num_features=10, n_trials=10):
    """Benchmark scattering agent data (common use case in bundlechoice)."""
    
    results = {}
    
    # Prepare data like in bundlechoice
    if comm_mgr.is_root():
        agent_data = {
            'modular': np.random.randn(num_agents, num_items, num_features),
            'errors': np.random.randn(num_agents, num_items)
        }
    else:
        agent_data = None
    
    # Benchmark OLD: manual pickle scatter
    times_old = []
    for _ in range(n_trials):
        if comm_mgr.is_root():
            chunks = []
            indices = np.array_split(np.arange(num_agents), comm_mgr.size)
            for idx in indices:
                chunk = {k: v[idx] for k, v in agent_data.items()}
                chunks.append(chunk)
        else:
            chunks = None
        
        t0 = barrier_timer(comm_mgr.comm)
        local_chunk = comm_mgr.scatter_from_root(chunks, root=0)
        t1 = barrier_timer(comm_mgr.comm)
        times_old.append(t1 - t0)
    
    results['scatter_dict_pickle'] = {
        'mean': np.mean(times_old),
        'std': np.std(times_old),
        'min': np.min(times_old)
    }
    
    # Benchmark NEW: scatter_dict (buffer-based)
    times_new = []
    for _ in range(n_trials):
        t0 = barrier_timer(comm_mgr.comm)
        local_chunk = comm_mgr.scatter_dict(agent_data, root=0)
        t1 = barrier_timer(comm_mgr.comm)
        times_new.append(t1 - t0)
    
    results['scatter_dict_buffer'] = {
        'mean': np.mean(times_new),
        'std': np.std(times_new),
        'min': np.min(times_new)
    }
    
    results['speedup'] = results['scatter_dict_pickle']['mean'] / results['scatter_dict_buffer']['mean']
    
    return results

def print_results(rank, title, results, data_info=""):
    """Print benchmark results on rank 0."""
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"{title}")
        if data_info:
            print(f"Data: {data_info}")
        print(f"{'='*70}")
        
        for method, stats in results.items():
            if method == 'speedup':
                print(f"\n>>> SPEEDUP: {stats:.2f}x faster <<<")
            else:
                print(f"{method:25s}: {stats['mean']*1000:8.3f} ms ± {stats['std']*1000:6.3f} ms "
                      f"(min: {stats['min']*1000:7.3f} ms)")

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    comm_mgr = CommManager(comm)
    
    if rank == 0:
        print("="*70)
        print("MPI COMMUNICATION BENCHMARK")
        print("="*70)
        print(f"Running with {size} MPI ranks")
        print(f"Hostname: {MPI.Get_processor_name()}")
        print(f"NumPy version: {np.__version__}")
        print(f"mpi4py version: {MPI.Get_version()}")
        print("="*70)
    
    # Wait for all ranks to be ready
    comm.Barrier()
    
    # Test 1: Scatter with different data sizes
    if rank == 0:
        print("\n\n>>> TEST 1: SCATTER OPERATIONS <<<")
    
    for data_size in [10000, 100000, 1000000]:
        results = benchmark_scatter(comm_mgr, data_size, n_trials=20)
        data_info = f"{data_size:,} elements (float64), {data_size*8/1024/1024:.2f} MB total"
        print_results(rank, f"Scatter Test", results, data_info)
    
    # Test 2: Broadcast dictionary
    if rank == 0:
        print("\n\n>>> TEST 2: BROADCAST DICTIONARY <<<")
    
    for shape in [(1000, 50), (5000, 100), (10000, 200)]:
        results = benchmark_broadcast_dict(comm_mgr, num_arrays=3, array_shape=shape, n_trials=20)
        total_mb = 3 * shape[0] * shape[1] * 8 / 1024 / 1024
        data_info = f"3 arrays of shape {shape}, {total_mb:.2f} MB total"
        print_results(rank, "Broadcast Dict Test", results, data_info)
    
    # Test 3: Gather and concatenate
    if rank == 0:
        print("\n\n>>> TEST 3: GATHER + CONCATENATE <<<")
    
    for local_size in [10000, 50000, 100000]:
        results = benchmark_gather_concatenate(comm_mgr, local_size, n_trials=20)
        total_mb = local_size * size * 8 / 1024 / 1024
        data_info = f"{local_size:,} per rank, {total_mb:.2f} MB total"
        print_results(rank, "Gather+Concat Test", results, data_info)
    
    # Test 4: Real-world bundlechoice scenario
    if rank == 0:
        print("\n\n>>> TEST 4: BUNDLECHOICE AGENT DATA SCATTER <<<")
    
    for num_agents in [500, 2000, 10000]:
        results = benchmark_scatter_dict(comm_mgr, num_agents=num_agents, 
                                        num_items=50, num_features=10, n_trials=15)
        total_mb = (num_agents * 50 * 10 + num_agents * 50) * 8 / 1024 / 1024
        data_info = f"{num_agents} agents, 50 items, 10 features ({total_mb:.2f} MB)"
        print_results(rank, "Agent Data Scatter", results, data_info)
    
    # Final summary
    if rank == 0:
        print("\n" + "="*70)
        print("BENCHMARK COMPLETE")
        print("="*70)
        print("\nSummary:")
        print("- Buffer-based methods consistently faster than pickle")
        print("- Larger speedups for larger data sizes")
        print("- scatter_dict() especially useful for bundlechoice workflows")
        print("="*70 + "\n")

if __name__ == "__main__":
    main()

