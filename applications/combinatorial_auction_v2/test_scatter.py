#!/bin/env python

from bundlechoice import BundleChoice
import numpy as np
import yaml
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

BASE_DIR = os.path.dirname(__file__)
IS_LOCAL = os.path.exists("/Users/enzo-macbookpro")
CONFIG_PATH = os.path.join(BASE_DIR, "config_local.yaml" if IS_LOCAL else "config.yaml")

if rank == 0:
    print("=" * 70)
    print("TEST DATA SCATTER")
    print("=" * 70)
    print(f"MPI Size: {comm.Get_size()}")
    print(f"Config: {CONFIG_PATH}")

# Load configuration
with open(CONFIG_PATH, 'r') as file:
    config = yaml.safe_load(file)

# Load data on rank 0
if rank == 0:
    INPUT_DIR = os.path.join(BASE_DIR, "input_data")
    obs_bundle = np.load(os.path.join(INPUT_DIR, "matching_i_j.npy"))

    num_agents = config["dimensions"]["num_agents"]
    num_items = config["dimensions"]["num_items"]
    num_features = config["dimensions"]["num_features"]
    num_simuls = config["dimensions"]["num_simuls"]

    if rank == 0:
        print(f"\nData dimensions:")
        print(f"  Agents: {num_agents}")
        print(f"  Items: {num_items}")
        print(f"  Features: {num_features}")
        print(f"  Simulations: {num_simuls}")
        print(f"  Total agents (simuls × agents): {num_simuls * num_agents}")

    item_data = {
        "modular": -np.eye(num_items),
        "quadratic": np.load(os.path.join(INPUT_DIR, "quadratic_characteristic_j_j_k.npy")),
        "weights": np.load(os.path.join(INPUT_DIR, "weight_j.npy"))
    }
    agent_data = {
        "modular": np.load(os.path.join(INPUT_DIR, "modular_characteristics_i_j_k.npy")),
        "capacity": np.load(os.path.join(INPUT_DIR, "capacity_i.npy")),
    }

    np.random.seed(1995)
    errors = np.random.normal(0, 1, size=(num_simuls, num_agents, num_items))

    input_data = {
        "item_data": item_data,
        "agent_data": agent_data,
        "errors": errors,
        "obs_bundle": obs_bundle
    }
    
    if rank == 0:
        print(f"\nInput data shapes:")
        print(f"  errors: {errors.shape}")
        print(f"  obs_bundle: {obs_bundle.shape}")
        print(f"  agent_data['modular']: {agent_data['modular'].shape}")
        print(f"  agent_data['capacity']: {agent_data['capacity'].shape}")
        print(f"  item_data['modular']: {item_data['modular'].shape}")
        print(f"  item_data['quadratic']: {item_data['quadratic'].shape}")
else:
    input_data = None

# Broadcast dimensions to all ranks
num_features = comm.bcast(num_features if rank == 0 else None, root=0)

# Initialize BundleChoice and load config
if rank == 0:
    print("\nInitializing BundleChoice...")
combinatorial_auction = BundleChoice()
combinatorial_auction.load_config(CONFIG_PATH)

# Test scatter step by step
if rank == 0:
    print("\n" + "=" * 70)
    print("STEP 1: Loading input data...")
    print("=" * 70)
combinatorial_auction.data.load(input_data)

if rank == 0:
    print("\n" + "=" * 70)
    print("STEP 2: Scattering data (step by step)...")
    print("=" * 70)

# Manually call scatter steps to isolate the issue
try:
    # Prepare data like scatter() does
    if combinatorial_auction.data.is_root():
        errors = combinatorial_auction.data._prepare_errors(combinatorial_auction.data.input_data.get("errors"))
        obs_bundles = combinatorial_auction.data.input_data.get("obs_bundle")
        agent_data = combinatorial_auction.data.input_data.get("agent_data") or {}
        
        idx_chunks = np.array_split(np.arange(combinatorial_auction.data.num_simuls * combinatorial_auction.data.num_agents), 
                                    combinatorial_auction.data.comm_manager.comm.Get_size())
        counts = [len(idx) for idx in idx_chunks]
        flat_counts = [c * combinatorial_auction.data.num_items for c in counts]
        
        if rank == 0:
            print(f"\nPrepared data:")
            print(f"  errors shape: {errors.shape}")
            print(f"  errors dtype: {errors.dtype}")
            print(f"  errors is contiguous: {errors.flags['C_CONTIGUOUS']}")
            print(f"  counts: {counts}")
            print(f"  flat_counts: {flat_counts}")
            print(f"  Total elements in errors: {errors.size}")
            print(f"  Sum of flat_counts: {sum(flat_counts)}")
    else:
        errors = None
        obs_bundles = None
        agent_data = None
        counts = None
        flat_counts = None
    
    counts = comm.bcast(counts, root=0)
    flat_counts = comm.bcast(flat_counts, root=0)
    
    # Test 1: Scatter errors
    if rank == 0:
        print(f"\n{'=' * 70}")
        print("TEST 1: Scattering errors array...")
        print(f"{'=' * 70}")
        print(f"  Sending array shape: {errors.shape}")
        print(f"  Sending array dtype: {errors.dtype}")
        print(f"  Sending array size: {errors.size}")
        print(f"  Expected receive size: {flat_counts[rank]}")
    
    comm.barrier()
    if rank == 0:
        print("  Calling scatter_array...")
    
    local_errors_flat = combinatorial_auction.data.comm_manager.scatter_array(
        send_array=errors, counts=flat_counts, root=0,
        dtype=errors.dtype if rank == 0 else np.float64
    )
    
    if rank == 0:
        print(f"  SUCCESS: Received {local_errors_flat.shape} on rank 0")
    
    comm.barrier()
    
    # Test 2: Scatter obs_bundles
    if rank == 0:
        print(f"\n{'=' * 70}")
        print("TEST 2: Scattering obs_bundles array...")
        print(f"{'=' * 70}")
        # Expand obs_bundles like in the actual code
        obs_chunks = []
        for idx in idx_chunks:
            obs_chunks.append(obs_bundles[idx % combinatorial_auction.data.num_agents])
        indexed_obs_bundles = np.concatenate(obs_chunks, axis=0)
        print(f"  Expanded obs_bundles shape: {indexed_obs_bundles.shape}")
        print(f"  Expanded obs_bundles dtype: {indexed_obs_bundles.dtype}")
        print(f"  Expanded obs_bundles size: {indexed_obs_bundles.size}")
        print(f"  Expected receive size: {flat_counts[rank]}")
        print(f"  Is contiguous: {indexed_obs_bundles.flags['C_CONTIGUOUS']}")
    else:
        indexed_obs_bundles = None
    
    comm.barrier()
    if rank == 0:
        print("  Calling scatter_array...")
    
    local_obs_bundles_flat = combinatorial_auction.data.comm_manager.scatter_array(
        send_array=indexed_obs_bundles, counts=flat_counts, root=0,
        dtype=indexed_obs_bundles.dtype if rank == 0 else np.bool_
    )
    
    if rank == 0:
        print(f"  SUCCESS: Received {local_obs_bundles_flat.shape} on rank 0")
    
    comm.barrier()
    if rank == 0:
        print(f"\n{'=' * 70}")
        print("All scatter operations completed successfully!")
        print(f"{'=' * 70}")
        
except Exception as e:
    if rank == 0:
        print(f"\n{'=' * 70}")
        print(f"ERROR: {type(e).__name__}: {e}")
        print(f"{'=' * 70}")
    import traceback
    if rank == 0:
        traceback.print_exc()
    comm.Abort(1)

if rank == 0:
    print("\n" + "=" * 70)
    print("Test completed successfully!")
    print("=" * 70)

