#!/bin/env python

import numpy as np

# Monkey patch compute_rank_features to handle empty arrays
# WARNING: This modifies FeatureManager.compute_rank_features at runtime.
# This patch is necessary to handle ranks with 0 local agents when using many MPI ranks.
# If bundlechoice library is updated, this patch may need to be adjusted.
from bundlechoice.feature_manager import FeatureManager
_original_compute_rank_features = FeatureManager.compute_rank_features

def _patched_compute_rank_features(self, local_bundles):
    """Patched version that handles empty arrays for ranks with 0 local agents."""
    if self.num_local_agents == 0 or len(local_bundles) == 0:
        return np.empty((0, self.num_features), dtype=np.float64)
    return _original_compute_rank_features(self, local_bundles)

FeatureManager.compute_rank_features = _patched_compute_rank_features

from bundlechoice import BundleChoice
import yaml
from mpi4py import MPI
import os
import pandas as pd

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Warn about monkey patch (after rank is defined)
if rank == 0:
    print("WARNING: Monkey patch active for FeatureManager.compute_rank_features (handles 0-agent ranks)")

BASE_DIR = os.path.dirname(__file__)

# Configuration for SE computation (separate from estimation)
NUM_SIMULS_SE = 3  # Number of simulations for SE computation
SKIP_VALIDATION = os.environ.get("SKIP_VALIDATION", "0") == "1"  # Set SKIP_VALIDATION=1 to skip validation

# Load theta from estimation
if rank == 0:
    OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")
    theta_path = os.path.join(OUTPUT_DIR, "theta.npy")
    theta_hat = np.load(theta_path)
    print(f"Loaded theta from {theta_path}")
    print(f"Theta shape: {theta_hat.shape}")
else:
    theta_hat = None

# Broadcast theta to all ranks
theta_hat = comm.bcast(theta_hat, root=0)

# Load data (same as estimation, but with SE-specific num_simuls)
if rank == 0:
    INPUT_DIR = os.path.join(BASE_DIR, "input_data")
    obs_bundle = np.load(os.path.join(INPUT_DIR, "matching_i_j.npy"))

    # Get dimensions from loaded data
    quadratic = np.load(os.path.join(INPUT_DIR, "quadratic_characteristic_j_j_k.npy"))
    weights = np.load(os.path.join(INPUT_DIR, "weight_j.npy"))
    modular_agent = np.load(os.path.join(INPUT_DIR, "modular_characteristics_i_j_k.npy"))
    capacity = np.load(os.path.join(INPUT_DIR, "capacity_i.npy"))
    
    num_agents = capacity.shape[0]
    num_items = weights.shape[0]
    num_features = modular_agent.shape[2] + num_items + quadratic.shape[2]

    item_data = {
        "modular": -np.eye(num_items),
        "quadratic": quadratic,
        "weights": weights
    }
    agent_data = {
        "modular": modular_agent,
        "capacity": capacity
    }

    # Generate errors for all simulations (will loop over them manually)
    np.random.seed(1995)
    errors_all_sims = np.random.normal(0, 1, size=(NUM_SIMULS_SE, num_agents, num_items))
    
    # For BundleChoice initialization, use first simulation's errors (library requires num_simuls=1)
    errors = errors_all_sims[0]  # Shape: (num_agents, num_items)

    input_data = {
        "item_data": item_data,
        "agent_data": agent_data,
        "errors": errors,
        "obs_bundle": obs_bundle
    }
    
    # Store errors_all_sims for later use
    errors_all_sims_storage = errors_all_sims
else:
    input_data = None
    num_agents = None
    num_items = None
    num_features = None
    errors_all_sims_storage = None

num_agents = comm.bcast(num_agents, root=0)
num_items = comm.bcast(num_items, root=0)
num_features = comm.bcast(num_features, root=0)
errors_all_sims = comm.bcast(errors_all_sims_storage, root=0)

# Initialize BundleChoice with SE-specific config
# NOTE: Must use num_simuls=1 due to library bug with agent_data expansion for num_simuls > 1
config_se = {
    "dimensions": {
        "num_agents": num_agents,
        "num_items": num_items,
        "num_features": num_features,
        "num_simuls": 1  # Must be 1 due to library limitation
    },
    "subproblem": {
        "name": "QuadKnapsack",
        "settings": {
            "TimeLimit": 1,  # 1 second per agent for faster testing
            "MIPGap_tol": 1e-2
        }
    },
    "row_generation": {
        "max_iters": 400,
        "theta_ubs": 1000
    }
}

bc = BundleChoice()
bc.load_config(config_se)
bc.data.load_and_scatter(input_data)
bc.features.build_from_data()
bc.subproblems.load()
bc.subproblems.initialize_local()

# -------------------------------
# OPTIONAL: warm-start plumbing
# -------------------------------
# We cannot guarantee how bundlechoice builds Gurobi models internally.
# But if subproblem objects expose a per-agent Gurobi model + decision vars,
# we can (a) cache the model (avoid rebuilding) and (b) feed the previous
# incumbent solution as a MIP start when resolving the same model with
# updated objective coefficients (theta/errors).
#
# Enable with:   export USE_WARM_START=1
# Disable with:  export USE_WARM_START=0
USE_WARM_START = os.environ.get("USE_WARM_START", "0") == "1"

# Cache last local bundles to use as a MIP start on the next solve
_last_local_bundles_start = None


def _try_set_mip_start_from_bundles(subproblems_obj, bundles_bool):
    """Best-effort MIP start.

    This relies on internal bundlechoice objects exposing (model, vars) per local agent.
    If internals differ, we silently do nothing.

    Expected mapping:
      - one model per local agent
      - binary var x[j] indicates whether item j is chosen
      - bundles_bool has shape (num_local_agents, num_items)

    If available, we set Var.Start for each x[j] and call model.update().
    """
    try:
        # Common patterns we have seen in similar codebases; adjust if needed.
        # 1) subproblems_obj.local_subproblems: list of per-agent subproblem instances
        local_sps = getattr(subproblems_obj, "local_subproblems", None)
        if local_sps is None:
            local_sps = getattr(subproblems_obj, "_local_subproblems", None)
        if local_sps is None:
            return

        for a, sp in enumerate(local_sps):
            model = getattr(sp, "model", None)
            if model is None:
                model = getattr(sp, "_model", None)
            xvars = getattr(sp, "x", None)
            if xvars is None:
                xvars = getattr(sp, "_x", None)
            if model is None or xvars is None:
                continue

            # xvars may be a list or dict-like indexed by item id
            row = bundles_bool[a]
            if hasattr(xvars, "__len__") and len(xvars) == row.shape[0]:
                for j in range(row.shape[0]):
                    try:
                        xvars[j].Start = float(row[j])
                    except Exception:
                        pass
            else:
                # dict-like
                for j in range(row.shape[0]):
                    try:
                        xvars[j].Start = float(row[j])
                    except Exception:
                        pass

            try:
                model.update()
            except Exception:
                pass

    except Exception:
        # Warm-start is best-effort; never crash SE computation.
        return


# Wrap solve_local to inject MIP starts when enabled
_original_solve_local = bc.subproblems.solve_local

def _solve_local_with_warm_start(theta):
    global _last_local_bundles_start
    if USE_WARM_START and _last_local_bundles_start is not None:
        _try_set_mip_start_from_bundles(bc.subproblems, _last_local_bundles_start)
    bundles = _original_solve_local(theta)
    # Cache for next call (only local bundles, not gathered)
    _last_local_bundles_start = bundles
    return bundles

bc.subproblems.solve_local = _solve_local_with_warm_start

if rank == 0 and USE_WARM_START:
    print("\nWARM START enabled: reusing previous bundles as MIP starts when possible")

def update_errors_for_simulation(bc, errors_sim):
    """
    Update errors in data_manager for a specific simulation (num_simuls=1 case).
    
    CRITICAL: This function mutates bc.data_manager.local_data["errors"] and assumes
    that solve_local() uses the errors from local_data. If errors are cached in subproblem
    objects, this will silently fail.
    """
    # errors_sim shape: (num_agents, num_items)
    num_agents = bc.data_manager.num_agents
    num_items = bc.data_manager.num_items
    num_local_agents = bc.data_manager.num_local_agents
    
    if rank == 0:
        errors_flat = errors_sim.reshape(-1)  # Flatten to (num_agents * num_items,)
        dtype = errors_flat.dtype
        # Compute counts matching data_manager's distribution
        # data_manager uses np.array_split, so we do the same
        size = comm.Get_size()
        idx_chunks = np.array_split(np.arange(num_agents), size)
        counts = [len(chunk) * num_items for chunk in idx_chunks]  # flat_counts
    else:
        errors_flat = None
        dtype = np.float64
        counts = None
    
    # Use scatter_array with explicit counts to match data_manager distribution
    local_errors_flat = bc.comm_manager.scatter_array(errors_flat, counts=counts, root=0, dtype=dtype)
    
    # Handle ranks with 0 agents
    if num_local_agents == 0:
        local_errors = np.empty((0, num_items), dtype=dtype)
    else:
        local_errors = local_errors_flat.reshape(num_local_agents, num_items)
    bc.data_manager.local_data["errors"] = local_errors

# VALIDATION: Verify that updating errors actually affects solve_local results
# This is critical - if errors are cached, update_errors_for_simulation is useless
if SKIP_VALIDATION:
    if rank == 0:
        print("\nSKIPPING VALIDATION (SKIP_VALIDATION=1 set)")
    # Restore errors for actual computation (broadcast first error set to all ranks)
    errors_first = comm.bcast(errors_all_sims_storage[0] if rank == 0 else None, root=0)
    update_errors_for_simulation(bc, errors_first)
else:
    if rank == 0:
        print("\nVALIDATION: Testing that error updates affect solve_local...")
    # Create two different error draws
    if rank == 0:
        test_errors_0 = np.random.RandomState(999).normal(0, 1, size=(num_agents, num_items))
        test_errors_1 = np.random.RandomState(888).normal(0, 1, size=(num_agents, num_items))
    else:
        test_errors_0 = None
        test_errors_1 = None

    # Broadcast test errors to all ranks
    test_errors_0 = comm.bcast(test_errors_0, root=0)
    test_errors_1 = comm.bcast(test_errors_1, root=0)

    # Update to first errors and solve (all ranks need to participate)
    update_errors_for_simulation(bc, test_errors_0)
    if rank == 0:
        print(f"  Solving with first error set (rank 0: {bc.data_manager.num_local_agents} agents)...", flush=True)
    if bc.data_manager.num_local_agents > 0:
        test_bundles_0 = bc.subproblems.solve_local(theta_hat)
        if rank == 0:
            print(f"  ✓ First solve completed", flush=True)
    else:
        test_bundles_0 = np.empty((0, bc.data_manager.num_items), dtype=bool)
    if rank == 0:
        print(f"  Computing features for first solve...", flush=True)
    test_features_0 = bc.feature_manager.compute_gathered_features(test_bundles_0)

    # Update to second errors and solve
    update_errors_for_simulation(bc, test_errors_1)
    if rank == 0:
        print(f"  Solving with second error set...", flush=True)
    if bc.data_manager.num_local_agents > 0:
        test_bundles_1 = bc.subproblems.solve_local(theta_hat)
        if rank == 0:
            print(f"  ✓ Second solve completed", flush=True)
    else:
        test_bundles_1 = np.empty((0, bc.data_manager.num_items), dtype=bool)
    if rank == 0:
        print(f"  Computing features for second solve...", flush=True)
    test_features_1 = bc.feature_manager.compute_gathered_features(test_bundles_1)

    # Check that results differ (only on rank 0)
    if rank == 0:
        if test_features_0 is None or test_features_1 is None:
            raise RuntimeError("VALIDATION FAILED: features are None on rank 0")
        diff_norm = np.linalg.norm(test_features_0 - test_features_1)
        if diff_norm < 1e-10:
            raise RuntimeError(
                f"VALIDATION FAILED: Updating errors does not affect solve_local results!\n"
                f"||features_0 - features_1|| = {diff_norm:.2e} (expected > 0)\n"
                f"This means errors are likely cached in subproblem objects. SEs will be WRONG."
            )
        print(f"✓ Validation passed: ||features_diff|| = {diff_norm:.2e} (errors affect results)")

    # SCATTER VALIDATION: Verify that errors are scattered correctly
    # We check both sum (catches major issues) and sum of squares (catches indexing bugs)
    if rank == 0:
        print("\nVALIDATION: Testing scatter correctness...")
        # Compute expected sums from errors_all_sims[0]
        expected_total_sum = errors_all_sims_storage[0].sum()
        expected_total_sum_sq = (errors_all_sims_storage[0] ** 2).sum()
        test_errors_scatter = errors_all_sims_storage[0]
    else:
        expected_total_sum = None
        expected_total_sum_sq = None
        test_errors_scatter = None

    # Broadcast test_errors_scatter to all ranks
    test_errors_scatter = comm.bcast(test_errors_scatter, root=0)

    # Update errors and compute local sums
    update_errors_for_simulation(bc, test_errors_scatter)
    if bc.data_manager.num_local_agents > 0:
        local_errors = bc.data_manager.local_data["errors"]
        local_sum = local_errors.sum()
        local_sum_sq = (local_errors ** 2).sum()
    else:
        local_sum = 0.0
        local_sum_sq = 0.0

    # Gather sums from all ranks
    all_sums = comm.allgather(local_sum)
    all_sums_sq = comm.allgather(local_sum_sq)
    total_sum = sum(all_sums)
    total_sum_sq = sum(all_sums_sq)

    if rank == 0:
        if not np.allclose(total_sum, expected_total_sum, rtol=1e-10):
            raise RuntimeError(
                f"VALIDATION FAILED: Scatter sum mismatch!\n"
                f"Expected total sum: {expected_total_sum:.6e}, Got: {total_sum:.6e}\n"
                f"This means errors are not scattered correctly. SEs will be WRONG."
            )
        if not np.allclose(total_sum_sq, expected_total_sum_sq, rtol=1e-10):
            raise RuntimeError(
                f"VALIDATION FAILED: Scatter sum-of-squares mismatch!\n"
                f"Expected total sum^2: {expected_total_sum_sq:.6e}, Got: {total_sum_sq:.6e}\n"
                f"This indicates indexing/ordering issues in scatter. SEs will be WRONG."
            )
        print(f"✓ Validation passed: scatter sum and sum-of-squares match (sum={total_sum:.6e})")

    # Restore errors for actual computation (broadcast first error set to all ranks)
    errors_first = comm.bcast(errors_all_sims_storage[0] if rank == 0 else None, root=0)
    update_errors_for_simulation(bc, errors_first)

# Extract beta indices (non-FE parameters)
if rank == 0:
    num_modular_agent = agent_data["modular"].shape[2]
    num_modular_item = item_data["modular"].shape[1]
    num_quadratic = item_data["quadratic"].shape[2]
    
    # Safety check: Beta indexing assumes exactly one modular-agent coefficient
    assert num_modular_agent == 1, \
        f"Beta indexing assumes exactly one modular-agent coefficient, got {num_modular_agent}"
    
    # I_FE = [num_modular_agent : num_modular_agent + num_modular_item]
    # I_beta = indices not in I_FE
    beta_indices = np.concatenate([
        np.arange(num_modular_agent),  # modular agent features
        np.arange(num_modular_agent + num_modular_item, num_features)  # quadratic features
    ]).astype(int)
    print(f"\nBeta indices (non-FE): {beta_indices}")
    print(f"Number of beta parameters: {len(beta_indices)}")
else:
    beta_indices = None

beta_indices = comm.bcast(beta_indices, root=0)

# Compute per-agent observed features directly
obs_bundles_local = bc.data_manager.local_data.get("obs_bundles")
if obs_bundles_local is None:
    raise RuntimeError("obs_bundles not found in data_manager.local_data")
agents_obs_features = bc.feature_manager.compute_gathered_features(obs_bundles_local)

# Verify observed features shape on rank 0
if rank == 0:
    if agents_obs_features is None:
        raise RuntimeError("agents_obs_features is None on rank 0")
    if agents_obs_features.shape != (num_agents, num_features):
        raise RuntimeError(
            f"agents_obs_features has wrong shape: expected ({num_agents}, {num_features}), "
            f"got {agents_obs_features.shape}"
        )
    print(f"Verified: agents_obs_features.shape = {agents_obs_features.shape}")

# AGENT ORDERING VALIDATION: Verify that compute_gathered_features returns rows in
# agent-ID order (0..N-1). This is CRITICAL because g_i[i] must correspond to agent i.
# If rows are permuted, we pair agent i's observed features with agent j's simulated features,
# making g_i wrong → B matrix wrong → sandwich SE wrong.
#
# Test strategy: Use bundle-dependent feature signature (item modular feature)
# - On each rank r, only the FIRST local agent chooses item 0 (others choose empty bundle)
# - Item modular features are -I (identity), so item 0 chosen sets feature[modular_agent_dim + 0] = -1
# - Verify that in gathered features, row[start_r] has item-0 feature set, rows[start_r+1:end_r] do not
# - This validates that row i corresponds to agent i
if not SKIP_VALIDATION:
    if rank == 0:
        print("\nVALIDATION: Testing agent ordering (bundle-dependent signature test)...")

    # Determine agent distribution across ranks (same as data_manager uses)
    size = comm.Get_size()
    idx_chunks = np.array_split(np.arange(num_agents), size)
    rank_start_indices = [chunk[0] if len(chunk) > 0 else num_agents for chunk in idx_chunks]
    rank_end_indices = [chunk[-1] + 1 if len(chunk) > 0 else num_agents for chunk in idx_chunks]

    # Get rank-specific info
    my_rank = comm.Get_rank()
    num_local_agents = bc.data_manager.num_local_agents
    num_items = bc.data_manager.num_items

    # Create test bundles: first local agent on each rank chooses item 0, others choose empty bundle
    test_bundles_ordering = np.zeros((num_local_agents, num_items), dtype=bool)
    if num_local_agents > 0:
        test_bundles_ordering[0, 0] = True  # First local agent chooses item 0

    # Compute features with these test bundles
    test_features_ordering = bc.feature_manager.compute_gathered_features(test_bundles_ordering)

    # On rank 0, verify ordering using bundle-dependent feature signature
    if rank == 0:
        if test_features_ordering is None:
            raise RuntimeError("VALIDATION FAILED: test_features_ordering is None")
        
        # Find the feature index for "item 0 chosen" in modular item features
        # Feature order: [modular_agent_features, modular_item_features, quadratic_features]
        # Modular item features are -I (negative identity), so item 0 corresponds to feature index: num_modular_agent + 0
        # We need to get num_modular_agent from the earlier definition (same as used for beta_indices)
        # Recompute from agent_data which is in scope here (defined on rank 0 earlier)
        num_modular_agent_check = agent_data["modular"].shape[2]
        item0_feature_idx = num_modular_agent_check + 0  # First modular item feature (item 0)
        
        # Verify ordering: for each rank block [start:end),
        # row[start] should have item0 feature set (negative), rows[start+1:end] should not (zero)
        all_good = True
        for r in range(size):
            start_idx = rank_start_indices[r]
            end_idx = rank_end_indices[r]
            if start_idx >= num_agents:
                continue
            
            # First agent on this rank should have item0 feature set (should be -1.0 for modular item)
            first_agent_item0_feature = test_features_ordering[start_idx, item0_feature_idx]
            if np.isclose(first_agent_item0_feature, 0.0, rtol=1e-10, atol=1e-10):
                all_good = False
                print(f"  FAIL: Rank {r}'s first agent (global id {start_idx}, row {start_idx}) "
                      f"has item0 feature = {first_agent_item0_feature:.6e} (expected < 0)")
            
            # All other agents on this rank should NOT have item0 feature set (should be 0.0)
            if end_idx > start_idx + 1:
                other_agents_item0_features = test_features_ordering[start_idx + 1:end_idx, item0_feature_idx]
                if not np.allclose(other_agents_item0_features, 0.0, rtol=1e-10, atol=1e-10):
                    all_good = False
                    print(f"  FAIL: Rank {r}'s agents {start_idx+1}..{end_idx-1} have item0 features != 0 "
                          f"(expected all 0): {other_agents_item0_features}")
        
        if not all_good:
            raise RuntimeError(
                "VALIDATION FAILED: Agent ordering is incorrect!\n"
                "compute_gathered_features does not return rows in agent ID order (0..N-1).\n"
                "This means per-agent subgradients g_i will be misaligned → B matrix wrong → sandwich SE WRONG."
            )
        print("✓ Validation passed: agent ordering is correct (bundle-dependent signature verified)")

def compute_per_agent_subgradients(bc, theta, errors_all_sims):
    """Compute per-agent subgradients ĝ_i(θ) = (1/S) ∑_s x_{i,B_i^{s,*}(θ)} − x_{i B̂_i}"""
    num_agents = bc.data_manager.num_agents
    num_features = bc.data_manager.num_features
    num_simuls = len(errors_all_sims)
    
    # Loop over simulations, solving and accumulating features
    all_features_per_sim = []
    for s in range(num_simuls):
        if rank == 0:
            print(f"  Simulation {s+1}/{num_simuls}: Updating errors...", flush=True)
        # Update errors for this simulation
        update_errors_for_simulation(bc, errors_all_sims[s])
        
        # Solve with updated errors (only if rank has agents)
        if rank == 0:
            print(f"  Simulation {s+1}/{num_simuls}: Solving subproblems (rank 0: {bc.data_manager.num_local_agents} agents)...", flush=True)
        if bc.data_manager.num_local_agents > 0:
            local_bundles = bc.subproblems.solve_local(theta)
            if rank == 0:
                print(f"  Simulation {s+1}/{num_simuls}: ✓ Solve completed", flush=True)
        else:
            # Rank with 0 agents: create empty array with correct shape
            local_bundles = np.empty((0, bc.data_manager.num_items), dtype=bool)
        
        # Gather features for this simulation
        if rank == 0:
            print(f"  Simulation {s+1}/{num_simuls}: Computing features...", flush=True)
        features_sim = bc.feature_manager.compute_gathered_features(local_bundles)
        
        if rank == 0:
            all_features_per_sim.append(features_sim)
            print(f"  Simulation {s+1}/{num_simuls}: ✓ Complete", flush=True)
    
    if rank == 0:
        # Stack features from all simulations: shape (num_simuls, num_agents, num_features)
        features_all = np.stack(all_features_per_sim, axis=0)  # (num_simuls, num_agents, num_features)
        
        # Average over simulations
        avg_simulated_features = features_all.mean(axis=0)  # (num_agents, num_features)
        
        # Subtract observed features: − x_{i B̂_i}
        per_agent_subgradients = avg_simulated_features - agents_obs_features  # (num_agents, num_features)
        
        return per_agent_subgradients
    else:
        return None

# Compute per-agent subgradients at theta_hat
if rank == 0:
    import time
    start_time = time.time()
    num_local_agents_rank0 = bc.data_manager.num_local_agents
    estimated_seconds = NUM_SIMULS_SE * num_local_agents_rank0 * 1.5  # 1s TimeLimit + overhead
    estimated_minutes = estimated_seconds / 60
    print(f"\nComputing per-agent subgradients at theta_hat (using {NUM_SIMULS_SE} simulations)...")
    print(f"  Rank 0: {num_local_agents_rank0} agents")
    print(f"  Estimated time: ~{estimated_seconds:.0f}s (~{estimated_minutes:.1f} min) per rank (worst case with 1s TimeLimit)", flush=True)
    if estimated_minutes > 15:
        print(f"  WARNING: Estimated time ({estimated_minutes:.1f} min) exceeds 15 minutes!", flush=True)
    else:
        print(f"  Proceeding (estimated {estimated_minutes:.1f} min < 15 min limit)", flush=True)
g_i_theta_hat = compute_per_agent_subgradients(bc, theta_hat, errors_all_sims)
if rank == 0:
    elapsed = time.time() - start_time
    print(f"  ✓ Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)", flush=True)

# Sanity check: average subgradient should be small
if rank == 0:
    g_bar_norm = np.linalg.norm(g_i_theta_hat.mean(axis=0))
    print(f"Average subgradient norm at theta_hat: {g_bar_norm:.6e}")

# Compute B matrix: B̂_β = (1/N) ∑_i ĝ_i^β(θ̂) ĝ_i^β(θ̂)ᵀ
if rank == 0:
    g_i_beta = g_i_theta_hat[:, beta_indices]  # (num_agents, num_beta)
    B_beta = (g_i_beta.T @ g_i_beta) / num_agents  # (num_beta, num_beta)
    print(f"\nB_beta shape: {B_beta.shape}")
    cond_B = np.linalg.cond(B_beta)
    print(f"B_beta condition number: {cond_B:.2e}")

def compute_avg_subgradient(bc, theta, errors_all_sims, verbose=False):
    """Compute average subgradient ĝ̄(θ) = (1/N) ∑_i ĝ_i(θ) without storing per-agent arrays"""
    num_agents = bc.data_manager.num_agents
    num_features = bc.data_manager.num_features
    num_simuls = len(errors_all_sims)
    
    # Loop over simulations, solving and accumulating features
    all_features_per_sim = []
    for s in range(num_simuls):
        if verbose and rank == 0:
            print(f"      Sim {s+1}/{num_simuls}: solving...", flush=True)
        # Update errors for this simulation
        update_errors_for_simulation(bc, errors_all_sims[s])
        
        # Solve with updated errors (only if rank has agents)
        if bc.data_manager.num_local_agents > 0:
            local_bundles = bc.subproblems.solve_local(theta)
        else:
            # Rank with 0 agents: create empty array with correct shape
            local_bundles = np.empty((0, bc.data_manager.num_items), dtype=bool)
        
        # Gather features for this simulation
        features_sim = bc.feature_manager.compute_gathered_features(local_bundles)
        
        if rank == 0:
            all_features_per_sim.append(features_sim)
    
    if rank == 0:
        # Stack and average over all (simulations, agents)
        features_all = np.stack(all_features_per_sim, axis=0)  # (num_simuls, num_agents, num_features)
        mean_sim = features_all.mean(axis=(0, 1))  # (num_features,) - mean over all sims and agents
        mean_obs = agents_obs_features.mean(axis=0)  # (num_features,)
        avg_subgradient = mean_sim - mean_obs  # (num_features,)
        return avg_subgradient
    else:
        return None

def compute_avg_subgradient_beta(bc, theta, beta_indices, errors_all_sims, verbose=False):
    """Compute average subgradient ĝ̄^β(θ) = (1/N) ∑_i ĝ_i^β(θ)"""
    g_bar = compute_avg_subgradient(bc, theta, errors_all_sims, verbose=verbose)
    if rank == 0:
        return g_bar[beta_indices]  # (num_beta,)
    else:
        return None

# Compute A matrix using finite differences
if rank == 0:
    import time
    start_time_A = time.time()
    print("\nComputing A matrix via finite differences...")
    num_beta = len(beta_indices)
    # Each column requires 2 calls to compute_avg_subgradient_beta (forward/backward differences)
    # Each call does NUM_SIMULS_SE simulations × num_local_agents solves
    num_local_agents_rank0 = bc.data_manager.num_local_agents
    estimated_per_column = NUM_SIMULS_SE * num_local_agents_rank0 * 1.5 * 2  # 2 for forward/backward
    estimated_total_A = estimated_per_column * num_beta
    estimated_minutes_A = estimated_total_A / 60
    print(f"  Computing {num_beta} columns (finite differences)")
    print(f"  Estimated time: ~{estimated_total_A:.0f}s (~{estimated_minutes_A:.1f} min) per rank", flush=True)
    if estimated_minutes_A > 15:
        print(f"  WARNING: Estimated time ({estimated_minutes_A:.1f} min) exceeds 15 minutes!", flush=True)
    else:
        print(f"  Proceeding (estimated {estimated_minutes_A:.1f} min < 15 min limit)", flush=True)
    A_beta = np.zeros((num_beta, num_beta))
    step_size_base = 1e-4
    
    for k_idx, k in enumerate(beta_indices):
        if rank == 0:
            print(f"  Column {k_idx+1}/{num_beta} (parameter index {k})...", flush=True)
        h_k = step_size_base * max(1.0, abs(theta_hat[k]))
        
        # Compute finite differences with step size adaptation
        max_retries = 2
        for retry in range(max_retries + 1):
            theta_plus = theta_hat.copy()
            theta_plus[k] += h_k
            theta_minus = theta_hat.copy()
            theta_minus[k] -= h_k
            
            if rank == 0:
                print(f"    Computing forward difference (theta+{h_k:.2e})...", flush=True)
            g_plus = compute_avg_subgradient_beta(bc, theta_plus, beta_indices, errors_all_sims, verbose=True)
            if rank == 0:
                print(f"    Computing backward difference (theta-{h_k:.2e})...", flush=True)
            g_minus = compute_avg_subgradient_beta(bc, theta_minus, beta_indices, errors_all_sims, verbose=True)
            
            diff = g_plus - g_minus
            if np.allclose(diff, 0.0, atol=1e-8) and retry < max_retries:
                # Step size adaptation for locally flat regions
                h_k *= 10
                continue
            
            A_beta[:, k_idx] = diff / (2 * h_k)
            if rank == 0:
                print(f"    ✓ Column {k_idx+1}/{num_beta} complete", flush=True)
            break
    
    elapsed_A = time.time() - start_time_A
    print(f"  ✓ A matrix completed in {elapsed_A:.1f}s ({elapsed_A/60:.1f} min)", flush=True)
    print(f"A_beta shape: {A_beta.shape}")
    cond_A = np.linalg.cond(A_beta)
    print(f"A_beta condition number: {cond_A:.2e}")
    if cond_A > 1e12:
        print("WARNING: A_beta ill-conditioned; SEs may be unstable")

# Compute sandwich variance: V̂_β = (1/N) · Â_β⁻¹ B̂_β Â_β⁻¹
if rank == 0:
    print("\nComputing sandwich variance...")
    try:
        A_inv = np.linalg.solve(A_beta, np.eye(num_beta))
        V_beta = (1.0 / num_agents) * (A_inv @ B_beta @ A_inv.T)
    except np.linalg.LinAlgError:
        print("Warning: A_beta is singular, using pseudoinverse")
        A_inv = np.linalg.pinv(A_beta)
        V_beta = (1.0 / num_agents) * (A_inv @ B_beta @ A_inv.T)
    
    se_beta = np.sqrt(np.diag(V_beta))
    
    print(f"\nStandard errors for beta parameters:")
    for k_idx, k in enumerate(beta_indices):
        print(f"  Index {k}: SE = {se_beta[k_idx]:.6f}, Theta = {theta_hat[k]:.6f}")
    
    # Save results
    OUTPUT_DIR = os.path.join(BASE_DIR, "estimation_results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Create DataFrame with results
    df = pd.DataFrame({
        "feature_index": beta_indices,
        "theta": theta_hat[beta_indices],
        "se": se_beta,
        "t_stat": theta_hat[beta_indices] / se_beta,
    })
    
    se_csv_path = os.path.join(OUTPUT_DIR, "sandwich_se.csv")
    df.to_csv(se_csv_path, index=False)
    print(f"\nSaved standard errors to {se_csv_path}")
    
    # Save full variance matrix
    V_beta_path = os.path.join(OUTPUT_DIR, "sandwich_variance_beta.npy")
    np.save(V_beta_path, V_beta)
    print(f"Saved variance matrix to {V_beta_path}")
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "sandwich_se_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Sandwich Standard Errors Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of beta parameters: {num_beta}\n")
        f.write(f"A_beta condition number: {np.linalg.cond(A_beta):.2e}\n")
        f.write(f"B_beta condition number: {np.linalg.cond(B_beta):.2e}\n\n")
        f.write("Parameter Estimates and Standard Errors:\n")
        for k_idx, k in enumerate(beta_indices):
            f.write(f"  Index {k}: Theta = {theta_hat[k]:.6f}, SE = {se_beta[k_idx]:.6f}, t-stat = {theta_hat[k]/se_beta[k_idx]:.6f}\n")
    
    print(f"Saved summary to {summary_path}")

