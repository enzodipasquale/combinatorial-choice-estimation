#!/usr/bin/env python3
"""
Data preparation script for combinatorial auction estimation.

Processes raw data from the Fox-Bajari replication package into the format
required by BundleChoice framework.

Usage:
    python prepare_data.py              # Uses default delta=4
    python prepare_data.py --delta 2    # Uses delta=2
    python prepare_data.py --delta 4    # Uses delta=4 (explicit)
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from datetime import datetime

# Data paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "114402-V1" / "Replication-Fox-and-Bajari" / "data"

def get_output_dir(delta: int, winners_only: bool = False, hq_distance: bool = False) -> Path:
    """Get output directory based on parameters."""
    suffix = f"delta{delta}"
    if winners_only:
        suffix += "_winners"
    if hq_distance:
        suffix += "_hqdist"
    return BASE_DIR / "input_data" / suffix

# Processing parameters
WEIGHT_ROUNDING_TICK = 1000
POP_CENTROID_PERCENTILE = 0  # Truncation threshold


def load_raw_data() -> dict:
    """Load all raw data files from the replication package."""
    print("Loading raw data files...")
    
    # BTA data (items/markets)
    bta_data = pd.read_csv(DATA_DIR / "btadata_2004_03_12_1.csv")
    print(f"  Loaded BTA data: {len(bta_data)} markets")
    
    # Bidder data (agents)
    bidder_data = pd.read_csv(DATA_DIR / "biddercblk_03_28_2004_pln.csv")
    print(f"  Loaded bidder data: {len(bidder_data)} bidders")
    
    # Adjacency matrix
    bta_adjacency = pd.read_csv(DATA_DIR / "btamatrix_merged.csv", header=None)
    bta_adjacency = bta_adjacency.drop(bta_adjacency.columns[0], axis=1)
    bta_adjacency_j_j = bta_adjacency.values.astype(float)
    print(f"  Loaded adjacency matrix: {bta_adjacency_j_j.shape}")
    
    # Geographic distance matrix
    geo_distance = pd.read_csv(
        DATA_DIR / "distancesmat_dio_perl_fixed.dat",
        delimiter=' ',
        header=None
    )
    geo_distance = geo_distance.drop(geo_distance.columns[-1], axis=1)
    geo_distance_j_j = geo_distance.values.astype(float)
    print(f"  Loaded geographic distance matrix: {geo_distance_j_j.shape}")
    
    # Travel survey matrix (read with header=None to get full 493x493 matrix)
    travel_survey = pd.read_csv(DATA_DIR / "american-travel-survey-1995-zero.csv", header=None)
    travel_survey_j_j = travel_survey.values.astype(float)
    print(f"  Loaded travel survey matrix: {travel_survey_j_j.shape}")
    
    # Air travel matrix (read with header=None to get full 493x493 matrix)
    air_travel = pd.read_csv(DATA_DIR / "air-travel-passengers-bta-year-1994.csv", header=None)
    air_travel_j_j = air_travel.values.astype(float)
    print(f"  Loaded air travel matrix: {air_travel_j_j.shape}")
    
    return {
        "bta_data": bta_data,
        "bidder_data": bidder_data,
        "bta_adjacency_j_j": bta_adjacency_j_j,
        "geo_distance_j_j": geo_distance_j_j,
        "travel_survey_j_j": travel_survey_j_j,
        "air_travel_j_j": air_travel_j_j,
    }


def process_weights_capacities(bidder_data: pd.DataFrame, bta_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Round weights and capacities to reduce encoding length."""
    # Handle NaN values in capacities
    capacity_i = bidder_data['pops_eligible'].fillna(0).to_numpy()
    weight_j = bta_data['pop90'].to_numpy()
    
    weight_j_rounded = np.round(weight_j / WEIGHT_ROUNDING_TICK).astype(int)
    capacity_i_rounded = np.round(capacity_i / WEIGHT_ROUNDING_TICK).astype(int)
    
    print(f"Processed weights/capacities:")
    print(f"  Num items: {len(weight_j_rounded)}, weight range: [{weight_j_rounded.min()}, {weight_j_rounded.max()}]")
    print(f"  Num agents: {len(capacity_i_rounded)}, capacity range: [{capacity_i_rounded.min()}, {capacity_i_rounded.max()}]")
    
    return weight_j_rounded, capacity_i_rounded


def generate_matching_matrix(
    bta_data: pd.DataFrame,
    num_agents: int,
    bidder_num_to_index: dict,
) -> np.ndarray:
    """Generate matching matrix from observed winners."""
    num_items = len(bta_data)
    matching_i_j = np.zeros((num_agents, num_items), dtype=bool)
    
    for j in range(num_items):
        winner_bidder_num = bta_data['bidder_num_fox'].values[j]
        # Skip FCC (9999) or any bidders not in our filtered list
        if winner_bidder_num in bidder_num_to_index:
            winner_id = bidder_num_to_index[winner_bidder_num]
            matching_i_j[winner_id, j] = True
    
    num_winners = np.unique(np.where(matching_i_j)[0]).size
    print(f"Generated matching matrix:")
    print(f"  Shape: {matching_i_j.shape}")
    print(f"  Total matches: {matching_i_j.sum()}")
    print(f"  Unique winners: {num_winners}")
    
    return matching_i_j


def build_pop_centroid_features(weight_j: np.ndarray, geo_distance_j_j: np.ndarray, delta: int = 4) -> np.ndarray:
    """Build population-weighted centroid interaction features.
    
    Args:
        weight_j: Population weights for each market
        geo_distance_j_j: Geographic distance matrix between markets
        delta: Distance decay exponent (2 or 4)
    """
    # Compute interaction matrix
    E_j_j = (weight_j[:, None] * weight_j[None, :]).astype(float)
    np.fill_diagonal(E_j_j, 0)
    
    # Apply distance decay
    mask = geo_distance_j_j > 0
    E_j_j[mask] /= (geo_distance_j_j[mask] ** delta)
    
    # Normalize
    pop_centroid_j_j = (weight_j[:, None] / weight_j.sum()) * (E_j_j / E_j_j.sum(1)[:, None])
    
    # Truncate (remove very small values)
    percentile_val = np.percentile(pop_centroid_j_j, POP_CENTROID_PERCENTILE)
    truncated_pop_centroid_j_j = np.where(pop_centroid_j_j > percentile_val, pop_centroid_j_j, 0)
    
    density = np.count_nonzero(truncated_pop_centroid_j_j) / truncated_pop_centroid_j_j.size
    print(f"Built population centroid features:")
    print(f"  Shape: {truncated_pop_centroid_j_j.shape}")
    print(f"  Density: {density:.4f}")
    print(f"  Sum: {truncated_pop_centroid_j_j.sum():.6f}")
    
    return truncated_pop_centroid_j_j


def normalize_interaction_matrix(matrix_j_j: np.ndarray, weight_j: np.ndarray) -> np.ndarray:
    """Normalize interaction matrix by outflow and weight."""
    matrix = matrix_j_j.copy() + 1e-15
    np.fill_diagonal(matrix, 0)
    
    outflow_j = matrix.sum(1)
    mask = outflow_j > 0
    matrix[mask] /= outflow_j[mask][:, None]
    matrix *= weight_j[:, None] / weight_j[mask].sum()
    
    return matrix


def build_quadratic_features(
    weight_j: np.ndarray,
    geo_distance_j_j: np.ndarray,
    travel_survey_j_j: np.ndarray,
    air_travel_j_j: np.ndarray,
    delta: int = 4,
) -> np.ndarray:
    """Build all quadratic item-item features.
    
    Args:
        weight_j: Population weights for each market
        geo_distance_j_j: Geographic distance matrix
        travel_survey_j_j: Travel survey interaction matrix
        air_travel_j_j: Air travel interaction matrix
        delta: Distance decay exponent (2 or 4)
    """
    quadratic_list = []
    
    # Population-weighted centroid
    pop_centroid_j_j = build_pop_centroid_features(weight_j, geo_distance_j_j, delta=delta)
    quadratic_list.append(pop_centroid_j_j)
    
    # Travel survey interactions
    quadratic_travel_j_j = normalize_interaction_matrix(travel_survey_j_j, weight_j)
    quadratic_list.append(quadratic_travel_j_j)
    
    # Air travel interactions
    quadratic_air_j_j = normalize_interaction_matrix(air_travel_j_j, weight_j)
    quadratic_list.append(quadratic_air_j_j)
    
    # Stack into (num_items, num_items, num_quadratic_features)
    quadratic_characteristic_j_j_k = np.stack(quadratic_list, axis=2)
    
    density = (quadratic_characteristic_j_j_k.sum(2) > 0).sum() / quadratic_characteristic_j_j_k.sum(2).size
    print(f"Built quadratic features:")
    print(f"  Shape: {quadratic_characteristic_j_j_k.shape}")
    print(f"  Density: {density:.4f}")
    print(f"  Sum per feature: {quadratic_characteristic_j_j_k.sum((0, 1))}")
    
    return quadratic_characteristic_j_j_k


def build_modular_features(
    capacity_i: np.ndarray,
    weight_j: np.ndarray,
    home_bta_i: np.ndarray = None,
    geo_distance_j_j: np.ndarray = None,
    include_hq_distance: bool = False,
) -> np.ndarray:
    """Build modular agent-item features.
    
    Args:
        capacity_i: Bidder capacities
        weight_j: Item weights (population)
        home_bta_i: Home BTA for each bidder (1-indexed), required if include_hq_distance=True
        geo_distance_j_j: Geographic distance matrix, required if include_hq_distance=True
        include_hq_distance: Whether to include HQ-to-item distance features
    """
    modular_list = []
    
    # Feature 1: eligibility_i * pop_j (normalized)
    modular_feat = (capacity_i[:, None] / weight_j.sum()) * (weight_j[None, :] / weight_j.sum())
    modular_list.append(modular_feat)
    
    # Features 2-3: HQ-to-item distance and distance squared
    if include_hq_distance:
        assert home_bta_i is not None, "home_bta_i required for HQ distance features"
        assert geo_distance_j_j is not None, "geo_distance_j_j required for HQ distance features"
        
        num_agents = len(capacity_i)
        num_items = geo_distance_j_j.shape[0]
        
        # Distance from bidder i's home BTA to item j's BTA
        hq_distance_i_j = np.zeros((num_agents, num_items))
        for i in range(num_agents):
            hq_idx = int(home_bta_i[i]) - 1  # Convert to 0-indexed
            hq_distance_i_j[i, :] = geo_distance_j_j[hq_idx, :]
        
        # Normalize distance (divide by max to keep scale reasonable)
        max_dist = hq_distance_i_j.max()
        if max_dist > 0:
            hq_distance_normalized = hq_distance_i_j / max_dist
        else:
            hq_distance_normalized = hq_distance_i_j
        
        modular_list.append(hq_distance_normalized)
        
        # Distance squared (normalized)
        hq_distance_sq_normalized = hq_distance_normalized ** 2
        modular_list.append(hq_distance_sq_normalized)
        
        print(f"  HQ distance features: mean={hq_distance_normalized.mean():.4f}, max={hq_distance_normalized.max():.4f}")
    
    # Stack into (num_agents, num_items, num_modular_features)
    modular_characteristics_i_j_k = np.stack(modular_list, axis=2)
    
    print(f"Built modular features:")
    print(f"  Shape: {modular_characteristics_i_j_k.shape}")
    
    return modular_characteristics_i_j_k


def save_processed_data(
    matching_i_j: np.ndarray,
    weight_j: np.ndarray,
    capacity_i: np.ndarray,
    modular_characteristics_i_j_k: np.ndarray,
    quadratic_characteristic_j_j_k: np.ndarray,
    delta: int = 4,
    winners_only: bool = False,
    hq_distance: bool = False,
) -> None:
    """Save all processed data to numpy files."""
    output_dir = get_output_dir(delta, winners_only, hq_distance)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(output_dir / "matching_i_j.npy", matching_i_j)
    np.save(output_dir / "weight_j.npy", weight_j)
    np.save(output_dir / "capacity_i.npy", capacity_i)
    np.save(output_dir / "modular_characteristics_i_j_k.npy", modular_characteristics_i_j_k)
    np.save(output_dir / "quadratic_characteristic_j_j_k.npy", quadratic_characteristic_j_j_k)
    
    # Save metadata
    metadata = {
        "delta": delta,
        "winners_only": winners_only,
        "hq_distance": hq_distance,
        "weight_rounding_tick": WEIGHT_ROUNDING_TICK,
        "pop_centroid_percentile": POP_CENTROID_PERCENTILE,
        "num_agents": int(capacity_i.shape[0]),
        "num_items": int(weight_j.shape[0]),
        "num_modular_features": int(modular_characteristics_i_j_k.shape[2]),
        "num_quadratic_features": int(quadratic_characteristic_j_j_k.shape[2]),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved processed data to {output_dir}:")
    print(f"  matching_i_j.npy: {matching_i_j.shape}")
    print(f"  weight_j.npy: {weight_j.shape}")
    print(f"  capacity_i.npy: {capacity_i.shape}")
    print(f"  modular_characteristics_i_j_k.npy: {modular_characteristics_i_j_k.shape}")
    print(f"  quadratic_characteristic_j_j_k.npy: {quadratic_characteristic_j_j_k.shape}")
    print(f"  metadata.json: delta={delta}, winners_only={winners_only}, hq_distance={hq_distance}")


def compute_feature_statistics(
    matching_i_j: np.ndarray,
    modular_characteristics_i_j_k: np.ndarray,
    quadratic_characteristic_j_j_k: np.ndarray,
) -> None:
    """Compute and print statistics on features at observed matching."""
    # Compute features at observed matching
    phi_modular = (modular_characteristics_i_j_k * matching_i_j[:, :, None]).sum(1)
    phi_quadratic = np.einsum('jlk,ij,il->ik', quadratic_characteristic_j_j_k, matching_i_j, matching_i_j)
    phi_hat_i_k = np.concatenate([phi_modular, phi_quadratic], axis=1)
    
    winning = np.unique(np.where(matching_i_j)[0])
    
    print(f"\nFeature statistics at observed matching:")
    print(f"  Total features: {phi_hat_i_k.shape[1]}")
    print(f"  Winning agents: {len(winning)}")
    print(f"  Mean features: {phi_hat_i_k[winning].mean(0)}")
    print(f"  Std features: {phi_hat_i_k[winning].std(0)}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare data for combinatorial auction estimation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python prepare_data.py                       # delta=4, all bidders
    python prepare_data.py --delta 2             # delta=2, all bidders
    python prepare_data.py --delta 4 --winners-only  # delta=4, winners only
    python prepare_data.py --delta 4 --hq-distance   # include HQ distance features
        """
    )
    parser.add_argument(
        "--delta", "-d",
        type=int,
        choices=[2, 4],
        default=4,
        help="Distance decay exponent for population/centroid feature (default: 4)"
    )
    parser.add_argument(
        "--winners-only", "-w",
        action="store_true",
        help="Filter to winning bidders only (those who win at least one item)"
    )
    parser.add_argument(
        "--hq-distance",
        action="store_true",
        help="Include HQ-to-item distance features (adds 2 modular features)"
    )
    return parser.parse_args()


def main(delta: int = 4, winners_only: bool = False, hq_distance: bool = False):
    """Main data preparation pipeline.
    
    Args:
        delta: Distance decay exponent (2 or 4)
        winners_only: If True, filter to winning bidders only
        hq_distance: If True, include HQ-to-item distance features
    """
    print("=" * 70)
    print("Combinatorial Auction Data Preparation")
    print(f"  Distance parameter δ = {delta}")
    print(f"  Winners only: {winners_only}")
    print(f"  HQ distance features: {hq_distance}")
    print("=" * 70)
    
    # Load raw data
    raw_data = load_raw_data()
    
    # Filter bidder data (remove FCC row with bidder_num_fox == 9999, if present)
    raw_data["bidder_data"] = raw_data["bidder_data"][raw_data["bidder_data"]['bidder_num_fox'] != 9999].reset_index(drop=True)
    
    # Create mapping from bidder_num_fox to index (0-based) for matching matrix
    bidder_num_to_index = {
        bidder_num: idx
        for idx, bidder_num in enumerate(raw_data["bidder_data"]['bidder_num_fox'].values)
    }
    
    # Load bidder home BTAs if needed
    home_bta_i = None
    if hq_distance:
        # Load from bidder_data.csv which has home BTA for each bidder
        bidder_hq_data = pd.read_csv(BASE_DIR / "bidder_data.csv")
        # Create mapping from bidder_num_fox to home BTA
        bidder_to_bta = dict(zip(bidder_hq_data['bidder_num_fox'], bidder_hq_data['bta']))
        # Get home BTA for each bidder in order
        home_bta_i = np.array([
            bidder_to_bta.get(bidder_num, 1)  # Default to BTA 1 if not found
            for bidder_num in raw_data["bidder_data"]['bidder_num_fox'].values
        ])
        print(f"Loaded home BTAs for {len(home_bta_i)} bidders")
    
    # Process weights and capacities
    weight_j, capacity_i = process_weights_capacities(
        raw_data["bidder_data"],
        raw_data["bta_data"]
    )
    
    # Generate matching matrix (using filtered bidder indices)
    matching_i_j = generate_matching_matrix(
        raw_data["bta_data"],
        len(capacity_i),
        bidder_num_to_index,
    )
    
    # Build features (before filtering agents)
    modular_characteristics_i_j_k = build_modular_features(
        capacity_i, 
        weight_j,
        home_bta_i=home_bta_i,
        geo_distance_j_j=raw_data["geo_distance_j_j"] if hq_distance else None,
        include_hq_distance=hq_distance,
    )
    quadratic_characteristic_j_j_k = build_quadratic_features(
        weight_j,
        raw_data["geo_distance_j_j"],
        raw_data["travel_survey_j_j"],
        raw_data["air_travel_j_j"],
        delta=delta,
    )
    
    # Filter to winners only if requested
    if winners_only:
        winner_indices = np.where(matching_i_j.sum(axis=1) > 0)[0]
        print(f"\nFiltering to winning bidders only:")
        print(f"  Original agents: {len(capacity_i)}")
        print(f"  Winning agents: {len(winner_indices)}")
        
        # Filter agent-level arrays
        capacity_i = capacity_i[winner_indices]
        matching_i_j = matching_i_j[winner_indices, :]
        modular_characteristics_i_j_k = modular_characteristics_i_j_k[winner_indices, :, :]
    
    # Compute statistics
    compute_feature_statistics(
        matching_i_j,
        modular_characteristics_i_j_k,
        quadratic_characteristic_j_j_k,
    )
    
    # Save processed data
    save_processed_data(
        matching_i_j,
        weight_j,
        capacity_i,
        modular_characteristics_i_j_k,
        quadratic_characteristic_j_j_k,
        delta=delta,
        winners_only=winners_only,
        hq_distance=hq_distance,
    )
    
    print("\n" + "=" * 70)
    print(f"Data preparation complete! (δ = {delta}, winners_only = {winners_only}, hq_distance = {hq_distance})")
    print("=" * 70)


if __name__ == "__main__":
    args = parse_args()
    main(delta=args.delta, winners_only=args.winners_only, hq_distance=args.hq_distance)

