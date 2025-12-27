#!/usr/bin/env python3
"""
Data preparation script for combinatorial auction estimation.

Processes raw data from the Fox-Bajari replication package into the format
required by BundleChoice framework.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple

# Data paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "114402-V1" / "Replication-Fox-and-Bajari" / "data"
OUTPUT_DIR = BASE_DIR / "input_data"

# Processing parameters
WEIGHT_ROUNDING_TICK = 1000
POP_CENTROID_DELTA = 4
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


def build_pop_centroid_features(weight_j: np.ndarray, geo_distance_j_j: np.ndarray) -> np.ndarray:
    """Build population-weighted centroid interaction features."""
    # Compute interaction matrix
    E_j_j = (weight_j[:, None] * weight_j[None, :]).astype(float)
    np.fill_diagonal(E_j_j, 0)
    
    # Apply distance decay
    mask = geo_distance_j_j > 0
    E_j_j[mask] /= (geo_distance_j_j[mask] ** POP_CENTROID_DELTA)
    
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
) -> np.ndarray:
    """Build all quadratic item-item features."""
    quadratic_list = []
    
    # Population-weighted centroid
    pop_centroid_j_j = build_pop_centroid_features(weight_j, geo_distance_j_j)
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
) -> np.ndarray:
    """Build modular agent-item features."""
    # eligibility_i * pop_j (normalized)
    modular_feat = (capacity_i[:, None] / weight_j.sum()) * (weight_j[None, :] / weight_j.sum())
    
    # Stack into (num_agents, num_items, num_modular_features)
    modular_characteristics_i_j_k = modular_feat[:, :, None]
    
    print(f"Built modular features:")
    print(f"  Shape: {modular_characteristics_i_j_k.shape}")
    
    return modular_characteristics_i_j_k


def save_processed_data(
    matching_i_j: np.ndarray,
    weight_j: np.ndarray,
    capacity_i: np.ndarray,
    modular_characteristics_i_j_k: np.ndarray,
    quadratic_characteristic_j_j_k: np.ndarray,
) -> None:
    """Save all processed data to numpy files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    np.save(OUTPUT_DIR / "matching_i_j.npy", matching_i_j)
    np.save(OUTPUT_DIR / "weight_j.npy", weight_j)
    np.save(OUTPUT_DIR / "capacity_i.npy", capacity_i)
    np.save(OUTPUT_DIR / "modular_characteristics_i_j_k.npy", modular_characteristics_i_j_k)
    np.save(OUTPUT_DIR / "quadratic_characteristic_j_j_k.npy", quadratic_characteristic_j_j_k)
    
    print(f"\nSaved processed data to {OUTPUT_DIR}:")
    print(f"  matching_i_j.npy: {matching_i_j.shape}")
    print(f"  weight_j.npy: {weight_j.shape}")
    print(f"  capacity_i.npy: {capacity_i.shape}")
    print(f"  modular_characteristics_i_j_k.npy: {modular_characteristics_i_j_k.shape}")
    print(f"  quadratic_characteristic_j_j_k.npy: {quadratic_characteristic_j_j_k.shape}")


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


def main():
    """Main data preparation pipeline."""
    print("=" * 70)
    print("Combinatorial Auction Data Preparation")
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
    
    # Build features
    modular_characteristics_i_j_k = build_modular_features(capacity_i, weight_j)
    quadratic_characteristic_j_j_k = build_quadratic_features(
        weight_j,
        raw_data["geo_distance_j_j"],
        raw_data["travel_survey_j_j"],
        raw_data["air_travel_j_j"],
    )
    
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
    )
    
    print("\n" + "=" * 70)
    print("Data preparation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

