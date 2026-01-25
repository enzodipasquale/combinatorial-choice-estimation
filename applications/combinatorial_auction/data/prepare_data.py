#!/usr/bin/env python3

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from datetime import datetime

# Data paths
BASE_DIR = Path(__file__).parent  # data/
DATA_DIR = BASE_DIR / "114402-V1" / "Replication-Fox-and-Bajari" / "data"

def get_output_dir(delta: int, winners_only: bool = False, hq_distance: bool = False) -> Path:
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

    
    # BTA data (items/markets)
    bta_data = pd.read_csv(DATA_DIR / "btadata_2004_03_12_1.csv")
   
    # Bidder data (agents)
    bidder_data = pd.read_csv(DATA_DIR / "biddercblk_03_28_2004_pln.csv")
   
    # Adjacency matrix
    bta_adjacency = pd.read_csv(DATA_DIR / "btamatrix_merged.csv", header=None)
    bta_adjacency = bta_adjacency.drop(bta_adjacency.columns[0], axis=1).values.astype(float)
    
    # Geographic distance matrix
    geo_distance = pd.read_csv(
        DATA_DIR / "distancesmat_dio_perl_fixed.dat",
        delimiter=' ',
        header=None
    )
    geo_distance = geo_distance.drop(geo_distance.columns[-1], axis=1).values.astype(float)
 
    
    # Travel survey matrix (read with header=None to get full 493x493 matrix)
    travel_survey = pd.read_csv(DATA_DIR / "american-travel-survey-1995-zero.csv", header=None).values.astype(float)
 
    
    # Air travel matrix (read with header=None to get full 493x493 matrix)
    air_travel = pd.read_csv(DATA_DIR / "air-travel-passengers-bta-year-1994.csv", header=None).values.astype(float)
   
    
    return {
        "bta_data": bta_data,
        "bidder_data": bidder_data,
        "bta_adjacency": bta_adjacency,
        "geo_distance": geo_distance,
        "travel_survey": travel_survey,
        "air_travel": air_travel,
    }


def process_weights_capacities(bidder_data: pd.DataFrame, bta_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    capacities = bidder_data['pops_eligible'].fillna(0).to_numpy()
    weights = bta_data['pop90'].to_numpy()
    
    weights_rounded = np.round(weights / WEIGHT_ROUNDING_TICK).astype(int)
    capacities_rounded = np.round(capacities / WEIGHT_ROUNDING_TICK).astype(int)
    
 
    
    return weights_rounded, capacities_rounded


def generate_matching_matrix(
    bta_data: pd.DataFrame,
    n_obs: int,
    bidder_num_to_index: dict,
) -> np.ndarray:
    n_items = len(bta_data)
    matching = np.zeros((n_obs, n_items), dtype=bool)
    
    for j in range(n_items):
        winner_bidder_num = bta_data['bidder_num_fox'].values[j]
        # Skip FCC (9999) or any bidders not in our filtered list
        if winner_bidder_num in bidder_num_to_index:
            winner_id = bidder_num_to_index[winner_bidder_num]
            matching[winner_id, j] = True
    
    num_winners = np.unique(np.where(matching)[0]).size
   
    
    return matching


def build_pop_centroid_features(weights: np.ndarray, geo_distance: np.ndarray, delta: int = 4) -> np.ndarray:
    E_j_j = (weights[:, None] * weights[None, :]).astype(float)
    np.fill_diagonal(E_j_j, 0)
    
    # Apply distance decay
    mask = geo_distance > 0
    E_j_j[mask] /= (geo_distance[mask] ** delta)
    
    # Normalize
    pop_centroid = (weights[:, None] / weights.sum()) * (E_j_j / E_j_j.sum(1)[:, None])
    
    # Truncate (remove very small values)
    percentile_val = np.percentile(pop_centroid, POP_CENTROID_PERCENTILE)
    truncated_pop_centroid = np.where(pop_centroid > percentile_val, pop_centroid, 0)
    
    return truncated_pop_centroid


def normalize_interaction_matrix(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # matrix = matrix.copy() + 1e-15
    np.fill_diagonal(matrix, 0)
    
    outflow = matrix.sum(1)
    mask = outflow > 0
    matrix[mask] /= outflow[mask][:, None]
    matrix *= weights[:, None] / weights[mask].sum()
    
    return matrix


def build_quadratic_features(
    weights: np.ndarray,
    geo_distance: np.ndarray,
    travel_survey: np.ndarray,
    air_travel: np.ndarray,
    delta: int = 4,
) -> np.ndarray:
    quadratic_list = []
    
    pop_centroid = build_pop_centroid_features(weights, geo_distance, delta=delta)
    quadratic_list.append(pop_centroid)
    
    quadratic_travel = normalize_interaction_matrix(travel_survey, weights)
    quadratic_list.append(quadratic_travel)
    
    quadratic_air = normalize_interaction_matrix(air_travel, weights)
    quadratic_list.append(quadratic_air)
    
    quadratic_features = np.stack(quadratic_list, axis=2)

    return quadratic_features


def build_modular_features(
    capacities: np.ndarray,
    weights: np.ndarray,
    home_bta_i: np.ndarray = None,
    geo_distance: np.ndarray = None,
    include_hq_distance: bool = False,
) -> np.ndarray:
    modular_list = []
    
    modular_feat = (capacities[:, None] / weights.sum()) * (weights[None, :] / weights.sum())
    modular_list.append(modular_feat)
    
    if include_hq_distance:
        assert home_bta_i is not None, "home_bta_i required for HQ distance features"
        assert geo_distance is not None, "geo_distance required for HQ distance features"
        
        n_obs = len(capacities)
        n_items = geo_distance.shape[0]
        
        hq_distance = np.zeros((n_obs, n_items))
        for i in range(n_obs):
            hq_idx = int(home_bta_i[i]) - 1
            hq_distance[i, :] = geo_distance[hq_idx, :]
        
        max_dist = hq_distance.max()
        if max_dist > 0:
            hq_distance_normalized = hq_distance / max_dist
        else:
            hq_distance_normalized = hq_distance
        
        modular_list.append(hq_distance_normalized)
        
        hq_distance_sq_normalized = hq_distance_normalized ** 2
        modular_list.append(hq_distance_sq_normalized)
        
     
    modular_features = np.stack(modular_list, axis=2)
 
    
    return modular_features


def build_input_data(
    matching: np.ndarray,
    weights: np.ndarray,
    capacities: np.ndarray,
    modular_features: np.ndarray,
    quadratic_features: np.ndarray,
) -> dict:
    """Build input_data dictionary for BundleChoice without saving to disk."""
    n_items = weights.shape[0]
    
    input_data = {
        "id_data": {
            "modular": modular_features,  
            "capacity": capacities,
            "obs_bundles": matching.astype(int),
        },
        "item_data": {
            "modular": -np.eye(n_items), 
            "quadratic": quadratic_features,  
            "weight": weights,
        }
    }
    
    return input_data



def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare data for combinatorial auction estimation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python prepare_data.py                              # delta=4, all bidders
    python prepare_data.py --delta 2                    # delta=2, all bidders
    python prepare_data.py --config ../point_estimate/config.yaml  # read from config
        """
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file (reads delta, winners_only, hq_distance from application section)"
    )
    parser.add_argument(
        "--delta", "-d",
        type=int,
        choices=[2, 4],
        default=None,
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
    args = parser.parse_args()
    
    # If config provided, read parameters from it
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        app_cfg = config.get("application", {})
        args.delta = app_cfg.get("delta", 4)
        args.winners_only = app_cfg.get("winners_only", False)
        args.hq_distance = app_cfg.get("hq_distance", False)
    
    # Default delta if not set
    if args.delta is None:
        args.delta = 4
    
    return args


def main(delta: int = 4, winners_only: bool = False, hq_distance: bool = False):

    # Load raw data
    raw_data = load_raw_data()
    
    # Filter bidder data:
    # 1. Remove FCC (bidder_num_fox == 9999) if present
    # 2. Remove bidder 256 (has NaN capacity and is not in original Fox-Bajari sample)
    bidder_data = raw_data["bidder_data"]
    n_before = len(bidder_data)
    bidder_data = bidder_data[bidder_data['bidder_num_fox'] != 9999]
    bidder_data = bidder_data[bidder_data['bidder_num_fox'] != 256]
    bidder_data = bidder_data.reset_index(drop=True)
    raw_data["bidder_data"] = bidder_data
    n_after = len(bidder_data)

    
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
     
    # Process weights and capacities
    weights, capacities = process_weights_capacities(
        raw_data["bidder_data"],
        raw_data["bta_data"]
    )
    
    # Generate matching matrix (using filtered bidder indices)
    matching = generate_matching_matrix(
        raw_data["bta_data"],
        len(capacities),
        bidder_num_to_index,
    )
    
    modular_features = build_modular_features(
        capacities, 
        weights,
        home_bta_i=home_bta_i,
        geo_distance=raw_data["geo_distance"] if hq_distance else None,
        include_hq_distance=hq_distance,
    )
    quadratic_features = build_quadratic_features(
        weights,
        raw_data["geo_distance"],
        raw_data["travel_survey"],
        raw_data["air_travel"],
        delta=delta,
    )
    
    if winners_only:
        winner_indices = np.where(matching.sum(axis=1) > 0)[0]

        
        capacities = capacities[winner_indices]
        matching = matching[winner_indices, :]
        modular_features = modular_features[winner_indices, :, :]
    

    input_data = build_input_data(
        matching,
        weights,
        capacities,
        modular_features,
        quadratic_features,
    )
    
    return input_data


if __name__ == "__main__":
    args = parse_args()
    main(delta=args.delta, winners_only=args.winners_only, hq_distance=args.hq_distance)

