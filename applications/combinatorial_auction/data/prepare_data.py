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
    bta_adjacency = bta_adjacency.drop(bta_adjacency.columns[0], axis=1)
    bta_adjacency_j_j = bta_adjacency.values.astype(float)
  
    
    # Geographic distance matrix
    geo_distance = pd.read_csv(
        DATA_DIR / "distancesmat_dio_perl_fixed.dat",
        delimiter=' ',
        header=None
    )
    geo_distance = geo_distance.drop(geo_distance.columns[-1], axis=1)
    geo_distance_j_j = geo_distance.values.astype(float)
 
    
    # Travel survey matrix (read with header=None to get full 493x493 matrix)
    travel_survey = pd.read_csv(DATA_DIR / "american-travel-survey-1995-zero.csv", header=None)
    travel_survey_j_j = travel_survey.values.astype(float)
 
    
    # Air travel matrix (read with header=None to get full 493x493 matrix)
    air_travel = pd.read_csv(DATA_DIR / "air-travel-passengers-bta-year-1994.csv", header=None)
    air_travel_j_j = air_travel.values.astype(float)
   
    
    return {
        "bta_data": bta_data,
        "bidder_data": bidder_data,
        "bta_adjacency_j_j": bta_adjacency_j_j,
        "geo_distance_j_j": geo_distance_j_j,
        "travel_survey_j_j": travel_survey_j_j,
        "air_travel_j_j": air_travel_j_j,
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


def build_pop_centroid_features(weights: np.ndarray, geo_distance_j_j: np.ndarray, delta: int = 4) -> np.ndarray:
    E_j_j = (weights[:, None] * weights[None, :]).astype(float)
    np.fill_diagonal(E_j_j, 0)
    
    # Apply distance decay
    mask = geo_distance_j_j > 0
    E_j_j[mask] /= (geo_distance_j_j[mask] ** delta)
    
    # Normalize
    pop_centroid_j_j = (weights[:, None] / weights.sum()) * (E_j_j / E_j_j.sum(1)[:, None])
    
    # Truncate (remove very small values)
    percentile_val = np.percentile(pop_centroid_j_j, POP_CENTROID_PERCENTILE)
    truncated_pop_centroid_j_j = np.where(pop_centroid_j_j > percentile_val, pop_centroid_j_j, 0)
    
    density = np.count_nonzero(truncated_pop_centroid_j_j) / truncated_pop_centroid_j_j.size
  
    
    return truncated_pop_centroid_j_j


def normalize_interaction_matrix(matrix_j_j: np.ndarray, weights: np.ndarray) -> np.ndarray:
    matrix = matrix_j_j.copy() + 1e-15
    np.fill_diagonal(matrix, 0)
    
    outflow_j = matrix.sum(1)
    mask = outflow_j > 0
    matrix[mask] /= outflow_j[mask][:, None]
    matrix *= weights[:, None] / weights[mask].sum()
    
    return matrix


def build_quadratic_features(
    weights: np.ndarray,
    geo_distance_j_j: np.ndarray,
    travel_survey_j_j: np.ndarray,
    air_travel_j_j: np.ndarray,
    delta: int = 4,
) -> np.ndarray:
    quadratic_list = []
    
    pop_centroid_j_j = build_pop_centroid_features(weights, geo_distance_j_j, delta=delta)
    quadratic_list.append(pop_centroid_j_j)
    
    quadratic_travel_j_j = normalize_interaction_matrix(travel_survey_j_j, weights)
    quadratic_list.append(quadratic_travel_j_j)
    
    quadratic_air_j_j = normalize_interaction_matrix(air_travel_j_j, weights)
    quadratic_list.append(quadratic_air_j_j)
    
    quadratic_features = np.stack(quadratic_list, axis=2)
    
    density = (quadratic_features.sum(2) > 0).sum() / quadratic_features.sum(2).size

    return quadratic_features


def build_modular_features(
    capacities: np.ndarray,
    weights: np.ndarray,
    home_bta_i: np.ndarray = None,
    geo_distance_j_j: np.ndarray = None,
    include_hq_distance: bool = False,
) -> np.ndarray:
    modular_list = []
    
    modular_feat = (capacities[:, None] / weights.sum()) * (weights[None, :] / weights.sum())
    modular_list.append(modular_feat)
    
    if include_hq_distance:
        assert home_bta_i is not None, "home_bta_i required for HQ distance features"
        assert geo_distance_j_j is not None, "geo_distance_j_j required for HQ distance features"
        
        n_obs = len(capacities)
        n_items = geo_distance_j_j.shape[0]
        
        hq_distance_i_j = np.zeros((n_obs, n_items))
        for i in range(n_obs):
            hq_idx = int(home_bta_i[i]) - 1
            hq_distance_i_j[i, :] = geo_distance_j_j[hq_idx, :]
        
        max_dist = hq_distance_i_j.max()
        if max_dist > 0:
            hq_distance_normalized = hq_distance_i_j / max_dist
        else:
            hq_distance_normalized = hq_distance_i_j
        
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
            "modular": modular_features,  # (n_obs, n_items, num_modular)
            "capacity": capacities,
            "obs_bundles": matching.astype(int),
        },
        "item_data": {
            "modular": -np.eye(n_items),  # Item fixed effects (negative identity matrix)
            "quadratic": quadratic_features,  # (n_items, n_items, num_quadratic)
            "weight": weights,
        }
    }
    
    return input_data


def save_processed_data(
    matching: np.ndarray,
    weights: np.ndarray,
    capacities: np.ndarray,
    modular_features: np.ndarray,
    quadratic_features: np.ndarray,
    delta: int = 4,
    winners_only: bool = False,
    hq_distance: bool = False,
) -> None:
    output_dir = get_output_dir(delta, winners_only, hq_distance)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_obs, n_items = matching.shape
    num_modular = modular_features.shape[2]
    num_quadratic = quadratic_features.shape[2]
    
    # Save feature data in new folder structure
    features_dir = output_dir / "features_data"
    (features_dir / "id_data" / "modular").mkdir(parents=True, exist_ok=True)
    (features_dir / "item_data" / "quadratic").mkdir(parents=True, exist_ok=True)
    
    # id modular: (n_obs, n_items) - one CSV per feature
    for k in range(num_modular):
        feature_data = modular_features[:, :, k]  # Shape: (n_obs, n_items)
        pd.DataFrame(feature_data, columns=[f"item_{j}" for j in range(n_items)]).to_csv(
            features_dir / "id_data" / "modular" / f"feature_{k}.csv", index=False
        )
    
    # item quadratic: (n_items, n_items) - one CSV per feature
    for k in range(num_quadratic):
        feature_data = quadratic_features[:, :, k]  # Shape: (n_items, n_items)
        pd.DataFrame(feature_data, columns=[f"item_{j}" for j in range(n_items)]).to_csv(
            features_dir / "item_data" / "quadratic" / f"feature_{k}.csv", index=False
        )
    
    # Save constraint data in other_data
    other_dir = output_dir / "other_data"
    (other_dir / "id_data").mkdir(parents=True, exist_ok=True)
    (other_dir / "item_data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"capacity": capacities}).to_csv(other_dir / "id_data" / "capacity.csv", index=False)
    pd.DataFrame({"weight": weights}).to_csv(other_dir / "item_data" / "weight.csv", index=False)
    
    # Save observed bundles
    obs_df = pd.DataFrame(matching.astype(int), columns=[f"item_{j}" for j in range(n_items)])
    obs_df.to_csv(output_dir / "obs_bundles.csv", index=False)
    
    # Save metadata
    metadata = {
        "delta": delta,
        "winners_only": winners_only,
        "hq_distance": hq_distance,
        "weight_rounding_tick": WEIGHT_ROUNDING_TICK,
        "pop_centroid_percentile": POP_CENTROID_PERCENTILE,
        "n_obs": n_obs,
        "n_items": n_items,
        "num_modular_features": num_modular,
        "num_quadratic_features": num_quadratic,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

def compute_feature_statistics(
    matching: np.ndarray,
    modular_features: np.ndarray,
    quadratic_features: np.ndarray,
) -> None:
    phi_modular = (modular_features * matching[:, :, None]).sum(1)
    phi_quadratic = np.einsum('jlk,ij,il->ik', quadratic_features, matching, matching)
    phi_hat_i_k = np.concatenate([phi_modular, phi_quadratic], axis=1)
    
    winning = np.unique(np.where(matching)[0])


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
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save processed data to disk (default: False)"
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


def main(delta: int = 4, winners_only: bool = False, hq_distance: bool = False, save_data: bool = False):

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
        geo_distance_j_j=raw_data["geo_distance_j_j"] if hq_distance else None,
        include_hq_distance=hq_distance,
    )
    quadratic_features = build_quadratic_features(
        weights,
        raw_data["geo_distance_j_j"],
        raw_data["travel_survey_j_j"],
        raw_data["air_travel_j_j"],
        delta=delta,
    )
    
    if winners_only:
        winner_indices = np.where(matching.sum(axis=1) > 0)[0]

        
        capacities = capacities[winner_indices]
        matching = matching[winner_indices, :]
        modular_features = modular_features[winner_indices, :, :]
    
    compute_feature_statistics(
        matching,
        modular_features,
        quadratic_features,
    )
    
    input_data = build_input_data(
        matching,
        weights,
        capacities,
        modular_features,
        quadratic_features,
    )
    
    if save_data:
        save_processed_data(
            matching,
            weights,
            capacities,
            modular_features,
            quadratic_features,
            delta=delta,
            winners_only=winners_only,
            hq_distance=hq_distance,
        )

    return input_data


if __name__ == "__main__":
    args = parse_args()
    main(delta=args.delta, winners_only=args.winners_only, hq_distance=args.hq_distance, save_data=args.save)

