#!/usr/bin/env python3

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Data paths
BASE_DIR = Path(__file__).parent  # data/
DATA_DIR = BASE_DIR / "114402-V1" / "Replication-Fox-and-Bajari" / "data"

WEIGHT_ROUNDING_TICK = 1000
POP_CENTROID_PERCENTILE = 90

NON_CONTINENTAL_MARKETS = (
    "Anchorag", "Fairbank", "Juneau,",           # Alaska
    "Hilo, HI", "Honolulu", "Kahului,", "Lihue, H",  # Hawaii
    "San Juan", "Mayaguez",                       # Puerto Rico
    "Guam", "US Virgi", "American", "Northern",  # territories
)

def load_raw_data():
    
    # BTA data (items/markets)
    bta_data = pd.read_csv(DATA_DIR / "btadata_2004_03_12_1.csv")
   
    # Bidder data (agents)
    bidder_data = pd.read_csv(DATA_DIR / "biddercblk_03_28_2004_pln.csv")
    form175 = pd.read_csv(DATA_DIR / "fccform175nomiss.csv", header=None)
    bidder_data.loc[bidder_data["bidder_num_fox"] == 67, "pops_eligible"] = form175.loc[form175[0] == 67, 3].item()
    bidder_data = bidder_data[bidder_data["bidder_num"] != 9999]
   
    # Quadratic BTA data
    bta_adjacency = pd.read_csv(DATA_DIR / "btamatrix_merged.csv", header=None)
    bta_adjacency = bta_adjacency.drop(bta_adjacency.columns[0], axis=1).values.astype(float)
    
    geo_distance = pd.read_csv(DATA_DIR / "distancesmat_dio_perl_fixed.dat", delimiter=' ',header=None)
    geo_distance = geo_distance.drop(geo_distance.columns[-1], axis=1).values.astype(float)
    travel_survey = pd.read_csv(DATA_DIR / "american-travel-survey-1995-zero.csv", header=None).values.astype(float)
    air_travel = pd.read_csv(DATA_DIR / "air-travel-passengers-bta-year-1994.csv", header=None).values.astype(float)
    
    return {
        "bta_data": bta_data,
        "bidder_data": bidder_data,
        "bta_adjacency": bta_adjacency,
        "geo_distance": geo_distance,
        "travel_survey": travel_survey,
        "air_travel": air_travel,
    }

def process_weight_capacity(bidder_data, bta_data, continental_ids):
    elig = bidder_data['pops_eligible'].to_numpy()
    pop = bta_data['pop90'].to_numpy()
    
    weight = np.round(pop / WEIGHT_ROUNDING_TICK).astype(int)
    capacity = np.round(elig / WEIGHT_ROUNDING_TICK).astype(int)
    
    if continental_ids is None:
        pop_sum = pop.sum()
    else:
        pop_sum = pop[continental_ids].sum()
    pop = pop/pop_sum
    elig = elig/pop_sum

    return weight, capacity, pop, elig 

def generate_matching_matrix(bidder_data, bta_data):
    n_items = len(bta_data)
    n_obs = len(bidder_data)
    matching = np.zeros((n_obs, n_items), dtype = bool)
    for j in range(n_items):
        winner_bidder_num_fox = bta_data['bidder_num_fox'].iloc[j]
        winned_id = np.where(bidder_data['bidder_num_fox'] == winner_bidder_num_fox)[0]
        matching[winned_id, j] = True
    return matching

def build_pop_centroid_features(pop, geo_distance, delta=4):
    pop_centroid = (pop[:, None] * pop[None, :]).astype(float)
    np.fill_diagonal(pop_centroid, 0)
    mask = geo_distance > 0
    pop_centroid[mask] /= (geo_distance[mask] ** delta)
    return pop_centroid


def normalize_interaction_matrix(matrix, pop):
    # matrix = matrix.copy() + 1e-15
    np.fill_diagonal(matrix, 0)
    outflow = matrix.sum(1)
    mask = outflow > 0
    matrix[mask] /= outflow[mask][:, None]
    matrix *= pop[:, None] 
    return matrix

def build_quadratic_features(pop, geo_distance, travel_survey, air_travel, delta=4):
    quadratic_list = []
    
    pop_centroid = build_pop_centroid_features(pop, geo_distance, delta=delta)
    pop_centroid = normalize_interaction_matrix(pop_centroid, pop)
    percentile_val = np.percentile(pop_centroid, POP_CENTROID_PERCENTILE)
    truncated_pop_centroid = np.where(pop_centroid > percentile_val, pop_centroid, 0)
    quadratic_list.append(truncated_pop_centroid)
    
    quadratic_travel = normalize_interaction_matrix(travel_survey, pop)
    quadratic_list.append(quadratic_travel)
    
    quadratic_air = normalize_interaction_matrix(air_travel, pop)
    quadratic_list.append(quadratic_air)
    
    quadratic_features = np.stack(quadratic_list, axis=2)

    return quadratic_features


def build_modular_features(elig, pop, home_bta_i=None, geo_distance=None, include_hq_distance=False):
    modular_list = []
    
    elig_times_pop = elig[:, None] * pop[None, :] 
    modular_list.append(elig_times_pop)
    modular_features = np.stack(modular_list, axis=2)
    
    return modular_features


def main(delta=4, winners_only=False, hq_distance=False, continental_only = False):

    # Load raw data
    raw_data = load_raw_data()
    home_bta_i = None
    if continental_only:
        continental_ids = np.where(~raw_data["bta_data"]['market'].isin(NON_CONTINENTAL_MARKETS))[0]
    else: 
        continental_ids = None

    weight, capacity, pop, elig,  = process_weight_capacity(
        raw_data["bidder_data"],
        raw_data["bta_data"],
        continental_ids
    )
    
    matching = generate_matching_matrix(
        raw_data["bidder_data"],
        raw_data["bta_data"],
    )
    
    modular_features = build_modular_features(
        elig, 
        pop,
        home_bta_i=home_bta_i,
        geo_distance=raw_data["geo_distance"],
        include_hq_distance=hq_distance,
    )
    quadratic_features = build_quadratic_features(
        pop,
        raw_data["geo_distance"],
        raw_data["travel_survey"],
        raw_data["air_travel"],
        delta=delta,
    )
    


    if continental_only:
        continental_ids = np.where(~raw_data["bta_data"]['market'].isin(NON_CONTINENTAL_MARKETS))[0]
        non_continental_ids = np.where(raw_data["bta_data"]['market'].isin(NON_CONTINENTAL_MARKETS))[0]

        # Filter bidders
        matching_non_continental = matching[:,non_continental_ids]
        bidders_to_keep = np.where((matching.sum(1) -matching_non_continental.sum(1) > 0) | (matching.sum(1) == 0))[0]

        #Filter BTAs
        pop = pop[continental_ids]
        matching = matching[:,continental_ids]
        modular_features = modular_features[:,continental_ids,:]
        quadratic_features = quadratic_features[continental_ids][:,continental_ids]
        weight = weight[continental_ids]

        # Filter Bidders
        capacity = capacity[bidders_to_keep]
        elig = elig[bidders_to_keep]
        matching = matching[bidders_to_keep, :]
        modular_features = modular_features[bidders_to_keep, :, :]

    if winners_only:
        winner_indices = np.where(matching.sum(axis=1) > 0)[0]
        capacity = capacity[winner_indices]
        elig = elig[winner_indices]
        matching = matching[winner_indices, :]
        modular_features = modular_features[winner_indices, :, :]

    n_items = weight.shape[0]
    input_data = {
        "id_data": {
            "modular": modular_features,  
            "capacity": capacity,
            "obs_bundles": matching,
        },
        "item_data": {
            "modular": -np.eye(n_items), 
            "quadratic": quadratic_features,  
            "weight": weight,
        }
    }
   
    print_descriptive_stats(modular_features, matching, quadratic_features, capacity, elig, weight, pop)
    return input_data


def print_descriptive_stats(modular_features, matching, quadratic_features, capacity, elig, weight, pop):
    print(" Sizes")
    print(modular_features.shape)
    print(capacity.shape)
    print(matching.shape)
    print(quadratic_features.shape)
    print(weight.shape)
    print(elig.shape)
    print(pop.shape)
    print(matching.sum())
    

    winners = np.where(matching.sum(1)>0)[0]
    m = matching[winners]

    mean_pop = ((m @ pop )/ pop.sum()).mean()
    std_pop = ((m @ pop )/ pop.sum()).std()
    
    modular_stats = (modular_features[winners] * m[:,:,None]).sum(1)

    print(" ")
    print(" Modular")
    print(mean_pop, std_pop)
    print(modular_stats.mean(), modular_stats.std())



    quad_stats = np.einsum('ij,jlk,il->ik',m , quadratic_features , m)
    print(" ")
    print(" Quadratic")
    print(quad_stats.mean(0))
    print(quad_stats.std(0))
    print(quad_stats.min(0))
    print(quad_stats.max(0))
    


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare data for combinatorial auction estimation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--delta", "-d",
        type=int,
        choices=[2, 4],
        default=None,
    )
    parser.add_argument(
        "--winners-only", "-w",
        action="store_true",
    )
    parser.add_argument(
        "--hq-distance",
        action="store_true",
    )

    parser.add_argument(
        "--continental-only", "-c",
        action="store_true",
    )

    args = parser.parse_args()
    
    if args.delta is None:
        args.delta = 4
    
    return args


if __name__ == "__main__":
    args = parse_args()
    main(delta=args.delta, winners_only=args.winners_only, hq_distance=args.hq_distance, continental_only =args.continental_only )

