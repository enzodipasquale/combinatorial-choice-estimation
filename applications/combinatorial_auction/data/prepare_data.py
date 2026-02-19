#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Data paths
BASE_DIR = Path(__file__).parent  # data/
DATA_DIR = BASE_DIR / "114402-V1" / "Replication-Fox-and-Bajari" / "data"

WEIGHT_ROUNDING_TICK = 1000
POP_CENTROID_PERCENTILE = 0
LB_QUADRATICS = 1e-10

NON_CONTINENTAL_MARKETS = (
    "Anchorag", "Fairbank", "Juneau,",           # Alaska
    "Hilo, HI", "Honolulu", "Kahului,", "Lihue, H",  # Hawaii
    "San Juan", "Mayaguez",                       # Puerto Rico
    "Guam", "US Virgi", "American", "Northern",  # territories
)

def build_hq_distance(bidder_data, bta_data, geo_distance):
    n_bid = len(bidder_data)
    n_bta = len(bta_data)
    hq_distance = np.zeros((n_bid, n_bta ))
    for i in range(n_bid):
        bta_i = bidder_data.loc[i,'bta']
        distance = geo_distance[bta_i]
        hq_distance[i] = distance / 1000000

    return hq_distance



def load_raw_data(continental_only):
    
    # BTA data (items/markets)
    bta_data = pd.read_csv(DATA_DIR / "btadata_2004_03_12_1.csv")
   
    # Bidder data (agents)
    bidder_data = pd.read_csv(DATA_DIR / "biddercblk_03_28_2004_pln.csv")
    form175 = pd.read_csv(DATA_DIR / "fccform175nomiss.csv", header=None)

    bidder_data.loc[bidder_data["bidder_num_fox"] == 67, "pops_eligible"] = form175.loc[form175[0] == 67, 3].item()
    bidder_data = bidder_data[bidder_data["bidder_num"] != 9999]

    # Merge assets, revenues and eligibility from form 175
    form175 = form175.rename(columns={0: "bidder_num_fox", 1: "assets", 2: "revenues", 3: "eligibility_form175" })
    bidder_data = bidder_data.merge(form175, on="bidder_num_fox", )

    # Merge additional data
    bidder_data_plus = pd.read_csv(BASE_DIR / "bidder_data.csv")
    cols = ['bidder_num_fox', 'bta', 'Applicant_Status', 'Legal_Classification']
    bidder_data = bidder_data.merge(bidder_data_plus[cols], on='bidder_num_fox')

    # DCR/DCC: swap bidder_num_fox so winner 190 gets 89M, winner 234 gets 11M
    mask_190 = bidder_data["bidder_num_fox"] == 190
    mask_234 = bidder_data["bidder_num_fox"] == 234
    print("Swapping bidder_num_fox 190 ↔ 234:")
    bidder_data.loc[mask_190, "bidder_num_fox"] = 234
    bidder_data.loc[mask_234, "bidder_num_fox"] = 190
    print(bidder_data.loc[mask_190].to_string())
    print(bidder_data.loc[mask_234].to_string())

    # Quadratic BTA data
    bta_adjacency = pd.read_csv(DATA_DIR / "btamatrix_merged.csv", header=None)
    bta_adjacency = bta_adjacency.drop(bta_adjacency.columns[0], axis=1).values.astype(float)
    geo_distance = pd.read_csv(DATA_DIR / "distancesmat_dio_perl_fixed.dat", delimiter=' ',header=None)
    geo_distance = geo_distance.drop(geo_distance.columns[-1], axis=1).values.astype(float)
    travel_survey = pd.read_csv(DATA_DIR / "american-travel-survey-1995-zero.csv", header=None).values.astype(float)
    air_travel = pd.read_csv(DATA_DIR / "air-travel-passengers-bta-year-1994.csv", header=None).values.astype(float)
    
    hq_distance = build_hq_distance(bidder_data, bta_data, geo_distance)

    if continental_only:
        continental_ids = np.where(~bta_data['market'].isin(NON_CONTINENTAL_MARKETS))[0]
        non_continental_ids = np.where(bta_data['market'].isin(NON_CONTINENTAL_MARKETS))[0]
        matching_full = generate_matching_matrix(bidder_data, bta_data)
        bidders_to_keep = np.where((matching_full.sum(1) - matching_full[:, non_continental_ids].sum(1) > 0)
                                    | (matching_full.sum(1) == 0))[0]
        bidder_data = bidder_data.iloc[bidders_to_keep].reset_index(drop=True)
        hq_distance = hq_distance[bidders_to_keep]

        bta_data = bta_data.iloc[continental_ids].reset_index(drop=True)
        bta_adjacency = bta_adjacency[continental_ids][:, continental_ids]
        geo_distance = geo_distance[continental_ids][:, continental_ids]
        travel_survey = travel_survey[continental_ids][:, continental_ids]
        air_travel = air_travel[continental_ids][:, continental_ids]
        hq_distance = hq_distance[:, continental_ids]

    
    return {
        "bta_data": bta_data,
        "bidder_data": bidder_data,
        "bta_adjacency": bta_adjacency,
        "geo_distance": geo_distance,
        "travel_survey": travel_survey,
        "air_travel": air_travel,
        "hq_distance": hq_distance
    }

def process_pd_to_np(bidder_data, bta_data):
    elig = bidder_data['pops_eligible'].to_numpy()
    pop = bta_data['pop90'].to_numpy()
    
    weight = np.round(pop // WEIGHT_ROUNDING_TICK).astype(int)
    capacity = np.round(elig // WEIGHT_ROUNDING_TICK).astype(int)
    
    pop_sum = pop.sum()
    pop = pop/pop_sum
    elig = elig/pop_sum
    assets = bidder_data['assets'].to_numpy() / bidder_data['assets'].max()
    revenues = bidder_data['revenues'].to_numpy() / bidder_data['revenues'].max()

    is_rural = bidder_data['Applicant_Status'].str.contains("Rural Telephone Company").values

    return weight, capacity, pop, elig, assets, revenues, is_rural

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
    matrix = matrix.copy() 
    np.fill_diagonal(matrix, 0)
    outflow = matrix.sum(1)
    mask = outflow > 0
    matrix[mask] /= outflow[mask][:, None]
    matrix *= pop[:, None] 
    np.fill_diagonal(matrix, 0)
    return matrix

def build_quadratic_features(pop, geo_distance, travel_survey, air_travel, delta=4):
    quadratic_list = []
    
    pop_centroid = build_pop_centroid_features(pop, geo_distance, delta=delta)
    pop_centroid = normalize_interaction_matrix(pop_centroid, pop)
    percentile_val = np.percentile(pop_centroid, POP_CENTROID_PERCENTILE)
    truncated_pop_centroid = np.where(pop_centroid > percentile_val, pop_centroid, 0)
    quadratic_list.append(truncated_pop_centroid)
    travel_survey = np.where(travel_survey == 0, LB_QUADRATICS, travel_survey)
    quadratic_travel = normalize_interaction_matrix(travel_survey, pop)
    quadratic_list.append(quadratic_travel )
    air_travel = np.where(air_travel == 0, LB_QUADRATICS, air_travel)
    quadratic_air = normalize_interaction_matrix(air_travel, pop)
    quadratic_list.append(quadratic_air)
    
    quadratic_features = np.stack(quadratic_list, axis=2)
    
    return quadratic_features


def build_modular_features(elig, pop, assets=None, revenues=None, is_rural = None, hq_distance = None):
    modular_list = []
    elig_times_pop = elig[:, None] * pop[None, :]
    modular_list.append(elig_times_pop)
    if assets is not None:
        modular_list.append(assets[:, None] * pop[None, :])
    if revenues is not None:
        modular_list.append(revenues[:, None] * pop[None, :])
    if is_rural is not None:
        modular_list.append(elig_times_pop * is_rural[:,None])
    if hq_distance is not None:
        modular_list.append(hq_distance)
        modular_list.append(hq_distance ** 2)

    
    return np.stack(modular_list, axis=2)



def main(delta=4, winners_only=False, form175_features=False, continental_only=False, hq_distance=False):

    raw_data = load_raw_data(continental_only)

    weight, capacity, pop, elig, assets, revenues, is_rural = process_pd_to_np(
        raw_data["bidder_data"],
        raw_data["bta_data"]
    )
    
    matching = generate_matching_matrix(
        raw_data["bidder_data"],
        raw_data["bta_data"],
    )
    
    modular_features = build_modular_features(
        elig, pop,
        assets=assets if form175_features else None,
        revenues=revenues if form175_features else None,
        is_rural = is_rural if form175_features else None,
        hq_distance = raw_data["hq_distance"] if hq_distance else None
    )
    quadratic_features = build_quadratic_features(
        pop,
        raw_data["geo_distance"],
        raw_data["travel_survey"],
        raw_data["air_travel"],
        delta=delta,
    )
    


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
   
    print_descriptive_stats(raw_data, modular_features, matching, quadratic_features, capacity, elig, weight, pop)
    return input_data


def print_descriptive_stats(raw_data, modular_features, matching, quadratic_features, capacity, elig, weight, pop):
    print("Sizes")
    print(f"  {'modular_features':<20} {modular_features.shape}")
    print(f"  {'capacity':<20} {capacity.shape}")
    print(f"  {'matching':<20} {matching.shape}")
    print(f"  {'quadratic_features':<20} {quadratic_features.shape}")
    print(f"  {'weight':<20} {weight.shape}")
    print(f"  {'elig':<20} {elig.shape}")
    print(f"  {'pop':<20} {pop.shape}")
    print(f"  {'matching.sum()':<20} {matching.sum()}")

    winners = np.where(matching.sum(1)>0)[0]
    m = matching[winners]

    mean_pop = ((m @ pop )/ pop.sum()).mean()
    std_pop = ((m @ pop )/ pop.sum()).std()
    modular_stats = (modular_features[winners] * m[:,:,None]).sum(1)

    print("Modular")
    print(f"  {'pop mean':<20} {mean_pop}")
    print(f"  {'pop std':<20} {std_pop}")
    print(f"  {'mod mean':<20} {modular_stats.mean(0)}")
    print(f"  {'mod std':<20} {modular_stats.std(0)}")

    quad_stats = np.einsum('ij,jlk,il->ik',m , quadratic_features , m)
    print("Quadratic")
    print(f"  {'mean':<20} {quad_stats.mean(0)}")
    print(f"  {'std':<20} {quad_stats.std(0)}")
    print(f"  {'min':<20} {quad_stats.min(0)}")
    print(f"  {'max':<20} {quad_stats.max(0)}")
    print(f"  {'sum':<20} {quadratic_features.sum((0,1))}")
    print(f"  {'density':<20} {(quadratic_features >0).sum((0,1))/ (quadratic_features.shape[0] **2)}")

    print("Other")
    violations = matching @ weight - capacity
    viold_id = np.where(violations>0)
    print(f"  {'viol_ids':<20} {viold_id}")
    print(f"  {'violations':<20} {violations[viold_id]}")
    


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
        "--form175-features", "-f",
        action="store_true",
    )
    parser.add_argument(
        "--hq-distance", "-hq",
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
    main(delta=args.delta, winners_only=args.winners_only, form175_features=args.form175_features, continental_only=args.continental_only, hq_distance=args.hq_distance)

