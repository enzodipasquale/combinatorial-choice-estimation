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

# ── Regressor registries ─────────────────────────────────────────────

MODULAR = {}
QUADRATIC = {}
QUADRATIC_ID = {}

def modular(name):
    def dec(fn):
        MODULAR[name] = fn
        return fn
    return dec

def quadratic(name):
    def dec(fn):
        QUADRATIC[name] = fn
        return fn
    return dec

def quadratic_id(name):
    def dec(fn):
        QUADRATIC_ID[name] = fn
        return fn
    return dec

# ── Modular regressors (each takes ctx, returns n_bidders × n_items) ─

@modular("elig_pop")
def _elig_pop(ctx):
    return ctx["elig"][:, None] * ctx["pop"][None, :]

@modular("assets_pop")
def _assets_pop(ctx):
    return ctx["assets"][:, None] * ctx["pop"][None, :]

@modular("revenues_pop")
def _revenues_pop(ctx):
    return ctx["revenues"][:, None] * ctx["pop"][None, :]

@modular("rural_elig_pop")
def _rural_elig_pop(ctx):
    return ctx["elig"][:, None] * ctx["pop"][None, :] * ctx["is_rural"][:, None]

@modular("hq_distance")
def _hq_distance(ctx):
    return ctx["hq_distance"]

@modular("hq_distance_sq")
def _hq_distance_sq(ctx):
    return ctx["hq_distance"] ** 2

@modular("log_hq_distance")
def _log_hq_distance(ctx):
    return np.log(ctx["hq_distance"] * 1000 + 1)

@modular("elig_hq_distance")
def _elig_hq_distance(ctx):
    return ctx["elig"][:, None] * ctx["hq_distance"]

@modular("elig_hq_distance_sq")
def _elig_hq_distance_sq(ctx):
    return ctx["elig"][:, None] * (ctx["hq_distance"] **2)

@modular("elig_hq_distance_cb")
def _elig_hq_distance_cb(ctx):
    return ctx["elig"][:, None] * (ctx["hq_distance"] **3)

@modular("elig_log_hq_distance")
def _elig_log_hq_distance(ctx):
    return ctx["elig"][:, None] * np.log(ctx["hq_distance"] * 1000 + 1)

@modular("elig_percapin")
def _elig_percapin(ctx):
    return ctx["elig"][:, None] * ctx["percapin"][None, :]

@modular("elig_hhinc35k")
def _elig_hhinc35k(ctx):
    return ctx["elig"][:, None] * ctx["hhinc35k"][None, :]

@modular("elig_density")
def _elig_density(ctx):
    return ctx["elig"][:, None] * ctx["density"][None, :]

@modular("elig_imwl")
def _elig_imwl(ctx):
    return ctx["elig"][:, None] * ctx["imwl"][None, :]

@modular("elig_price")
def _elig_price(ctx):
    return ctx["elig"][:, None] * ctx["price"][None, :]

# ── Quadratic regressors (each takes ctx, returns n_items × n_items) ─

@quadratic("adjacency")
def _adjacency(ctx):
    return normalize_interaction_matrix(ctx["bta_adjacency"], ctx["pop"])

def _pop_centroid(ctx, delta):
    pc = build_pop_centroid_features(ctx["pop"], ctx["geo_distance"], delta=delta)
    pc = normalize_interaction_matrix(pc, ctx["pop"])
    percentile_val = np.percentile(pc, POP_CENTROID_PERCENTILE)
    return np.where(pc > percentile_val, pc, 0)

@quadratic("pop_centroid_delta2")
def _pop_centroid_delta2(ctx):
    return _pop_centroid(ctx, delta=2)

@quadratic("pop_centroid_delta4")
def _pop_centroid_delta4(ctx):
    return _pop_centroid(ctx, delta=4)

@quadratic("travel_survey")
def _travel_survey(ctx):
    ts = np.where(ctx["travel_survey"] == 0, LB_QUADRATICS, ctx["travel_survey"])
    return normalize_interaction_matrix(ts, ctx["pop"])

@quadratic("air_travel")
def _air_travel(ctx):
    at = np.where(ctx["air_travel"] == 0, LB_QUADRATICS, ctx["air_travel"])
    return normalize_interaction_matrix(at, ctx["pop"])

# ── Quadratic-ID regressors (each takes ctx, returns n_obs × n_items × n_items) ─

@quadratic_id("elig_adjacency")
def _elig_adjacency(ctx):
    adj = QUADRATIC["adjacency"](ctx)
    return ctx["elig"][:, None, None] * adj[None, :, :]

@quadratic_id("rural_adjacency")
def _rural_adjacency(ctx):
    adj = QUADRATIC["adjacency"](ctx)
    return ctx["is_rural"][:, None, None] * adj[None, :, :]

@quadratic_id("elig_pop_centroid_delta2")
def _elig_pop_centroid_delta2(ctx):
    pc = _pop_centroid(ctx, delta=2)
    return ctx["elig"][:, None, None] * pc[None, :, :]

@quadratic_id("elig_pop_centroid_delta4")
def _elig_pop_centroid_delta4(ctx):
    pc = _pop_centroid(ctx, delta=4)
    return ctx["elig"][:, None, None] * pc[None, :, :]

@quadratic_id("rural_pop_centroid_delta4")
def _rural_pop_centroid_delta4(ctx):
    pc = _pop_centroid(ctx, delta=4)
    return ctx["is_rural"][:, None, None] * pc[None, :, :]

@quadratic_id("assets_adjacency")
def _assets_adjacency(ctx):
    adj = QUADRATIC["adjacency"](ctx)
    return ctx["assets"][:, None, None] * adj[None, :, :]

@quadratic_id("assets_pop_centroid_delta2")
def _assets_pop_centroid_delta2(ctx):
    pc = _pop_centroid(ctx, delta=2)
    return ctx["assets"][:, None, None] * pc[None, :, :]

@quadratic_id("assets_pop_centroid_delta4")
def _assets_pop_centroid_delta4(ctx):
    pc = _pop_centroid(ctx, delta=4)
    return ctx["assets"][:, None, None] * pc[None, :, :]

# ── Data loading  ─────────────────────────────────────────

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

    is_rural = bidder_data['Applicant_Status'].str.contains("Rural Telephone Company").fillna(False).values

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

# ── Helpers (registry functions) ─────────────────────────────

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

# ── Phase 2: context builder ────────────────────────────────────────

def build_context(raw_data):
    weight, capacity, pop, elig, assets, revenues, is_rural = process_pd_to_np(
        raw_data["bidder_data"], raw_data["bta_data"]
    )
    matching = generate_matching_matrix(raw_data["bidder_data"], raw_data["bta_data"])
    bta = raw_data["bta_data"]
    percapin = bta["percapin"].to_numpy().astype(float) / bta["percapin"].max()
    hhinc35k = bta["hhinc35k"].to_numpy().astype(float)
    density = bta["density"].to_numpy().astype(float) / bta["density"].max()
    imwl = bta["imwl"].to_numpy().astype(float) / bta["imwl"].max()
    price = bta["bid"].to_numpy().astype(float) / 1e9
    return {
        "weight": weight,
        "capacity": capacity,
        "pop": pop,
        "elig": elig,
        "assets": assets,
        "revenues": revenues,
        "is_rural": is_rural,
        "matching": matching,
        "hq_distance": raw_data["hq_distance"],
        "geo_distance": raw_data["geo_distance"],
        "bta_adjacency": raw_data["bta_adjacency"],
        "travel_survey": raw_data["travel_survey"],
        "air_travel": raw_data["air_travel"],
        "percapin": percapin,
        "hhinc35k": hhinc35k,
        "density": density,
        "imwl": imwl,
        "price": price,
    }

# ── Phase 3: feature builder ────────────────────────────────────────

def build_features(registry, names, ctx):
    layers = [registry[name](ctx) for name in names]
    return np.stack(layers, axis=-1).astype(np.float64)

# ── Main ─────────────────────────────────────────────────────────────

def main(winners_only=False, continental_only=False,
         modular_regressors=None, quadratic_regressors=None, quadratic_id_regressors=None):

    # Phase 1: load & filter
    raw_data = load_raw_data(continental_only)

    # Phase 2: build context
    ctx = build_context(raw_data)

    # Phase 3: build selected features
    if modular_regressors is None:
        modular_regressors = list(MODULAR.keys())
    if quadratic_regressors is None:
        quadratic_regressors = list(QUADRATIC.keys())
    if quadratic_id_regressors is None:
        quadratic_id_regressors = []

    modular_features = build_features(MODULAR, modular_regressors, ctx)
    quadratic_features = build_features(QUADRATIC, quadratic_regressors, ctx)
    quadratic_id_features = build_features(QUADRATIC_ID, quadratic_id_regressors, ctx) if quadratic_id_regressors else None

    matching = ctx["matching"]
    capacity = ctx["capacity"]
    weight = ctx["weight"]
    elig = ctx["elig"]
    pop = ctx["pop"]

    if winners_only:
        winner_indices = np.where(matching.sum(axis=1) > 0)[0]
        capacity = capacity[winner_indices]
        elig = elig[winner_indices]
        matching = matching[winner_indices, :]
        modular_features = modular_features[winner_indices, :, :]
        if quadratic_id_features is not None:
            quadratic_id_features = quadratic_id_features[winner_indices, :, :, :]

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
    if quadratic_id_features is not None:
        input_data["id_data"]["quadratic"] = quadratic_id_features

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
    parser.add_argument("--winners-only", "-w", action="store_true")
    parser.add_argument("--continental-only", "-c", action="store_true")
    parser.add_argument(
        "--modular", nargs="*", default=None,
        help=f"Modular regressors (available: {', '.join(MODULAR)})",
    )
    parser.add_argument(
        "--quadratic", nargs="*", default=None,
        help=f"Quadratic regressors (available: {', '.join(QUADRATIC)})",
    )
    parser.add_argument(
        "--quadratic-id", nargs="*", default=None,
        help=f"Quadratic-ID regressors (available: {', '.join(QUADRATIC_ID)})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        winners_only=args.winners_only,
        continental_only=args.continental_only,
        modular_regressors=args.modular,
        quadratic_regressors=args.quadratic,
        quadratic_id_regressors=args.quadratic_id,
    )
