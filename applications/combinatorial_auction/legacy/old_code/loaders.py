import numpy as np
import pandas as pd
from pathlib import Path
from .registries import WEIGHT_ROUNDING_TICK, QUADRATIC

BASE_DIR = Path(__file__).parent
DATASETS_DIR = BASE_DIR / "datasets"
RAW_DIR = DATASETS_DIR / "114402-V1" / "Replication-Fox-and-Bajari" / "data"
AB_DIR = DATASETS_DIR / "ab"

NON_CONTINENTAL = (
    "Anchorag", "Fairbank", "Juneau,",
    "Hilo, HI", "Honolulu", "Kahului,", "Lihue, H",
    "San Juan", "Mayaguez",
    "Guam", "US Virgi", "American", "Northern",
)


def load_bta_data():
    bta_data = pd.read_csv(RAW_DIR / "btadata_2004_03_12_1.csv")
    bidder_data = pd.read_csv(RAW_DIR / "biddercblk_03_28_2004_pln.csv")
    form175 = pd.read_csv(RAW_DIR / "fccform175nomiss.csv", header=None)

    bidder_data.loc[bidder_data["bidder_num_fox"] == 67, "pops_eligible"] = form175.loc[form175[0] == 67, 3].item()
    bidder_data = bidder_data[bidder_data["bidder_num"] != 9999]
    form175 = form175.rename(columns={0: "bidder_num_fox", 1: "assets", 2: "revenues", 3: "eligibility_form175"})
    bidder_data = bidder_data.merge(form175, on="bidder_num_fox")

    extra = pd.read_csv(DATASETS_DIR / "bidder_data.csv")[["bidder_num_fox", "bta", "Applicant_Status"]]
    extra = extra.rename(columns={"bta": "hq_bta"})
    extra["designated"] = extra["Applicant_Status"].notna().astype(int)
    bidder_data = bidder_data.merge(extra[["bidder_num_fox", "hq_bta", "designated"]], on="bidder_num_fox")

    # DCR/DCC swap
    m190, m234 = bidder_data["bidder_num_fox"] == 190, bidder_data["bidder_num_fox"] == 234
    bidder_data.loc[m190, "bidder_num_fox"] = 234
    bidder_data.loc[m234, "bidder_num_fox"] = 190

    bta_adjacency = pd.read_csv(RAW_DIR / "btamatrix_merged.csv", header=None).drop(columns=0).values.astype(float)
    geo_distance = pd.read_csv(RAW_DIR / "distancesmat_dio_perl_fixed.dat", delimiter=' ', header=None)
    geo_distance = geo_distance.drop(columns=geo_distance.columns[-1]).values.astype(float)
    travel_survey = pd.read_csv(RAW_DIR / "american-travel-survey-1995-zero.csv", header=None).values.astype(float)
    air_travel = pd.read_csv(RAW_DIR / "air-travel-passengers-bta-year-1994.csv", header=None).values.astype(float)

    # continental filter
    cont = np.where(~bta_data["market"].isin(NON_CONTINENTAL))[0]
    non_cont = np.where(bta_data["market"].isin(NON_CONTINENTAL))[0]
    obs = _c_obs_bundles(bidder_data, bta_data)
    keep = np.where((obs.sum(1) - obs[:, non_cont].sum(1) > 0) | (obs.sum(1) == 0))[0]
    bidder_data = bidder_data.iloc[keep].reset_index(drop=True)
    bta_data = bta_data.iloc[cont].reset_index(drop=True)
    bta_adjacency = bta_adjacency[cont][:, cont]
    geo_distance = geo_distance[cont][:, cont]
    travel_survey = travel_survey[cont][:, cont]
    air_travel = air_travel[cont][:, cont]

    return dict(bta_data=bta_data, bidder_data=bidder_data, bta_adjacency=bta_adjacency,
                geo_distance=geo_distance, travel_survey=travel_survey, air_travel=air_travel)


def build_context(raw):
    bta, bidder = raw["bta_data"], raw["bidder_data"]
    pop = bta["pop90"].to_numpy().astype(float)
    elig = bidder["pops_eligible"].to_numpy().astype(float)
    pop_sum = pop.sum()

    weight = np.round(pop // WEIGHT_ROUNDING_TICK).astype(int)
    capacity = np.round(elig // WEIGHT_ROUNDING_TICK).astype(int)
    assets = bidder["assets"].to_numpy().astype(float)
    designated = bidder["designated"].to_numpy().astype(float)

    bta_nums = bta["bta"].astype(int).values
    bta_idx = {b: i for i, b in enumerate(bta_nums)}
    hq_bta_idx = np.array([bta_idx.get(int(h), 0) for h in bidder["hq_bta"]])

    return dict(
        weight=weight, capacity=capacity,
        pop=pop / pop_sum, elig=elig / pop_sum, pop_sum=pop_sum,
        c_obs_bundles=_c_obs_bundles(bidder, bta),
        geo_distance=raw["geo_distance"], bta_adjacency=raw["bta_adjacency"],
        travel_survey=raw["travel_survey"], air_travel=raw["air_travel"],
        percapin=bta["percapin"].to_numpy().astype(float) / bta["percapin"].max(),
        hhinc35k=bta["hhinc35k"].to_numpy().astype(float),
        density=bta["density"].to_numpy().astype(float) / bta["density"].max(),
        imwl=bta["imwl"].to_numpy().astype(float) / bta["imwl"].max(),
        price=bta["bid"].to_numpy().astype(float) / 1e9,
        assets=assets / assets.max(),
        designated=designated,
        hq_bta_idx=hq_bta_idx,
    )


def _c_obs_bundles(bidder_data, bta_data):
    n_obs, n_items = len(bidder_data), len(bta_data)
    m = np.zeros((n_obs, n_items), dtype=bool)
    for j in range(n_items):
        winner = bta_data["bidder_num_fox"].iloc[j]
        m[np.where(bidder_data["bidder_num_fox"] == winner)[0], j] = True
    return m


def load_aggregation_matrix(btas):
    df = pd.read_csv(RAW_DIR / "cntysv2000_census-bta-may2009.csv", encoding="latin-1")
    df = df.assign(bta=pd.to_numeric(df["BTA"], errors="coerce"),
                   mta=pd.to_numeric(df["MTA"], errors="coerce")).dropna()
    m = df[df["bta"].isin(btas)]
    bta_col = {b: i for i, b in enumerate(btas)}
    mta_ids = sorted(m["mta"].unique().astype(int).tolist())
    mta_row = {mta: i for i, mta in enumerate(mta_ids)}
    A = np.zeros((len(mta_ids), len(btas)))
    for _, r in m.iterrows():
        A[mta_row[int(r["mta"])], bta_col[int(r["bta"])]] = 1
    return A


def continental_mta_nums(btas):
    df = pd.read_csv(RAW_DIR / "cntysv2000_census-bta-may2009.csv", encoding="latin-1")
    df = df[pd.to_numeric(df["BTA"], errors="coerce").notna()]
    return sorted(df[df["BTA"].astype(int).isin(btas)]["MTA"].astype(int).unique())


def load_round_capacities(bidder_data, n_simulations):
    """Build (n_obs, S) capacity matrix with equally spaced rounds.

    Simulation s uses round floor(1 + s * (max_round - 1) / (S - 1)),
    so sim 0 = round 1, sim S-1 = last round, others equally spaced.

    To stay on the same scale as baseline capacity (pops_eligible // TICK),
    we use the decay ratio: cap[i,s] = baseline_cap[i] * max_elig[i,r] / max_elig[i,1].
    """
    elig_df = pd.read_csv(RAW_DIR / "cblock-eligibility.csv")
    max_round = elig_df["round_num"].max()
    n_obs = len(bidder_data)

    swap = {190: 234, 234: 190}
    elig_lookup = elig_df.set_index(["bidder_num_fox", "round_num"])["max_elig"]

    if n_simulations == 1:
        rounds = [1]
    else:
        rounds = [int(np.round(1 + s * (max_round - 1) / (n_simulations - 1)))
                  for s in range(n_simulations)]

    baseline_cap = np.round(
        bidder_data["pops_eligible"].to_numpy().astype(float) // WEIGHT_ROUNDING_TICK
    ).astype(int)

    capacity = np.zeros((n_obs, n_simulations), dtype=int)
    for i in range(n_obs):
        fox_orig = swap.get(int(bidder_data.iloc[i]["bidder_num_fox"]),
                            int(bidder_data.iloc[i]["bidder_num_fox"]))
        try:
            elig_r1 = elig_lookup.loc[(fox_orig, 1)]
        except KeyError:
            continue
        if elig_r1 <= 0:
            continue
        for s, r in enumerate(rounds):
            try:
                elig_r = elig_lookup.loc[(fox_orig, r)]
            except KeyError:
                capacity[i, s] = 0
                continue
            capacity[i, s] = int(np.round(baseline_cap[i] * elig_r / elig_r1))

    return capacity


def load_ab_data():
    winners = pd.read_csv(AB_DIR / "winning_bids.csv")
    bidders = pd.read_csv(AB_DIR / "bidders.csv")
    bidders = bidders[bidders["name"] != "FCC"].reset_index(drop=True)
    bidders["fcc_acct"] = bidders["fcc_acct"].astype(int)
    winners["fcc_acct"] = winners["fcc_acct"].astype(int)
    return winners, bidders


# ── Winner filtering and last-round capacity ──────────────────────────


def filter_winners(input_data):
    """Keep only observations with nonzero observed bundles."""
    b_obs = input_data["id_data"]["obs_bundles"]
    keep = b_obs.sum(1) > 0
    for k, v in input_data["id_data"].items():
        if isinstance(v, np.ndarray) and v.shape[0] == len(keep):
            input_data["id_data"][k] = v[keep]
    return input_data, keep


def last_round_capacity(bidder_data, keep_mask=None):
    """Compute capacity from each bidder's last active round.

    Returns array of shape (n_kept,) in the same units as baseline capacity
    (pops_eligible // WEIGHT_ROUNDING_TICK).
    """
    elig_df = pd.read_csv(RAW_DIR / "cblock-eligibility.csv")
    swap = {190: 234, 234: 190}

    if keep_mask is not None:
        bidder_data = bidder_data.iloc[np.where(keep_mask)[0]].reset_index(drop=True)

    n = len(bidder_data)
    baseline = bidder_data["pops_eligible"].to_numpy().astype(float)
    capacity = np.zeros(n, dtype=int)

    for i in range(n):
        fox_swapped = int(bidder_data.iloc[i]["bidder_num_fox"])
        fox_orig = swap.get(fox_swapped, fox_swapped)
        sub = elig_df[(elig_df["bidder_num_fox"] == fox_orig) & (elig_df["max_elig"] > 0)]
        if len(sub) == 0:
            capacity[i] = int(np.round(baseline[i] // WEIGHT_ROUNDING_TICK))
            continue
        last = sub.sort_values("round_num").iloc[-1]
        r1 = elig_df[(elig_df["bidder_num_fox"] == fox_orig) & (elig_df["round_num"] == 1)]["max_elig"].iloc[0]
        if r1 > 0:
            capacity[i] = int(np.round(baseline[i] * last["max_elig"] / r1 // WEIGHT_ROUNDING_TICK))
        else:
            capacity[i] = 0

    return capacity


def build_cholesky_factor(error_correlation):
    """Build Cholesky factor from a QUADRATIC registry entry name, or return None."""
    if error_correlation is None:
        return None
    raw = load_bta_data()
    ctx = build_context(raw)
    Q = QUADRATIC[error_correlation](ctx)
    Sigma = (Q + Q.T) / 2
    np.fill_diagonal(Sigma, 1.0)
    return np.linalg.cholesky(Sigma)
