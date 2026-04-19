"""Raw data I/O + derived context for the C-block application.

The continental filter (drop AK/HI/PR/Guam/etc.) is applied by load_raw once
and never revisited; every downstream array (BTAs, bidders, adjacency,
distances, travel matrices) is consistent with the filter.
"""
import numpy as np
import pandas as pd
from pathlib import Path

from .covariates import WEIGHT_ROUNDING_TICK, QUADRATIC

DATASETS = Path(__file__).parent / "datasets"
RAW = DATASETS / "114402-V1" / "Replication-Fox-and-Bajari" / "data"

#  Market strings in btadata's `market` column are truncated to 8 chars in the
#  source CSV; these prefixes match AK/HI/PR/Guam/USVI/ASamoa/NMI markets.
NON_CONTINENTAL = (
    "Anchorag", "Fairbank", "Juneau,",
    "Hilo, HI", "Honolulu", "Kahului,", "Lihue, H",
    "San Juan", "Mayaguez",
    "Guam", "US Virgi", "American", "Northern",
)

# Fox-Bajari swap: bidder 190 ↔ 234 (DCR/DCC alias correction).
_FOX_SWAP = {190: 234, 234: 190}


def _obs_bundles(bidders, btas):
    n_obs, n_items = len(bidders), len(btas)
    m = np.zeros((n_obs, n_items), dtype=bool)
    winners_by_item = btas["bidder_num_fox"].values
    for j, winner in enumerate(winners_by_item):
        m[bidders["bidder_num_fox"].values == winner, j] = True
    return m


def load_raw():
    """Load raw CSVs + apply the continental filter. Pure data I/O."""
    btas    = pd.read_csv(RAW / "btadata_2004_03_12_1.csv")
    bidders = pd.read_csv(RAW / "biddercblk_03_28_2004_pln.csv")
    f175    = pd.read_csv(RAW / "fccform175nomiss.csv", header=None)

    # Historical fixes from Fox-Bajari's replication archive:
    #   * bidder 67 has a missing pops_eligible in the main file; pull it from
    #     FCC Form 175 (column 3 in the headerless CSV = eligibility).
    #   * bidder_num == 9999 is a sentinel row for unused slots; drop it.
    bidders.loc[bidders["bidder_num_fox"] == 67, "pops_eligible"] = (
        f175.loc[f175[0] == 67, 3].item()
    )
    bidders = bidders[bidders["bidder_num"] != 9999]
    f175 = f175.rename(columns={0: "bidder_num_fox", 1: "assets",
                                2: "revenues", 3: "eligibility_form175"})
    bidders = bidders.merge(f175, on="bidder_num_fox")

    extra = pd.read_csv(DATASETS / "bidder_data.csv")[
        ["bidder_num_fox", "bta", "Applicant_Status"]
    ].rename(columns={"bta": "hq_bta"})
    extra["designated"] = extra["Applicant_Status"].notna().astype(int)
    bidders = bidders.merge(extra[["bidder_num_fox", "hq_bta", "designated"]],
                            on="bidder_num_fox")

    # DCR/DCC swap (uses the same mapping as _FOX_SWAP used downstream).
    swapped = bidders["bidder_num_fox"].map(_FOX_SWAP).fillna(bidders["bidder_num_fox"])
    bidders["bidder_num_fox"] = swapped.astype(int)

    adj   = pd.read_csv(RAW / "btamatrix_merged.csv", header=None)\
              .drop(columns=0).values.astype(float)
    geo   = pd.read_csv(RAW / "distancesmat_dio_perl_fixed.dat",
                        delimiter=' ', header=None)
    geo   = geo.drop(columns=geo.columns[-1]).values.astype(float)
    trav  = pd.read_csv(RAW / "american-travel-survey-1995-zero.csv",
                        header=None).values.astype(float)
    air   = pd.read_csv(RAW / "air-travel-passengers-bta-year-1994.csv",
                        header=None).values.astype(float)

    # Continental filter. Drop non-continental BTAs; keep bidders who won
    # at least one continental BTA, plus loser bidders (no wins).
    is_cont = ~btas["market"].isin(NON_CONTINENTAL)
    cont    = np.where(is_cont)[0]

    obs_pre  = _obs_bundles(bidders, btas)
    cont_win = obs_pre[:, cont].sum(1)
    any_win  = obs_pre.sum(1)
    keep_bidders = np.where((cont_win > 0) | (any_win == 0))[0]

    bidders = bidders.iloc[keep_bidders].reset_index(drop=True)
    btas    = btas.iloc[cont].reset_index(drop=True)
    adj     = adj[cont][:, cont]
    geo     = geo[cont][:, cont]
    trav    = trav[cont][:, cont]
    air     = air[cont][:, cont]

    # Continental-share normalizations (dimensionless, sum to 1 over continental
    # BTAs). Used by the IV regression's `pop` regressor and by downstream
    # compute_xi; same convention as ctx["pop"] / ctx["price_share"].
    btas["pop90_share"] = btas["pop90"] / btas["pop90"].sum()
    btas["bid_share"]   = btas["bid"]   / btas["bid"].sum()

    return dict(bta_data=btas, bidder_data=bidders,
                bta_adjacency=adj, geo_distance=geo,
                travel_survey=trav, air_travel=air)


def build_context(raw):
    """Normalized, derived quantities keyed by covariate-registry names.

    Convention: pop_j and elig_i are both divided by the continental total
    population, so (elig_i * pop_j) is dimensionless and pop-scaled errors
    share the same scale as the elig_pop covariate.
    """
    btas, bidders = raw["bta_data"], raw["bidder_data"]
    pop90 = btas["pop90"].to_numpy(dtype=float)
    elig  = bidders["pops_eligible"].to_numpy(dtype=float)
    assets = bidders["assets"].to_numpy(dtype=float)
    desig  = bidders["designated"].to_numpy(dtype=float)
    pop_sum = pop90.sum()

    bta_nums = btas["bta"].astype(int).values
    bta_idx  = {b: i for i, b in enumerate(bta_nums)}
    hq_idx   = np.array([bta_idx.get(int(h), 0) for h in bidders["hq_bta"]])

    return dict(
        # integer capacities / weights for the combinatorial solver
        weight   = (pop90 // WEIGHT_ROUNDING_TICK).astype(int),
        capacity = (elig  // WEIGHT_ROUNDING_TICK).astype(int),
        # normalized (same pop_sum divisor → consistent across covariates & errors)
        pop = pop90 / pop_sum,
        elig = elig / pop_sum,
        pop_sum = pop_sum,
        # observed bundles (post continental filter)
        c_obs_bundles = _obs_bundles(bidders, btas),
        # geography & interaction matrices
        geo_distance  = raw["geo_distance"],
        bta_adjacency = raw["bta_adjacency"],
        travel_survey = raw["travel_survey"],
        air_travel    = raw["air_travel"],
        # BTA covariates. percapin/density/imwl are on raw scales so we divide
        # by their max; hhinc35k is already a share in [0, 1].
        percapin = btas["percapin"].to_numpy(dtype=float) / btas["percapin"].max(),
        hhinc35k = btas["hhinc35k"].to_numpy(dtype=float),
        density  = btas["density"].to_numpy(dtype=float)  / btas["density"].max(),
        imwl     = btas["imwl"].to_numpy(dtype=float)     / btas["imwl"].max(),
        price    = btas["bid"].to_numpy(dtype=float) / 1e9,
        # share-normalized price (bid_j / Σ_k bid_k), same convention as `pop`.
        # Used by the pop_price error specification.
        price_share = btas["bid"].to_numpy(dtype=float) / btas["bid"].sum(),
        # bidder covariates
        assets     = assets / assets.max(),
        designated = desig,
        hq_bta_idx = hq_idx,
    )


def load_aggregation_matrix(btas):
    """Return (A, mta_nums) where A is the BTA→MTA incidence (n_mta, n_bta)."""
    df = pd.read_csv(RAW / "cntysv2000_census-bta-may2009.csv", encoding="latin-1")
    df = df.assign(bta=pd.to_numeric(df["BTA"], errors="coerce"),
                   mta=pd.to_numeric(df["MTA"], errors="coerce")).dropna()
    df = df[df["bta"].isin(btas)]

    bta_col = {b: i for i, b in enumerate(btas)}
    mta_nums = sorted(df["mta"].unique().astype(int).tolist())
    mta_row = {m: i for i, m in enumerate(mta_nums)}

    A = np.zeros((len(mta_nums), len(btas)))
    for _, r in df.iterrows():
        A[mta_row[int(r["mta"])], bta_col[int(r["bta"])]] = 1
    return A, mta_nums


# ── Winner filtering and last-round capacity ──────────────────────────

def filter_winners(input_data):
    """Keep only observations (bidders) with a nonzero observed bundle.
    Returns (input_data, keep_mask). Updates every id_data array in place."""
    b = input_data["id_data"]["obs_bundles"]
    keep = b.sum(1) > 0
    for k, v in input_data["id_data"].items():
        if isinstance(v, np.ndarray) and v.shape[0] == keep.shape[0]:
            input_data["id_data"][k] = v[keep]
    return input_data, keep


def last_round_capacity(bidder_data, keep_mask=None):
    """Per-bidder capacity from their last active round, in WEIGHT_ROUNDING_TICK units."""
    elig = pd.read_csv(RAW / "cblock-eligibility.csv")
    if keep_mask is not None:
        bidder_data = bidder_data.iloc[np.where(keep_mask)[0]].reset_index(drop=True)

    baseline = bidder_data["pops_eligible"].to_numpy(dtype=float)
    out = np.zeros(len(bidder_data), dtype=int)
    for i, fox in enumerate(bidder_data["bidder_num_fox"].astype(int)):
        orig = _FOX_SWAP.get(fox, fox)
        sub = elig[(elig["bidder_num_fox"] == orig) & (elig["max_elig"] > 0)]
        if sub.empty:
            out[i] = int(baseline[i] // WEIGHT_ROUNDING_TICK)
            continue
        last = sub.sort_values("round_num").iloc[-1]["max_elig"]
        r1 = elig[(elig["bidder_num_fox"] == orig) & (elig["round_num"] == 1)]["max_elig"]
        if len(r1) and r1.iloc[0] > 0:
            out[i] = int(np.round(baseline[i] * last / r1.iloc[0] // WEIGHT_ROUNDING_TICK))
    return out


def cholesky_factor(ctx, covariate_name):
    """Cholesky factor L of Σ, where Σ = sym(Q) with its diagonal *set* to 1,
    and Q is the QUADRATIC covariate with `covariate_name`. Returns None if
    `covariate_name` is None.

    Note: the diagonal is set to 1 (not rescaled). Off-diagonals are assumed
    already small enough that the result is PSD — the Cholesky call is the
    hard PSD check. Raises np.linalg.LinAlgError if not PSD.
    """
    if covariate_name is None:
        return None
    Q = QUADRATIC[covariate_name](ctx)
    Sigma = (Q + Q.T) / 2
    np.fill_diagonal(Sigma, 1.0)
    return np.linalg.cholesky(Sigma)
