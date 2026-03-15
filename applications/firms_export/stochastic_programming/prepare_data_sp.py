from pathlib import Path
import sys
import pickle
import hashlib
import numpy as np

SP_DIR = Path(__file__).resolve().parent
DATA_DIR = SP_DIR.parent / "data"
CACHE_DIR = SP_DIR / ".cache"
sys.path.insert(0, str(DATA_DIR))
from prepare_data import (
    load_raw_data, filter_dataframe, build_revenue, compute_exp_rev,
    build_all_gravity_features, build_obs_bundles_by_year,
)


def _zero_diag(M):
    out = M.copy()
    np.fill_diagonal(out, 0.0)
    return out


def build_context_sp(dataframe, expected_revenue, pairwise_features,
                     home_to_dest, destinations, home,
                     beta=0.8, end_buffer=3, n_sample=None,
                     unconstrained=False):
    all_years = sorted(dataframe["y"].unique())
    firms_all = dataframe["f"].unique()
    n_dest = len(destinations)

    obs_all = build_obs_bundles_by_year(dataframe, firms_all, all_years, destinations)
    # obs_all: (n_firms_all, n_all_years, n_dest)

    # Find entry year for each firm (first year with any export)
    any_export = obs_all.any(axis=2)
    first_year_idx = any_export.argmax(axis=1)
    has_export = any_export.any(axis=1)

    year_0 = all_years[0]
    year_last = all_years[-1]
    entry_years = np.array(all_years)[first_year_idx]

    # Keep only firms that entered AFTER the first year of the dataset
    # and whose entry year allows at least end_buffer years of "period 2" data
    max_entry_year = year_last - end_buffer
    firm_mask = has_export & (entry_years > year_0) & (entry_years <= max_entry_year)
    firms = firms_all[firm_mask]
    entry_years_kept = entry_years[firm_mask]

    obs_kept = obs_all[firm_mask]       # (n_firms, n_all_years, n_dest)
    exp_rev_kept = expected_revenue[firm_mask]  # (n_firms, n_all_years, n_dest)

    # For each firm, the observation years are [entry_year, ..., max_entry_year].
    # Each (firm, year) pair becomes one observation.
    # period 1 = this year's bundle, period 2 = perpetuity of future choices.
    # state_chars = previous year's bundle.
    # For period 2 "observed bundle" (Q-function), use next year's actual choice.

    year_to_idx = {y: i for i, y in enumerate(all_years)}
    obs_years = [y for y in all_years if year_0 < y <= max_entry_year]

    records = []
    for fi in range(len(firms)):
        ey = entry_years_kept[fi]
        for y in obs_years:
            if y < ey:
                continue
            ti = year_to_idx[y]
            records.append((fi, ti, y))

    n_obs = len(records)
    firm_idx = np.array([r[0] for r in records])
    year_idx = np.array([r[1] for r in records])

    # Build arrays: bundles, state, revenue, obs_bundles_2
    obs_bundles = np.zeros((n_obs, n_dest), dtype=bool)
    state_chars = np.zeros((n_obs, n_dest), dtype=float)
    rev_chars_1 = np.zeros((n_obs, 1, n_dest))     # (n_obs, 1, M) — 1 revenue char
    rev_chars_2 = np.zeros((n_obs, 1, n_dest))     # period 2 revenue
    obs_bundles_2 = np.zeros((n_obs, n_dest), dtype=bool)  # next year actual choice

    for obs_i, (fi, ti, y) in enumerate(records):
        obs_bundles[obs_i] = obs_kept[fi, ti]
        # state = previous year's bundle (ti-1); if first year, all zeros
        if ti > 0:
            state_chars[obs_i] = obs_kept[fi, ti - 1].astype(float)

        # Revenue characteristics: expected revenue for this firm-year-dest
        rev_chars_1[obs_i, 0] = exp_rev_kept[fi, ti]

        # Period 2 revenue: average of remaining years [ti+1, ...]
        future = exp_rev_kept[fi, ti + 1:]
        if len(future) > 0:
            rev_chars_2[obs_i, 0] = future.mean(axis=0)
        else:
            rev_chars_2[obs_i, 0] = exp_rev_kept[fi, ti]

        # Observed period 2 bundle: next year's actual choice
        if ti + 1 < len(all_years):
            obs_bundles_2[obs_i] = obs_kept[fi, ti + 1]

    if n_sample is not None and n_sample < n_obs:
        idx = np.random.default_rng(0).choice(n_obs, n_sample, replace=False)
        idx.sort()
        obs_bundles = obs_bundles[idx]
        state_chars = state_chars[idx]
        rev_chars_1 = rev_chars_1[idx]
        rev_chars_2 = rev_chars_2[idx]
        obs_bundles_2 = obs_bundles_2[idx]
        firm_idx = firm_idx[idx]
        year_idx = year_idx[idx]
        n_obs = n_sample

    # Capacity
    if unconstrained:
        capacity = np.full(n_obs, n_dest + 1)
    else:
        cap_per_firm = obs_kept.any(axis=1).sum(axis=1)   # (n_firms,)
        capacity = cap_per_firm[firm_idx]

    # Revenue in millions of dollars
    rev_chars_1 = rev_chars_1 / 1e6
    rev_chars_2 = rev_chars_2 / 1e6

    # Distance in thousands of km
    dist_home = home_to_dest["dist"] / 1e3
    entry_chars = dist_home

    # Synergy: proximity = exp(-dist) where dist is in thousands of km
    dist_raw = pairwise_features["dist"].values.astype(float) / 1e3
    syn_chars = _zero_diag(np.exp(-dist_raw))

    sizes = obs_bundles.sum(1)
    print(f"\nSP Context: {n_obs} obs from {len(firms)} firms, {n_dest} destinations")
    print(f"  Entry years: {all_years[1]}-{max_entry_year}, beta={beta}, buffer={end_buffer}")
    print(f"  Bundle sizes: mean={sizes.mean():.1f}, "
          f"median={np.median(sizes):.0f}, max={sizes.max()}")
    print(f"  Obs years: {obs_years[0]}-{obs_years[-1]}, {len(obs_years)} years")

    return {
        "n_obs": n_obs,
        "n_dest": n_dest,
        "M": n_dest,
        "firms": firms,
        "destinations": destinations,
        "beta": beta,
        "obs_bundles": obs_bundles,
        "obs_bundles_2": obs_bundles_2,
        "state_chars": state_chars,
        "rev_chars_1": rev_chars_1,
        "rev_chars_2": rev_chars_2,
        "entry_chars": entry_chars,
        "syn_chars": syn_chars,
        "capacity": capacity,
        "firm_idx": firm_idx,
        "year_idx": year_idx,
    }


def build_input_data_sp(ctx, R):
    return {
        "id_data": {
            "state_chars":   ctx["state_chars"],
            "capacity":      ctx["capacity"],
            "rev_chars_1":   ctx["rev_chars_1"],
            "rev_chars_2":   ctx["rev_chars_2"],
            "obs_bundles":   ctx["obs_bundles"],
            "obs_bundles_2": ctx["obs_bundles_2"],
        },
        "item_data": {
            "syn_chars":   ctx["syn_chars"],
            "entry_chars": ctx["entry_chars"],
            "beta":        ctx["beta"],
            "R":           R,
        },
    }


def main(country="MEX", keep_top=50, beta=0.8, end_buffer=3, n_sample=None,
         unconstrained=False):
    key = repr((country, keep_top, beta, end_buffer, n_sample, unconstrained))
    key_hash = hashlib.md5(key.encode()).hexdigest()[:8]
    cache_file = CACHE_DIR / f"ctx_{key_hash}.pkl"
    if cache_file.exists():
        print(f"Loading cached context from {cache_file.name}")
        return pickle.loads(cache_file.read_bytes())

    dataframe = filter_dataframe(keep_top, load_raw_data(country))
    revenue = build_revenue(dataframe)
    expected_revenue = compute_exp_rev(revenue)
    pairwise, dest_feats, home_to_dest, destinations = \
        build_all_gravity_features(dataframe, DATA_DIR / "datasets", home=country)
    ctx = build_context_sp(dataframe, expected_revenue, pairwise,
                           home_to_dest, destinations, country,
                           beta=beta, end_buffer=end_buffer, n_sample=n_sample,
                           unconstrained=unconstrained)

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_bytes(pickle.dumps(ctx))
    print(f"Cached context to {cache_file.name}")
    return ctx


if __name__ == "__main__":
    ctx = main()
    input_data = build_input_data_sp(ctx, R=50)
    for src in ["id_data", "item_data"]:
        print(f"\n{src}:")
        for k, v in input_data[src].items():
            shape = v.shape if hasattr(v, 'shape') else v
            print(f"  {k:<20} {shape}")
