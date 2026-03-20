"""
Data preparation for the quadratic static model.

Sample selection:
  - ALL firms present in the data (no entry year filter).
  - Observations start from year_start + start_buffer.
  - A firm is "active" in year t if it exported to >= 1 destination in t,
    or if it has never yet had an all-zero year.
  - The first all-zero year IS included; subsequent years are dropped.
"""
import importlib.util
from pathlib import Path
import pickle
import hashlib
import numpy as np

_data_prep_path = str(Path(__file__).resolve().parent.parent / "data" / "prepare_data.py")
_spec = importlib.util.spec_from_file_location("data_prepare_data", _data_prep_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
load_raw_data = _mod.load_raw_data
filter_dataframe = _mod.filter_dataframe
build_revenue = _mod.build_revenue
compute_exp_rev = _mod.compute_exp_rev
build_all_gravity_features = _mod.build_all_gravity_features
build_obs_bundles_by_year = _mod.build_obs_bundles_by_year
_zero_diag = _mod._zero_diag

SP_DIR = Path(__file__).resolve().parent
CACHE_DIR = SP_DIR / ".cache"


def build_context(dataframe, expected_revenue, pairwise_features,
                  home_to_dest, destinations, home,
                  start_buffer=1, n_sample=None):
    all_years = sorted(dataframe["y"].unique())
    firms_all = dataframe["f"].unique()
    n_dest = len(destinations)

    obs_all = build_obs_bundles_by_year(dataframe, firms_all, all_years,
                                         destinations)
    has_export = obs_all.any(axis=2).any(axis=1)
    firms_mask = has_export
    firms = firms_all[firms_mask]
    obs_kept = obs_all[firms_mask]
    exp_rev_kept = expected_revenue[firms_mask]

    year_to_idx = {y: i for i, y in enumerate(all_years)}
    obs_start = all_years[start_buffer]

    any_export = obs_kept.any(axis=2)
    first_export_idx = any_export.argmax(axis=1)

    records = []
    for fi in range(len(firms)):
        entry_year = all_years[first_export_idx[fi]]
        firm_start = max(entry_year, obs_start)
        inactive = False
        for y in all_years:
            if y < firm_start:
                continue
            if inactive:
                break
            ti = year_to_idx[y]
            records.append((fi, ti, y))
            if not obs_kept[fi, ti].any():
                inactive = True

    n_obs = len(records)
    firm_idx = np.array([r[0] for r in records])
    year_idx = np.array([r[1] for r in records])

    obs_bundles = np.zeros((n_obs, n_dest), dtype=bool)
    rev_chars = np.zeros((n_obs, n_dest))

    for obs_i, (fi, ti, y) in enumerate(records):
        obs_bundles[obs_i] = obs_kept[fi, ti]
        rev_chars[obs_i] = exp_rev_kept[fi, ti]

    if n_sample is not None and n_sample < n_obs:
        idx = np.random.default_rng(0).choice(n_obs, n_sample, replace=False)
        idx.sort()
        obs_bundles = obs_bundles[idx]
        rev_chars = rev_chars[idx]
        firm_idx = firm_idx[idx]
        year_idx = year_idx[idx]
        n_obs = n_sample

    rev_chars = rev_chars / 1e3

    dist_home = home_to_dest["dist"] / 1e3
    dist_raw = pairwise_features["dist"].values.astype(float) / 1e3
    syn_chars = _zero_diag(np.exp(-dist_raw))

    return {
        "n_obs": n_obs,
        "n_dest": n_dest,
        "M": n_dest,
        "firms": firms,
        "destinations": destinations,
        "obs_bundles": obs_bundles,
        "rev_chars": rev_chars,
        "dist_home": dist_home,
        "syn_chars": syn_chars,
        "pairwise_features": pairwise_features,
        "home_to_dest": home_to_dest,
        "firm_idx": firm_idx,
        "year_idx": year_idx,
    }


def main(country="MEX", keep_top=20, start_buffer=1, n_sample=None):
    key = repr(("quadratic_static", country, keep_top, start_buffer, n_sample))
    key_hash = hashlib.md5(key.encode()).hexdigest()[:8]
    cache_file = CACHE_DIR / f"ctx_{key_hash}.pkl"
    if cache_file.exists():
        return pickle.loads(cache_file.read_bytes())

    dataframe = filter_dataframe(keep_top, load_raw_data(country))
    revenue = build_revenue(dataframe) / 1e3
    expected_revenue = compute_exp_rev(revenue)
    pairwise, dest_feats, home_to_dest, destinations = \
        build_all_gravity_features(dataframe, home=country)
    ctx = build_context(dataframe, expected_revenue, pairwise,
                        home_to_dest, destinations, country,
                        start_buffer=start_buffer, n_sample=n_sample)

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_bytes(pickle.dumps(ctx))
    return ctx


if __name__ == "__main__":
    ctx = main()
    print(f"n_obs={ctx['n_obs']}, M={ctx['M']}")
    print(f"unique firms: {len(np.unique(ctx['firm_idx']))}")
    print(f"bundle sizes: mean={ctx['obs_bundles'].sum(1).mean():.2f}, "
          f"max={ctx['obs_bundles'].sum(1).max()}, "
          f"zero={( ~ctx['obs_bundles'].any(1)).mean():.1%}")
