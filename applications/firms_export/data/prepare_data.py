from pathlib import Path
import zipfile
import urllib.request
import pickle
import hashlib
import numpy as np
import pandas as pd
from scipy.special import logsumexp

SP_DIR = Path(__file__).resolve().parent
DATA_DIR = SP_DIR.parent / "data" / "datasets"
CACHE_DIR = SP_DIR / ".cache"


def load_raw_data(country):
    return pd.read_stata(DATA_DIR / f"{country}.dta")


def filter_dataframe(keep_top, dataframe):
    dataframe = dataframe.drop(columns=["q", "c"])
    dataframe = dataframe[dataframe["d"] != "OTH"]
    top_d = dataframe["d"].value_counts().nlargest(keep_top).index
    return dataframe[dataframe["d"].isin(top_d)]


def build_revenue(dataframe):
    revenue_df = dataframe.pivot_table(
        index=["f", "y"], columns="d", values="v",
        aggfunc="sum", fill_value=0,
    )
    firms = dataframe["f"].unique()
    years = dataframe["y"].unique()
    full_index = pd.MultiIndex.from_product([firms, years], names=["f", "y"])
    revenue_df = revenue_df.reindex(full_index, fill_value=0).reset_index()
    vals_columns = revenue_df.columns[2:]
    return revenue_df[vals_columns].to_numpy().reshape(
        len(firms), len(years), len(vals_columns))


def build_obs_bundles_by_year(dataframe, firms, years, destinations):
    firm_map = {f: i for i, f in enumerate(firms)}
    year_map = {y: i for i, y in enumerate(years)}
    dest_map = {d: i for i, d in enumerate(destinations)}
    obs = np.zeros((len(firms), len(years), len(destinations)), dtype=bool)
    df = dataframe[dataframe["v"] > 0]
    fi = df["f"].map(firm_map)
    yi = df["y"].map(year_map)
    di = df["d"].map(dest_map)
    valid = fi.notna() & yi.notna() & di.notna()
    obs[fi[valid].astype(int).values,
        yi[valid].astype(int).values,
        di[valid].astype(int).values] = True
    return obs


def _build_tensor(FEs, shape):
    tensor = np.zeros(shape=shape)
    n_axis = len(shape)
    for axis, FE in FEs.items():
        shp = [1] * n_axis
        shp[axis] = -1
        tensor += FE.reshape(shp)
    return tensor


def _build_tensor_except(FEs, shape, skip_axis):
    tensor = np.zeros(shape=shape)
    n_axis = len(shape)
    for axis, FE in FEs.items():
        if axis == skip_axis:
            continue
        shp = [1] * n_axis
        shp[axis] = -1
        tensor += FE.reshape(shp)
    return tensor


def compute_exp_rev(y, max_iters=100):
    shape = y.shape
    n_axis = len(shape)
    FEs = {}
    log_marg_y = {}
    other = {}
    for axis, length in enumerate(shape):
        FEs[axis] = np.zeros(length)
        other_ax = tuple(i for i in range(n_axis) if i != axis)
        other[axis] = other_ax
        log_marg_y[axis] = np.log(y.sum(axis=other_ax))
    for iter in range(max_iters):
        max_delta = 0.0
        for axis in range(n_axis):
            tensor_except = _build_tensor_except(FEs, shape, axis)
            updated = log_marg_y[axis] - logsumexp(tensor_except, axis=other[axis])
            max_delta = max(max_delta, float(np.max(np.abs(updated - FEs[axis]))))
            FEs[axis] = updated
        if max_delta < 1e-12:
            break
    return np.exp(_build_tensor(FEs, shape))


CEPII_GRAVITY_URL = "https://www.cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip"

RAW_TO_CEPII = {
    "VIR": None,
}

GRAVITY_TIME_INVARIANT = [
    "dist",
    "distw_harmonic",
    "distw_arithmetic",
    "contig",
    "comlang_off",
    "comlang_ethno",
    "col_dep_ever",
    "comcol",
    "comrelig",
    "comleg_posttrans",
]

DIAG_ZERO_VARS = {"dist", "distw_harmonic", "distw_arithmetic",
                   "col_dep_ever", "comcol"}

GRAVITY_TIME_VARYING = [
    "fta_wto",
    "rta_coverage",
]

NAN_FILL_ZERO = {"col_dep_ever", "rta_coverage"}

DEST_TIME_VARYING = ["gdp_d", "gdpcap_d", "pop_d", "eu_d"]

ALL_GRAVITY_VARS = GRAVITY_TIME_INVARIANT + GRAVITY_TIME_VARYING


def download_cepii_gravity(cache_dir):
    cache_dir = cache_dir / "cepii_gravity"
    cache_dir.mkdir(parents=True, exist_ok=True)
    csv_files = list(cache_dir.glob("Gravity_V*.csv"))
    if csv_files:
        return csv_files[0]
    zip_path = cache_dir / "gravity.zip"
    urllib.request.urlretrieve(CEPII_GRAVITY_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(cache_dir)
    zip_path.unlink()
    csv_files = list(cache_dir.glob("Gravity_V*.csv"))
    if not csv_files:
        raise FileNotFoundError("No Gravity_V*.csv found in extracted CEPII archive")
    return csv_files[0]


def validate_destinations_against_cepii(destinations, cache_dir):
    countries_path = cache_dir / "cepii_gravity" / "Countries_V202211.csv"
    if not countries_path.exists():
        download_cepii_gravity(cache_dir)
    cepii_iso3 = set(pd.read_csv(countries_path)["iso3"].unique())
    validated, dropped = [], []
    for d in destinations:
        if d in cepii_iso3:
            validated.append(d)
        elif d in RAW_TO_CEPII:
            mapped = RAW_TO_CEPII[d]
            if mapped is not None:
                validated.append(mapped)
            else:
                dropped.append(d)
        else:
            dropped.append(d)
    return validated


def load_cepii_gravity(cache_dir, destinations, home=None):
    csv_path = download_cepii_gravity(cache_dir)
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    desired_cols = (["iso3_o", "iso3_d", "year"]
                    + ALL_GRAVITY_VARS + DEST_TIME_VARYING)
    usecols = [c for c in desired_cols if c in header]
    relevant = set(destinations) | ({home} if home else set())
    filtered = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=100_000):
        mask = chunk["iso3_o"].isin(relevant) & chunk["iso3_d"].isin(relevant)
        if mask.any():
            filtered.append(chunk[mask])
    gravity_df = pd.concat(filtered, ignore_index=True)
    return gravity_df


def build_pairwise_df(gravity_df, destinations, variable, year=None):
    if variable not in gravity_df.columns:
        return None
    df = gravity_df[["iso3_o", "iso3_d", "year", variable]].copy()
    if year is not None:
        use_year = min(year, df["year"].max())
        df = df[df["year"] == use_year]
    else:
        df = df.dropna(subset=[variable])
        df = df.sort_values("year").drop_duplicates(
            subset=["iso3_o", "iso3_d"], keep="last")
    pivot = df.pivot_table(
        index="iso3_o", columns="iso3_d", values=variable, aggfunc="first")
    pivot = pivot.reindex(index=destinations, columns=destinations)
    diag_val = 0.0 if variable in DIAG_ZERO_VARS else 1.0
    for d in destinations:
        if d in pivot.index and d in pivot.columns:
            pivot.loc[d, d] = diag_val
    return pivot


def build_all_gravity_features(dataframe, home=None):
    destinations = sorted(dataframe["d"].unique())
    years = sorted(dataframe["y"].unique())

    destinations = validate_destinations_against_cepii(destinations, DATA_DIR)
    gravity_df = load_cepii_gravity(DATA_DIR, destinations, home=home)

    pairwise = {}
    for var in GRAVITY_TIME_INVARIANT:
        feat = build_pairwise_df(gravity_df, destinations, var)
        if feat is not None:
            if var in NAN_FILL_ZERO:
                feat.fillna(0.0, inplace=True)
            pairwise[var] = feat

    for var in GRAVITY_TIME_VARYING:
        for year in years:
            feat = build_pairwise_df(gravity_df, destinations, var, year=year)
            if feat is not None:
                if var in NAN_FILL_ZERO:
                    feat.fillna(0.0, inplace=True)
                pairwise[f"{var}_{year}"] = feat

    dest_features = {}
    for var in DEST_TIME_VARYING:
        feat = build_pairwise_df(gravity_df, destinations, var)
        if feat is not None:
            dest_features[var] = feat

    home_to_dest = {}
    if home:
        home_df = gravity_df[gravity_df["iso3_o"] == home]
        for var in GRAVITY_TIME_INVARIANT:
            if var not in home_df.columns:
                continue
            sub = home_df[["iso3_d", var]].dropna(subset=[var])
            sub = sub.drop_duplicates("iso3_d", keep="last")
            vals = sub.set_index("iso3_d")[var].reindex(destinations)
            home_to_dest[var] = vals.values.astype(float)

    return pairwise, dest_features, home_to_dest, destinations


def _zero_diag(M):
    out = M.copy()
    np.fill_diagonal(out, 0.0)
    return out


def build_context(dataframe, expected_revenue, pairwise_features,
                  home_to_dest, destinations, home,
                  end_buffer=3, n_sample=None, filter_active=True):
    all_years = sorted(dataframe["y"].unique())
    firms_all = dataframe["f"].unique()
    n_dest = len(destinations)

    obs_all = build_obs_bundles_by_year(dataframe, firms_all, all_years,
                                        destinations)

    any_export = obs_all.any(axis=2)
    first_year_idx = any_export.argmax(axis=1)
    has_export = any_export.any(axis=1)

    year_0 = all_years[0]
    year_last = all_years[-1]
    entry_years = np.array(all_years)[first_year_idx]

    max_entry_year = year_last - end_buffer
    firm_mask = (has_export & (entry_years > year_0)
                 & (entry_years <= max_entry_year))
    firms = firms_all[firm_mask]
    entry_years_kept = entry_years[firm_mask]

    obs_kept = obs_all[firm_mask]
    exp_rev_kept = expected_revenue[firm_mask]

    year_to_idx = {y: i for i, y in enumerate(all_years)}
    obs_years = [y for y in all_years if year_0 < y <= max_entry_year]

    records = []
    for fi in range(len(firms)):
        ey = entry_years_kept[fi]
        for y in obs_years:
            if y < ey:
                continue
            ti = year_to_idx[y]
            if filter_active and y != ey:
                if ti > 0 and not obs_kept[fi, ti - 1].any():
                    continue
            records.append((fi, ti, y))

    n_obs = len(records)
    firm_idx = np.array([r[0] for r in records])
    year_idx = np.array([r[1] for r in records])

    obs_bundles = np.zeros((n_obs, n_dest), dtype=bool)
    state_chars = np.zeros((n_obs, n_dest), dtype=float)
    rev_chars_1 = np.zeros((n_obs, 1, n_dest))
    rev_chars_2 = np.zeros((n_obs, 1, n_dest))
    obs_bundles_2 = np.zeros((n_obs, n_dest), dtype=bool)

    for obs_i, (fi, ti, y) in enumerate(records):
        obs_bundles[obs_i] = obs_kept[fi, ti]
        if ti > 0:
            state_chars[obs_i] = obs_kept[fi, ti - 1].astype(float)
        rev_chars_1[obs_i, 0] = exp_rev_kept[fi, ti]
        future = exp_rev_kept[fi, ti + 1:]
        if len(future) > 0:
            rev_chars_2[obs_i, 0] = future.mean(axis=0)
        else:
            rev_chars_2[obs_i, 0] = exp_rev_kept[fi, ti]
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

    rev_chars_1 = rev_chars_1 / 1e3
    rev_chars_2 = rev_chars_2 / 1e3

    dist_home = home_to_dest["dist"] / 1e3
    entry_chars = dist_home

    dist_raw = pairwise_features["dist"].values.astype(float) / 1e3
    syn_chars = _zero_diag(np.exp(-dist_raw))

    return {
        "n_obs": n_obs,
        "n_dest": n_dest,
        "M": n_dest,
        "firms": firms,
        "destinations": destinations,
        "obs_bundles": obs_bundles,
        "obs_bundles_2": obs_bundles_2,
        "state_chars": state_chars,
        "rev_chars_1": rev_chars_1,
        "rev_chars_2": rev_chars_2,
        "entry_chars": entry_chars,
        "home_to_dest": home_to_dest,
        "pairwise_features": pairwise_features,
        "syn_chars": syn_chars,
        "firm_idx": firm_idx,
        "year_idx": year_idx,
    }


def build_input_data(ctx, R, beta=0.0):
    return {
        "id_data": {
            "state_chars":   ctx["state_chars"],
            "rev_chars_1":   ctx["rev_chars_1"],
            "rev_chars_2":   ctx["rev_chars_2"],
            "obs_bundles":   ctx["obs_bundles"],
            "obs_bundles_2": ctx["obs_bundles_2"],
        },
        "item_data": {
            "syn_chars":   ctx["syn_chars"],
            "entry_chars": ctx["entry_chars"],
            "beta":        beta,
            "R":           R,
        },
    }


def main(country="MEX", keep_top=50, end_buffer=3, n_sample=None,
         filter_active=True):
    key = repr((country, keep_top, end_buffer, n_sample, filter_active))
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
                        end_buffer=end_buffer, n_sample=n_sample,
                        filter_active=filter_active)

    CACHE_DIR.mkdir(exist_ok=True)
    cache_file.write_bytes(pickle.dumps(ctx))
    return ctx


if __name__ == "__main__":
    ctx = main()
    input_data = build_input_data(ctx, R=50)
    for src in ["id_data", "item_data"]:
        print(f"\n{src}:")
        for k, v in input_data[src].items():
            shape = v.shape if hasattr(v, 'shape') else v
            print(f"  {k:<20} {shape}")
