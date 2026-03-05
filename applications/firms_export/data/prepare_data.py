from pathlib import Path
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from scipy.special import logsumexp


def load_raw_data(country):
    data_dir = Path(__file__).resolve().parent / "datasets"
    return pd.read_stata(data_dir / f"{country}.dta")


def filter_dataframe(keep_top, dataframe):
    dataframe = dataframe.drop(columns=["q", "c"])
    dataframe = dataframe[dataframe["d"] != "OTH"]
    top_d = dataframe["d"].value_counts().nlargest(keep_top).index
    return dataframe[dataframe["d"].isin(top_d)]


def build_revenue(dataframe):
    revenue_df = dataframe.pivot_table(
        index=["f", "y"],
        columns="d",
        values="v",
        aggfunc="sum",
        fill_value=0,
    )

    firms = dataframe["f"].unique()
    years = dataframe["y"].unique()
    full_index = pd.MultiIndex.from_product([firms, years], names=["f", "y"])

    revenue_df = revenue_df.reindex(full_index, fill_value=0).reset_index()
    vals_columns = revenue_df.columns[2:]
    revenue = revenue_df[vals_columns].to_numpy().reshape((len(firms), len(years), len(vals_columns)))
    return revenue


def build_tensor(FEs, shape):
    tensor = np.zeros(shape=shape)
    n_axis = len(shape)
    for axis, FE in FEs.items():
        shp = [1] * n_axis
        shp[axis] = -1
        tensor += FE.reshape(shp)
    return tensor


def build_tensor_except(FEs, shape, skip_axis):
    tensor = np.zeros(shape=shape)
    n_axis = len(shape)
    for axis, FE in FEs.items():
        if axis == skip_axis:
            continue
        shp = [1] * n_axis
        shp[axis] = -1
        tensor += FE.reshape(shp)
    return tensor

def compute_exp_rev(y, max_iters = 100):
    shape = y.shape
    n_axis = len(shape)
    FEs = {}
    log_marg_y = {}
    other = {}

    for axis, length in enumerate(shape):
        FEs[axis] = np.zeros(length)
        other_ax = tuple(i for i in range(n_axis) if i != axis)
        other[axis] = other_ax
        log_marg_y[axis] = np.log(y.sum(axis = other_ax))

    for iter in range(max_iters):
        max_delta = 0.0
        for axis in range(n_axis):
            tensor_except = build_tensor_except(FEs, shape, axis)
            updated = log_marg_y[axis] - logsumexp(tensor_except, axis=other[axis])
            max_delta = max(max_delta, float(np.max(np.abs(updated - FEs[axis]))))
            FEs[axis] = updated
        if max_delta < 1e-12:
            break

    print(f"Converged in {iter+1} iterations (max_delta={max_delta:.3e})")
    return np.exp(build_tensor(FEs, shape))


# --- CEPII Gravity Features (V202211, Conte, Cotterlaz & Mayer 2022) ---

CEPII_GRAVITY_URL = "https://www.cepii.fr/DATA_DOWNLOAD/gravity/data/Gravity_csv_V202211.zip"

# Raw data codes not in CEPII: map to equivalent or None to drop
RAW_TO_CEPII = {
    "VIR": None,  # US Virgin Islands — not in CEPII
}

GRAVITY_TIME_INVARIANT = [
    "dist",             # Distance between main cities (km)
    "distw_harmonic",   # Population-weighted distance (harmonic)
    "distw_arithmetic", # Population-weighted distance (arithmetic)
    "contig",           # Contiguity (share a border)
    "comlang_off",      # Common official language
    "comlang_ethno",    # Common ethnic language (>= 9% speakers)
    "col_dep_ever",     # Ever had colonial dependency
    "comcol",           # Common colonizer post-1945
    "comrelig",         # Religious proximity index [0,1]
    "comleg_posttrans",  # Common legal origin (post-transition)
]

DIAG_ZERO_VARS = {"dist", "distw_harmonic", "distw_arithmetic",
                   "col_dep_ever", "comcol"}

GRAVITY_TIME_VARYING = [
    "fta_wto",              # Free trade agreement (notified to WTO)
    "rta_coverage",         # RTA depth: 1=goods, 2=+services, 3=+investment
]

NAN_FILL_ZERO = {"col_dep_ever", "rta_coverage"}

DEST_TIME_VARYING = [
    "gdp_d", "gdpcap_d", "pop_d",
    "eu_d",
]

GRAVITY_GMT_COLS = ["gmt_offset_2020_o", "gmt_offset_2020_d"]
ALL_GRAVITY_VARS = GRAVITY_TIME_INVARIANT + GRAVITY_TIME_VARYING


def download_cepii_gravity(cache_dir):
    cache_dir = cache_dir / "cepii_gravity"
    cache_dir.mkdir(parents=True, exist_ok=True)

    csv_files = list(cache_dir.glob("Gravity_V*.csv"))
    if csv_files:
        print(f"CEPII Gravity data already cached at {csv_files[0].name}")
        return csv_files[0]

    zip_path = cache_dir / "gravity.zip"
    print("Downloading CEPII Gravity dataset (~200 MB)...")
    urllib.request.urlretrieve(CEPII_GRAVITY_URL, zip_path)
    print("Download complete. Extracting...")

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(cache_dir)
    zip_path.unlink()

    csv_files = list(cache_dir.glob("Gravity_V*.csv"))
    if not csv_files:
        raise FileNotFoundError("No Gravity_V*.csv found in extracted CEPII archive")

    print(f"Extracted: {csv_files[0].name}")
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
                print(f"  {d} -> mapped to {mapped}")
                validated.append(mapped)
            else:
                print(f"  {d} -> dropped (no CEPII equivalent)")
                dropped.append(d)
        else:
            print(f"  WARNING: {d} not in CEPII and no mapping defined — dropped")
            dropped.append(d)

    if dropped:
        print(f"  Dropped {len(dropped)} destinations: {dropped}")
    else:
        print(f"  All {len(validated)} destinations validated against CEPII")

    return validated


def load_cepii_gravity(cache_dir, destinations, home=None):
    csv_path = download_cepii_gravity(cache_dir)

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    desired_cols = (["iso3_o", "iso3_d", "year"]
                    + ALL_GRAVITY_VARS + GRAVITY_GMT_COLS + DEST_TIME_VARYING)
    usecols = [c for c in desired_cols if c in header]
    missing = [c for c in desired_cols if c not in header]
    if missing:
        print(f"  Warning: columns not found in CEPII data: {missing}")

    print("Reading CEPII Gravity CSV (chunked)...")
    relevant = set(destinations) | ({home} if home else set())
    filtered = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=100_000):
        mask = chunk["iso3_o"].isin(relevant) & chunk["iso3_d"].isin(relevant)
        if mask.any():
            filtered.append(chunk[mask])

    gravity_df = pd.concat(filtered, ignore_index=True)
    print(f"Loaded {len(gravity_df)} rows for {len(destinations)} destinations")
    return gravity_df


def build_pairwise_df(gravity_df, destinations, variable, year=None):
    if variable not in gravity_df.columns:
        print(f"  Skipping {variable}: not in CEPII data")
        return None

    df = gravity_df[["iso3_o", "iso3_d", "year", variable]].copy()

    if year is not None:
        max_cepii_year = df["year"].max()
        use_year = min(year, max_cepii_year)
        if use_year != year:
            print(f"  {variable}: year {year} > max CEPII year {max_cepii_year}, using {max_cepii_year}")
        df = df[df["year"] == use_year]
    else:
        # CEPII has duplicate country_ids (e.g. DEU.1/DEU.2) where one
        # variant carries NaN. Drop NaN first, then take the most recent.
        df = df.dropna(subset=[variable])
        df = df.sort_values("year").drop_duplicates(
            subset=["iso3_o", "iso3_d"], keep="last"
        )

    pivot = df.pivot_table(
        index="iso3_o", columns="iso3_d", values=variable, aggfunc="first"
    )
    pivot = pivot.reindex(index=destinations, columns=destinations)

    # Diagonal: CEPII excludes (i,i) pairs; fill with definitional values
    diag_val = 0.0 if variable in DIAG_ZERO_VARS else 1.0
    for d in destinations:
        if d in pivot.index and d in pivot.columns:
            pivot.loc[d, d] = diag_val

    return pivot


def build_dest_df(gravity_df, destinations, variable, years):
    if variable not in gravity_df.columns:
        print(f"  Skipping {variable}: not in CEPII data")
        return None
    df = gravity_df[["iso3_d", "year", variable]].copy()
    df = df.dropna(subset=[variable])
    df = df.drop_duplicates(subset=["iso3_d", "year"], keep="last")
    pivot = df.pivot_table(index="iso3_d", columns="year", values=variable, aggfunc="first")
    pivot = pivot.reindex(index=destinations, columns=years)
    return pivot


def build_all_gravity_features(dataframe, cache_dir, home=None):
    destinations = sorted(dataframe["d"].unique())
    years = sorted(dataframe["y"].unique())

    print("Validating destination codes against CEPII:")
    destinations = validate_destinations_against_cepii(destinations, cache_dir)

    gravity_df = load_cepii_gravity(cache_dir, destinations, home=home)

    pairwise = {}

    print("Building time-invariant pairwise features:")
    for var in GRAVITY_TIME_INVARIANT:
        feat = build_pairwise_df(gravity_df, destinations, var)
        if feat is not None:
            n_missing = int(feat.isna().sum().sum())
            if var in NAN_FILL_ZERO and n_missing > 0:
                feat.fillna(0.0, inplace=True)
                print(f"  {var}: {feat.shape}, filled {n_missing} NaN with 0")
            else:
                print(f"  {var}: {feat.shape}, NaN={n_missing}")
            pairwise[var] = feat

    print("Building time-varying pairwise features:")
    for var in GRAVITY_TIME_VARYING:
        for year in years:
            feat = build_pairwise_df(gravity_df, destinations, var, year=year)
            if feat is not None:
                n_missing = int(feat.isna().sum().sum())
                if var in NAN_FILL_ZERO and n_missing > 0:
                    feat.fillna(0.0, inplace=True)
                    print(f"  {var}_{year}: {feat.shape}, filled {n_missing} NaN with 0")
                else:
                    print(f"  {var}_{year}: {feat.shape}, NaN={n_missing}")
                pairwise[f"{var}_{year}"] = feat

    # Timezone difference from GMT offsets
    if all(c in gravity_df.columns for c in GRAVITY_GMT_COLS):
        print("Building timezone difference feature:")
        gmt = gravity_df.drop_duplicates(subset=["iso3_o", "iso3_d"], keep="last")
        gmt_pivot_o = gmt.pivot_table(
            index="iso3_o", columns="iso3_d", values="gmt_offset_2020_o", aggfunc="first"
        )
        gmt_pivot_d = gmt.pivot_table(
            index="iso3_o", columns="iso3_d", values="gmt_offset_2020_d", aggfunc="first"
        )
        tz_diff = (gmt_pivot_o - gmt_pivot_d).abs()
        tz_diff = tz_diff.reindex(index=destinations, columns=destinations)
        for d in destinations:
            if d in tz_diff.index and d in tz_diff.columns:
                tz_diff.loc[d, d] = 0.0
        pairwise["tz_diff"] = tz_diff
        n_missing = tz_diff.isna().sum().sum()
        print(f"  tz_diff: {tz_diff.shape}, NaN={n_missing}")

    # Destination-level time-varying features
    dest_features = {}
    print("Building destination-level time-varying features:")
    for var in DEST_TIME_VARYING:
        feat = build_dest_df(gravity_df, destinations, var, years)
        if feat is not None:
            n_missing = int(feat.isna().sum().sum())
            print(f"  {var}: {feat.shape}, NaN={n_missing}")
            dest_features[var] = feat

    # Home-to-destination features
    home_to_dest = {}
    if home:
        print("Building home-to-destination features:")
        home_df = gravity_df[gravity_df["iso3_o"] == home]
        for var in GRAVITY_TIME_INVARIANT:
            if var not in home_df.columns:
                continue
            sub = home_df[["iso3_d", var]].dropna(subset=[var])
            sub = sub.drop_duplicates("iso3_d", keep="last")
            vals = sub.set_index("iso3_d")[var].reindex(destinations)
            home_to_dest[var] = vals.values.astype(float)
        if all(c in home_df.columns for c in GRAVITY_GMT_COLS):
            sub = home_df.drop_duplicates("iso3_d", keep="last")
            home_gmt = sub["gmt_offset_2020_o"].iloc[0]
            gmt_d = sub.set_index("iso3_d")["gmt_offset_2020_d"].reindex(destinations)
            home_to_dest["tz_diff"] = np.abs(home_gmt - gmt_d.values)
        n_nan = sum(int(np.isnan(v).sum()) for v in home_to_dest.values())
        print(f"  {len(home_to_dest)} features, total NaN={n_nan}")

    return pairwise, dest_features, home_to_dest, destinations



# ══════════════════════════════════════════════════════════════════════
# Dynamic model: items = (destination, year) pairs
# Item ordering: year-major, item index = t * n_dest + j
# ══════════════════════════════════════════════════════════════════════

YEAR_INIT = 2001
YEAR_END = 2005


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


def compute_discount_weights(entry_years, years, rho):
    years_arr = np.array(years)
    tau = years_arr[None, :] - entry_years[:, None]
    weights = np.where(tau >= 0, rho ** tau, 0.0)
    weights[:, -1] = np.where(tau[:, -1] >= 0,
                               rho ** tau[:, -1] / (1 - rho), 0.0)
    return weights


def build_context(dataframe, expected_revenue, pairwise_features, dest_features,
                  home_to_dest, destinations, home, discount_factor=0.95,
                  n_sample=None):
    all_years = sorted(dataframe["y"].unique())
    firms_all = dataframe["f"].unique()
    n_dest = len(destinations)

    obs_all = build_obs_bundles_by_year(dataframe, firms_all, all_years, destinations)
    any_export = obs_all.any(axis=2)
    entry_idx = any_export.argmax(axis=1)
    has_export = any_export.any(axis=1)
    entry_years = np.array(all_years)[entry_idx]

    firm_mask = has_export & (entry_years >= YEAR_INIT) & (entry_years <= YEAR_END)
    firms = firms_all[firm_mask]
    entry_years = entry_years[firm_mask]

    year_sel = [i for i, y in enumerate(all_years) if y >= YEAR_INIT]
    years = [all_years[i] for i in year_sel]
    n_years = len(years)
    n_items = n_dest * n_years

    obs_by_year = obs_all[firm_mask][:, year_sel]
    exp_rev = expected_revenue[firm_mask][:, year_sel]
    n_obs = len(firms)

    if n_sample is not None and n_sample < n_obs:
        idx = np.random.default_rng(0).choice(n_obs, n_sample, replace=False)
        idx.sort()
        firms, obs_by_year = firms[idx], obs_by_year[idx]
        entry_years, exp_rev = entry_years[idx], exp_rev[idx]
        n_obs = n_sample

    obs_bundles = obs_by_year.reshape(n_obs, n_items)
    discount_weights = compute_discount_weights(entry_years, years, discount_factor)
    entry_discount_weights = (1 - discount_factor) * discount_weights

    years_arr = np.array(years)
    feasible_years = (years_arr[None, :] >= entry_years[:, None])
    constraint_mask = np.repeat(feasible_years, n_dest, axis=1)

    pairwise_ti = {}
    for var in GRAVITY_TIME_INVARIANT:
        if var in pairwise_features:
            pairwise_ti[var] = pairwise_features[var].values.astype(float)
    if "tz_diff" in pairwise_features:
        pairwise_ti["tz_diff"] = pairwise_features["tz_diff"].values.astype(float)

    pairwise_tv = {}
    for var in GRAVITY_TIME_VARYING:
        pairwise_tv[var] = np.stack(
            [pairwise_features[f"{var}_{y}"].values for y in years], axis=0
        )

    dest_tv = {var: feat.values.astype(float) for var, feat in dest_features.items()}

    sizes = obs_bundles.sum(1)
    print(f"\nContext: {n_obs} firms, {n_dest} destinations, {n_years} years, "
          f"{n_items} items = (dest, year) pairs")
    print(f"  Entry years: {YEAR_INIT}-{YEAR_END}, discount: {discount_factor}")
    print(f"  Bundle sizes: mean={sizes.mean():.1f}, "
          f"median={np.median(sizes):.0f}, max={sizes.max()}")

    return {
        "n_obs": n_obs, "n_dest": n_dest, "n_years": n_years, "n_items": n_items,
        "firms": firms, "destinations": destinations, "years": years, "home": home,
        "expected_revenue": exp_rev,
        "obs_bundles": obs_bundles,
        "constraint_mask": constraint_mask,
        "discount_weights": discount_weights,
        "entry_discount_weights": entry_discount_weights,
        "pairwise_ti": pairwise_ti,
        "pairwise_tv": pairwise_tv,
        "dest_tv": dest_tv,
        "home_to_dest": home_to_dest,
    }


def _zero_diag(M):
    out = M.copy()
    np.fill_diagonal(out, 0.0)
    return out


def build_input_data(ctx):
    rev = ctx["expected_revenue"] / ctx["expected_revenue"].std()
    dist_home = ctx["home_to_dest"]["dist"] / ctx["home_to_dest"]["dist"].std()

    dist_raw = ctx["pairwise_ti"]["dist"]
    dist_scaled = dist_raw / dist_raw[dist_raw > 0].std()
    proximity = _zero_diag(np.exp(-dist_scaled))

    obs_3d = ctx["obs_bundles"].reshape(ctx["n_obs"], ctx["n_years"], ctx["n_dest"])
    capacity = obs_3d.any(axis=1).sum(axis=1)

    return {
        "id_data": {
            "obs_bundles":      ctx["obs_bundles"],
            "constraint_mask":  ctx["constraint_mask"],
            "discount_weights": ctx["discount_weights"],
            "entry_discount_weights": ctx["entry_discount_weights"],
            "capacity":         capacity,
            "modular_3d":       rev[:, :, :, None],
        },
        "item_data": {
            "n_dest":        ctx["n_dest"],
            "n_years":       ctx["n_years"],
            "modular_1d":    dist_home[:, None],
            "entry_1d":      np.ones((ctx["n_dest"], 1)),
            "quadratic_2d":  np.stack([
                proximity,
                (dist_home[:, None] + dist_home[None, :]) / 2 * proximity,
            ], axis=-1),
            "consec_1d":     dist_home[:, None],
        },
    }


# ── Run pipeline ─────────────────────────────────────────────────────

def main(country="MEX", keep_top=50, discount_factor=0.95, n_sample=None):
    data_dir = Path(__file__).resolve().parent / "datasets"
    dataframe = filter_dataframe(keep_top, load_raw_data(country))
    revenue = build_revenue(dataframe)
    expected_revenue = compute_exp_rev(revenue)
    pairwise, dest_feats, home_to_dest, destinations = \
        build_all_gravity_features(dataframe, data_dir, home=country)
    ctx = build_context(dataframe, expected_revenue, pairwise, dest_feats,
                        home_to_dest, destinations, country, discount_factor,
                        n_sample=n_sample)
    return ctx


if __name__ == "__main__":
    ctx = main()
    input_data = build_input_data(ctx)
    for src in ["id_data", "item_data"]:
        print(f"\n{src}:")
        for k, v in input_data[src].items():
            shape = v.shape if hasattr(v, 'shape') else v
            print(f"  {k:<20} {shape}")
