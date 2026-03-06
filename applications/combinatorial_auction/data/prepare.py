import numpy as np
from .registries import MODULAR, QUADRATIC, QUADRATIC_ID, WEIGHT_ROUNDING_TICK
from .loaders import (load_bta_data, build_context, load_aggregation_matrix,
                      continental_mta_nums, load_ab_data)

AB_ELIG_BANDWIDTH = 30.0


def prepare(dataset, modular_regressors, quadratic_regressors,
            quadratic_id_regressors=(), item_modular="fe",
            separate_ab_quadratics=False):
    raw = load_bta_data()
    ctx = build_context(raw)
    fn = {"c_block": _c_block, "ab_block": _ab_block, "joint": _joint}[dataset]
    kwargs = dict(separate_ab_quadratics=separate_ab_quadratics) if dataset == "joint" else {}
    input_data, meta = fn(raw, ctx, modular_regressors, quadratic_regressors, quadratic_id_regressors, **kwargs)
    if item_modular == "price":
        _apply_price(input_data, raw, meta, dataset)

    # dimensions
    n_obs, n_items = input_data["id_data"]["obs_bundles"].shape
    n_id_mod = input_data["id_data"]["modular"].shape[-1]
    n_item_mod = input_data["item_data"]["modular"].shape[-1]
    n_id_quad = input_data["id_data"]["quadratic"].shape[-1] if "quadratic" in input_data["id_data"] else 0
    n_item_quad = input_data["item_data"]["quadratic"].shape[-1]
    meta.update(
        n_obs=n_obs,
        n_items=n_items,
        n_id_mod=n_id_mod,
        n_item_mod=n_item_mod,
        n_id_quad=n_id_quad,
        n_item_quad=n_item_quad,
        n_covariates=n_id_mod + n_item_mod + n_id_quad + n_item_quad,
    )

    # covariate names (only named covariates, FEs are unnamed)
    names = {i: n for i, n in enumerate(modular_regressors)}
    off = n_id_mod
    if item_modular == "price":
        names[off] = "price"
    off += n_item_mod
    sep = separate_ab_quadratics and dataset == "joint"
    for i, n in enumerate(quadratic_id_regressors):
        names[off + i] = f"{n}_c" if sep else n
    if sep:
        for i, n in enumerate(quadratic_id_regressors):
            names[off + len(quadratic_id_regressors) + i] = f"{n}_ab"
    off += n_id_quad
    for i, n in enumerate(quadratic_regressors):
        names[off + i] = f"{n}_c" if sep else n
    if sep:
        for i, n in enumerate(quadratic_regressors):
            names[off + len(quadratic_regressors) + i] = f"{n}_ab"
    meta["covariate_names"] = names

    return input_data, meta


def _build_features(registry, names, ctx):
    return np.stack([registry[name](ctx) for name in names], axis=-1).astype(np.float64)


def _mta_regressors(ctx, A, mta_nums, winners, elig_norm):
    mta_feats = {k: A @ ctx[k] for k in ["pop", "percapin", "hhinc35k", "density", "imwl"]}
    regs = {}
    for k, v in mta_feats.items():
        regs[f"elig_{k}"] = (lambda f=v: elig_norm[:, None] * f[None, :])
    mta_avg = winners.groupby("mta_num")["price"].mean()
    price_mta = np.array([mta_avg.get(m, 0) for m in mta_nums]) / 1e9
    regs["elig_price"] = lambda: elig_norm[:, None] * price_mta[None, :]
    return regs


def _ab_obs_bundles(winners, bidders, mta_idx):
    n_obs, n_mtas = len(bidders), len(mta_idx)
    acct_to_bidder = {a: i for i, a in enumerate(bidders["fcc_acct"])}
    obs = np.zeros((n_obs, n_mtas), dtype=int)
    for _, row in winners.iterrows():
        bi = acct_to_bidder.get(row["fcc_acct"])
        mi = mta_idx.get(row["mta_num"])
        if bi is not None and mi is not None:
            obs[bi, mi] = 1
    return obs


def _aggregate_quadratics(ctx, quad_names, A):
    bta_quad = _build_features(QUADRATIC, quad_names, ctx)
    n_qfeat = bta_quad.shape[-1]
    Q_mta = np.stack([A @ bta_quad[:, :, k] @ A.T for k in range(n_qfeat)], axis=-1)
    return bta_quad, Q_mta


# ── C-block ──────────────────────────────────────────────────────────

def _c_block(raw, ctx, mod_names, quad_names, qid_names):
    mod = _build_features(MODULAR, mod_names, ctx)
    quad = _build_features(QUADRATIC, quad_names, ctx)
    qid = _build_features(QUADRATIC_ID, qid_names, ctx) if qid_names else None
    n_items = len(raw["bta_data"])

    id_data = {
        "modular": mod,
        "capacity": ctx["capacity"],
        "obs_bundles": ctx["c_obs_bundles"],
    }
    if qid is not None:
        id_data["quadratic"] = qid

    item_data = {
        "modular": -np.eye(n_items, dtype=np.float64),
        "quadratic": quad,
        "weight": ctx["weight"],
    }

    meta = {
        "n_btas": n_items,
        "raw": raw,
    }
    return {"id_data": id_data, "item_data": item_data}, meta


# ── AB-block ─────────────────────────────────────────────────────────

def _ab_block(raw, ctx, mod_names, quad_names, qid_names):
    btas = raw["bta_data"]["bta"].values.astype(int)
    A = load_aggregation_matrix(btas)
    n_mtas = A.shape[0]
    mta_nums = continental_mta_nums(btas)
    mta_idx = {m: i for i, m in enumerate(mta_nums)}

    winners, bidders = load_ab_data()
    winners = winners[winners["mta_num"].isin(mta_nums)].reset_index(drop=True)
    ab_elig = bidders["eligibility"].to_numpy().astype(float)
    elig_norm = ab_elig / (ctx["pop_sum"] * AB_ELIG_BANDWIDTH)

    obs = _ab_obs_bundles(winners, bidders, mta_idx)
    mta_reg = _mta_regressors(ctx, A, mta_nums, winners, elig_norm)
    mod = np.stack([mta_reg[name]() for name in mod_names], axis=-1).astype(np.float64)

    _, Q_mta = _aggregate_quadratics(ctx, quad_names, A)

    # quadratic_id
    qid = None
    if qid_names:
        qi2q = {name: i for i, name in enumerate(quad_names)}
        layers = [elig_norm[:, None, None] * Q_mta[None, :, :, qi2q[n.replace("elig_", "")]]
                  for n in qid_names]
        qid = np.stack(layers, axis=-1).astype(np.float64)

    id_data = {
        "modular": mod,
        "capacity": np.round(ab_elig / AB_ELIG_BANDWIDTH // WEIGHT_ROUNDING_TICK).astype(int),
        "obs_bundles": obs,
    }
    if qid is not None:
        id_data["quadratic"] = qid

    item_data = {
        "modular": -np.eye(n_mtas, dtype=np.float64),
        "quadratic": Q_mta,
        "weight": (A @ ctx["weight"].astype(np.float64)).astype(int),
    }

    meta = {
        "n_btas": len(raw["bta_data"]),
        "n_mtas": n_mtas,
        "A": A,
        "continental_mta_nums": mta_nums,
        "raw": raw,
    }
    return {"id_data": id_data, "item_data": item_data}, meta


# ── Joint ────────────────────────────────────────────────────────────

def _joint(raw, ctx, mod_names, quad_names, qid_names, separate_ab_quadratics=False):
    btas = raw["bta_data"]["bta"].values.astype(int)
    A = load_aggregation_matrix(btas)
    n_btas, n_mtas = A.shape[1], A.shape[0]
    n_items = n_btas + n_mtas
    mta_nums = continental_mta_nums(btas)
    mta_idx = {m: i for i, m in enumerate(mta_nums)}

    # C-block
    c_mod = _build_features(MODULAR, mod_names, ctx)
    c_quad, Q_mta = _aggregate_quadratics(ctx, quad_names, A)
    n_obs_c = c_mod.shape[0]

    c_qid = None
    if qid_names:
        c_qid = _build_features(QUADRATIC_ID, qid_names, ctx)

    # AB
    winners, bidders = load_ab_data()
    winners = winners[winners["mta_num"].isin(mta_nums)].reset_index(drop=True)
    n_obs_ab = len(bidders)
    ab_elig = bidders["eligibility"].to_numpy().astype(float)
    elig_norm = ab_elig / (ctx["pop_sum"] * AB_ELIG_BANDWIDTH)
    ab_capacity = np.round(ab_elig / AB_ELIG_BANDWIDTH // WEIGHT_ROUNDING_TICK).astype(int)
    mta_weight = (A @ ctx["weight"].astype(np.float64)).astype(int)

    ab_obs = _ab_obs_bundles(winners, bidders, mta_idx)
    mta_reg = _mta_regressors(ctx, A, mta_nums, winners, elig_norm)

    ab_mod = np.stack([mta_reg[n]() for n in mod_names], axis=-1).astype(np.float64)

    n_qfeat = c_quad.shape[-1]

    # AB quadratic_id
    ab_qid = None
    if qid_names:
        qi2q = {name: i for i, name in enumerate(quad_names)}
        layers = [elig_norm[:, None, None] * Q_mta[None, :, :, qi2q[n.replace("elig_", "")]]
                  for n in qid_names]
        ab_qid = np.stack(layers, axis=-1).astype(np.float64)

    # block-diagonal stacking
    n_obs = n_obs_c + n_obs_ab

    id_mod = np.zeros((n_obs, n_items, c_mod.shape[-1]), dtype=np.float64)
    id_mod[:n_obs_c, :n_btas] = c_mod
    id_mod[n_obs_c:, n_btas:] = ab_mod

    obs_bundles = np.zeros((n_obs, n_items), dtype=int)
    obs_bundles[:n_obs_c, :n_btas] = ctx["c_obs_bundles"]
    obs_bundles[n_obs_c:, n_btas:] = ab_obs

    item_mask = np.zeros((n_obs, n_items), dtype=np.int32)
    item_mask[:n_obs_c, :n_btas] = 1
    item_mask[n_obs_c:, n_btas:] = 1

    n_qfeat_total = 2 * n_qfeat if separate_ab_quadratics else n_qfeat
    item_quad = np.zeros((n_items, n_items, n_qfeat_total), dtype=np.float64)
    item_quad[:n_btas, :n_btas, :n_qfeat] = c_quad
    item_quad[n_btas:, n_btas:, -n_qfeat:] = Q_mta

    id_data = {
        "modular": id_mod,
        "obs_bundles": obs_bundles,
        "item_mask": item_mask,
        "capacity": np.concatenate([ctx["capacity"], ab_capacity]),
    }
    if c_qid is not None:
        n_qid = c_qid.shape[-1]
        n_qid_total = 2 * n_qid if separate_ab_quadratics else n_qid
        id_quad = np.zeros((n_obs, n_items, n_items, n_qid_total), dtype=np.float64)
        id_quad[:n_obs_c, :n_btas, :n_btas, :n_qid] = c_qid
        if ab_qid is not None:
            id_quad[n_obs_c:, n_btas:, n_btas:, -n_qid:] = ab_qid
        id_data["quadratic"] = id_quad

    item_data = {
        "modular": -np.eye(n_items, dtype=np.float64),
        "quadratic": item_quad,
        "weight": np.concatenate([ctx["weight"], mta_weight]),
    }

    meta = {
        "n_btas": n_btas,
        "n_mtas": n_mtas,
        "n_obs_c": n_obs_c,
        "n_obs_ab": n_obs_ab,
        "A": A,
        "continental_mta_nums": mta_nums,
        "raw": raw,
    }
    return {"id_data": id_data, "item_data": item_data}, meta


# ── Price replacement ────────────────────────────────────────────────

def _apply_price(input_data, raw, meta, dataset):
    price_bta = raw["bta_data"]["bid"].to_numpy().astype(float) / 1e9

    if dataset == "c_block":
        input_data["item_data"]["modular"] = -price_bta[:, None]
    else:
        mta_nums = meta["continental_mta_nums"]
        winners, _ = load_ab_data()
        winners = winners[winners["mta_num"].isin(mta_nums)]
        mta_avg = winners.groupby("mta_num")["price"].mean()
        price_mta = np.array([mta_avg.get(m, 0) for m in mta_nums]) / 1e9

        if dataset == "ab_block":
            input_data["item_data"]["modular"] = -price_mta[:, None]
        else:
            input_data["item_data"]["modular"] = -np.concatenate([price_bta, price_mta])[:, None]
