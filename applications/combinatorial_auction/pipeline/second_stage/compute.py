"""Per-bootstrap-draw welfare decomposition for a single spec.

Pipeline (per draw b):
    θ = bootstrap_thetas[b],  u = bootstrap_u_hat[b]
    surplus_b   = (u reshaped (n_obs, n_sim), mean over sims, then sum over i)
    contrib[k]  = θ[k] * xbar[k]                        # named covariates only
    δ           = −θ_fe                                  # BTA-level intercepts
    iv          = second_stage(δ, price, raw, use_blp)  # α₀, α₁, demand_controls
    ξ           = δ − α₀ + α₁·p − Z'γ
    entropy_b   = surplus_b − Σ contrib[named] − δ.sum()
    δ = α₀_part + price_part + controls_part + ξ_part   (identity, reported)

The single source of truth for features is data.prepare — xbar is assembled
from the arrays it already produced, not by re-invoking the registry.
"""
import json, yaml
from pathlib import Path
import numpy as np

from ...data.prepare import prepare
from ...data.loaders import load_raw
from .iv import simple_instruments, second_stage as run_second_stage, compute_xi

APP_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS  = APP_ROOT / "results"
CONFIGS  = APP_ROOT / "configs"


def _xbar_from_input(input_data, b_obs):
    """Assemble x̄ from the arrays produced by prepare(), matching the
    combest convention used by main: FE block = b_obs.sum(axis=0)."""
    id_mod    = input_data["id_data"]["modular"]
    quad_item = input_data["item_data"]["quadratic"]
    quad_id   = input_data["id_data"].get("quadratic")

    n_id_mod    = id_mod.shape[-1]
    n_btas      = b_obs.shape[1]
    n_id_quad   = quad_id.shape[-1]  if quad_id is not None else 0
    n_item_quad = quad_item.shape[-1]
    n_cov = n_id_mod + n_btas + n_id_quad + n_item_quad

    xbar = np.zeros(n_cov)
    # modular
    for k in range(n_id_mod):
        xbar[k] = (b_obs * id_mod[:, :, k]).sum()
    # FE
    xbar[n_id_mod:n_id_mod + n_btas] = b_obs.sum(axis=0)
    # quadratic_id: Σ_i b_i @ Q_i @ b_i
    off = n_id_mod + n_btas
    for k in range(n_id_quad):
        xbar[off + k] = np.einsum("ij,ijk,ik->", b_obs, quad_id[:, :, :, k], b_obs) \
            if False else sum(b_obs[i] @ quad_id[i, :, :, k] @ b_obs[i]
                              for i in range(b_obs.shape[0]))
    off += n_id_quad
    # quadratic: b @ Q @ b, same Q for each i
    for k in range(n_item_quad):
        xbar[off + k] = sum(b_obs[i] @ quad_item[:, :, k] @ b_obs[i]
                            for i in range(b_obs.shape[0]))
    return xbar


def decompose(spec_stem, *, configs_dir=CONFIGS, results_dir=RESULTS):
    """Return (rows, named_order) for a single spec."""
    cfg = yaml.safe_load(open(configs_dir / f"{spec_stem}.yaml"))
    app = cfg["application"]

    input_data, meta = prepare(
        modular_regressors       = app.get("modular_regressors", []),
        quadratic_regressors     = app.get("quadratic_regressors", []),
        quadratic_id_regressors  = app.get("quadratic_id_regressors", []),
        winners_only             = app.get("winners_only", False),
        capacity_source          = app.get("capacity_source", "initial"),
    )
    raw = load_raw()
    price = raw["bta_data"]["bid"].to_numpy(dtype=float) / 1e9
    b_obs = input_data["id_data"]["obs_bundles"].astype(float)
    n_obs = b_obs.shape[0]
    n_btas = b_obs.shape[1]

    n_id_mod  = meta["n_id_mod"]
    n_id_quad = meta["n_id_quad"]
    mod_names  = app.get("modular_regressors", [])
    qid_names  = app.get("quadratic_id_regressors", [])
    quad_names = app.get("quadratic_regressors", [])
    named_order = mod_names + qid_names + quad_names
    use_blp = app.get("error_scaling") == "pop"
    pop_threshold = app.get("pop_threshold", 500_000)

    r = json.load(open(results_dir / spec_stem / "bootstrap_result.json"))
    if "xbar" in r:
        xbar = np.array(r["xbar"])
    else:
        xbar = _xbar_from_input(input_data, b_obs)

    boot_thetas = np.asarray(r["bootstrap_thetas"])
    boot_u_hats = np.asarray(r["bootstrap_u_hat"])
    converged   = r.get("converged", [True] * len(boot_thetas))
    n_sim = boot_u_hats.shape[1] // n_obs

    # Cache simple-IV instruments once (used by every draw when use_blp=False).
    simple_cache = None if use_blp else simple_instruments(raw)

    rows = []
    for b in range(len(boot_thetas)):
        if not converged[b]:
            continue
        th = boot_thetas[b]
        u  = boot_u_hats[b]

        surplus = u.reshape(n_obs, n_sim).mean(axis=1).sum()
        delta   = -th[n_id_mod:n_id_mod + n_btas]
        fe_total = delta.sum()

        contrib = th * xbar
        # named-covariate indices: all except FE block
        named_idx = list(range(n_id_mod)) + list(range(n_id_mod + n_btas, len(th)))
        named = {name: float(th[named_idx[i]]) for i, name in enumerate(named_order)}
        contrib_by_name = {name: float(contrib[named_idx[i]])
                           for i, name in enumerate(named_order)}

        iv = run_second_stage(
            delta, price, raw, use_blp=use_blp,
            simple_instruments_cached=simple_cache,
            pop_threshold=pop_threshold,
        )
        a0, a1, dc = iv["a0"], iv["a1"], iv["demand_controls"]
        xi = compute_xi(delta, price, a0, a1, dc, raw["bta_data"])

        controls_part = 0.0
        if dc:
            for v, c in dc.items():
                controls_part += c * raw["bta_data"][v].to_numpy(dtype=float).sum()

        entropy = surplus - sum(contrib_by_name.values()) - fe_total

        rows.append(dict(
            a0=a0, a1=a1, se_a0=iv["se_a0"], se_a1=iv["se_a1"], r2=iv["r2"],
            surplus=surplus, entropy=entropy, fe_total=fe_total,
            a0_part=n_btas * a0, price_part=-a1 * price.sum(),
            controls_part=controls_part, xi_part=xi.sum(),
            **{f"theta_{k}": v for k, v in named.items()},
            **{f"contrib_{k}": v for k, v in contrib_by_name.items()},
        ))
    return rows, named_order


def decompose_all(specs):
    """Run decompose() for multiple specs; skip specs without a bootstrap result."""
    out = {}
    for stem in specs:
        if not (RESULTS / stem / "bootstrap_result.json").exists():
            print(f"[{stem}] no bootstrap_result.json, skip")
            continue
        out[stem] = decompose(stem)
    return out
