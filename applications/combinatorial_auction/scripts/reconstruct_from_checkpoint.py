"""Reconstruct point_estimate/result.json from a bootstrap checkpoint.

combest writes `master.lp` / `master.sol` under
    results/<spec>/bootstrap/checkpoints/.../point_estimate/
during bootstrap. If an HPC run finished the point-estimate phase but
didn't ship back result.json (e.g. the job was interrupted before
bootstrap completed), this script rebuilds result.json directly from
the Gurobi solution file so downstream analysis (post-estimation,
slides, counterfactuals) can treat the spec identically to a locally
computed run.

Usage:
    python -m applications.combinatorial_auction.scripts.reconstruct_from_checkpoint SPEC
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

APP_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = APP_ROOT.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from applications.combinatorial_auction.data.prepare import prepare
from applications.combinatorial_auction.scripts.second_stage.compute import _xbar


_OBJ_RE = re.compile(r"Objective value\s*=\s*(\S+)")
_VAR_RE = re.compile(r"^(\w+)\[(\d+)\]\s+(\S+)\s*$")


def _parse_sol(path: Path):
    """Return (objective, {var_prefix: np.ndarray})."""
    obj = None
    buckets: dict[str, dict[int, float]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            m = _OBJ_RE.search(line)
            if m:
                obj = float(m.group(1))
            continue
        m = _VAR_RE.match(line)
        if not m:
            continue
        name, idx, val = m.group(1), int(m.group(2)), float(m.group(3))
        buckets.setdefault(name, {})[idx] = val
    arrays = {}
    for name, d in buckets.items():
        n = max(d) + 1
        a = np.zeros(n)
        for k, v in d.items():
            a[k] = v
        arrays[name] = a
    return obj, arrays


def reconstruct(spec: str) -> Path:
    cfg_path = APP_ROOT / "configs" / f"{spec}.yaml"
    config = yaml.safe_load(open(cfg_path))
    app = config["application"]

    sol_path = (APP_ROOT / "results" / spec / "bootstrap" / "checkpoints"
                / "checkpoints" / "point_estimate" / "master.sol")
    if not sol_path.exists():
        # Allow a single-checkpoints nesting too.
        alt = (APP_ROOT / "results" / spec / "bootstrap"
               / "checkpoints" / "point_estimate" / "master.sol")
        if alt.exists():
            sol_path = alt
        else:
            raise FileNotFoundError(f"no master.sol under {sol_path.parent} or {alt.parent}")

    objective, arrays = _parse_sol(sol_path)
    if "parameter" not in arrays or "utility" not in arrays:
        raise ValueError(f"{sol_path} missing parameter/utility vars; found {list(arrays)}")
    theta_hat = arrays["parameter"]
    u_hat = arrays["utility"]

    # Rebuild meta + xbar from the config so downstream code behaves
    # identically to a fresh estimate.py run.
    input_data, meta = prepare(
        modular_regressors      = app.get("modular_regressors", []),
        quadratic_regressors    = app.get("quadratic_regressors", []),
        quadratic_id_regressors = app.get("quadratic_id_regressors", []),
        winners_only            = app.get("winners_only", False),
        capacity_source         = app.get("capacity_source", "initial"),
        upper_triangular_quadratic = app.get("upper_triangular_quadratic", False),
    )
    n_obs   = meta["n_obs"]
    n_items = meta["n_items"]
    n_cov   = meta["n_covariates"]

    n_sim_cfg = config["dimensions"].get("n_simulations")
    if u_hat.size % n_obs != 0:
        raise ValueError(f"utility length {u_hat.size} not divisible by n_obs={n_obs}")
    n_sim_from_u = u_hat.size // n_obs
    if n_sim_cfg is not None and n_sim_cfg != n_sim_from_u:
        print(f"  warning: config n_simulations={n_sim_cfg} but "
              f"u_hat implies {n_sim_from_u}; using u_hat value")
    n_sim = n_sim_from_u

    if theta_hat.size != n_cov:
        raise ValueError(f"theta length {theta_hat.size} ≠ meta n_covariates={n_cov} "
                         f"(config/covariates mismatch)")

    b_obs = input_data["id_data"]["obs_bundles"].astype(float)
    xbar = _xbar(input_data, b_obs)

    # Shape the config block the way estimate.py writes it.
    config["dimensions"].update(
        n_obs=n_obs, n_items=n_items,
        n_covariates=n_cov, n_simulations=n_sim,
        covariate_names=meta["covariate_names"],
    )
    app.update(n_id_mod=meta["n_id_mod"], n_item_mod=meta["n_item_mod"],
               n_id_quad=meta["n_id_quad"], n_item_quad=meta["n_item_quad"])

    payload = {
        "theta_hat": theta_hat.tolist(),
        "u_hat":     u_hat.tolist(),
        "xbar":      xbar.tolist(),
        "n_obs":     n_obs,
        "n_simulations": n_sim,
        "converged": True,          # checkpoint is only written on convergence
        "objective": objective,
        "iterations": None,         # unknown from sol file
        "config":    config,
        "meta":      meta,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "reconstructed_from": str(sol_path.relative_to(REPO_ROOT)),
    }

    out_dir = APP_ROOT / "results" / spec / "point_estimate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.json"
    json.dump(payload, open(out_path, "w"), indent=2)

    print(f"  spec:        {spec}")
    print(f"  sol file:    {sol_path.relative_to(REPO_ROOT)}")
    print(f"  theta:       {theta_hat.size} params  "
          f"(mod={meta['n_id_mod']+meta['n_item_mod']}, "
          f"FE={meta['n_items']}, "
          f"qid={meta['n_id_quad']}, q={meta['n_item_quad']})")
    print(f"  u_hat:       {u_hat.size} = {n_obs} × {n_sim}")
    print(f"  objective:   {objective}")
    print(f"  wrote:       {out_path.relative_to(REPO_ROOT)}")
    return out_path


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", help="spec stem, e.g. pop_scaling_large_2")
    reconstruct(ap.parse_args().spec)
