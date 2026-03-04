#!/usr/bin/env python3
# Shared results saving/loading for combinatorial auction estimation runs.

import csv, json, os, datetime
import numpy as np
from pathlib import Path

APP_DIR = Path(__file__).parent
OUTPUT_DIR = APP_DIR / "estimation_results"

POINT_ESTIMATE_CSV = "point_estimate_runs.csv"
BOOTSTRAP_CSV = "se_bootstrap_runs.csv"
COUNTERFACTUAL_CSV = "counterfactual_runs.csv"


# ── Helpers ───────────────────────────────────────────────────────────

def _slurm_meta():
    return {k: os.environ.get(v, "") for k, v in
            [("slurm_job_id", "SLURM_JOB_ID"), ("slurm_node", "SLURM_NODELIST"),
             ("slurm_ntasks", "SLURM_NTASKS")]}


def _config_row(config):
    app = config.get("application", {})
    dims = config.get("dimensions", {})
    cst = config.get("constraints", {})
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        **_slurm_meta(),
        "winners_only": app.get("winners_only", False),
        "continental_only": app.get("continental_only", False),
        "rescale_features": app.get("rescale_features", False),
        "error_seed": app.get("error_seed"),
        "modular_regressors": json.dumps(app.get("modular_regressors", [])),
        "quadratic_regressors": json.dumps(app.get("quadratic_regressors", [])),
        "quadratic_id_regressors": json.dumps(app.get("quadratic_id_regressors", [])),
        "mod_bounds": json.dumps(app.get("mod_bounds", {})),
        "quad_bounds": json.dumps(app.get("quad_bounds", {})),
        "quad_id_bounds": json.dumps(app.get("quad_id_bounds", {})),
        "pop_dominates_travel": cst.get("pop_dominates_travel", False),
        "n_simulations": dims.get("n_simulations"),
    }
    return row


def _append_row(csv_path, row):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


# ── Save ──────────────────────────────────────────────────────────────

def save_point_estimate(config, result, n_obs, n_items, n_covariates):
    row = _config_row(config)
    row.update(n_obs=n_obs, n_items=n_items, n_covariates=n_covariates,
               theta_hat=json.dumps(result.theta_hat.tolist()),
               converged=result.converged,
               num_iterations=result.num_iterations,
               final_objective=result.final_objective,
               n_constraints=result.n_constraints,
               total_time=result.total_time)
    _append_row(OUTPUT_DIR / POINT_ESTIMATE_CSV, row)


def save_bootstrap(config, se_result, n_obs, n_items, n_covariates,
                   num_bootstrap, boot_seed):
    row = _config_row(config)
    row.update(n_obs=n_obs, n_items=n_items, n_covariates=n_covariates,
               num_bootstrap=num_bootstrap, bootstrap_seed=boot_seed,
               n_samples=se_result.n_samples, confidence=se_result.confidence,
               mean_se=float(np.mean(se_result.se)),
               max_se=float(np.max(se_result.se)),
               theta_mean=json.dumps(se_result.mean.tolist()),
               se=json.dumps(se_result.se.tolist()),
               ci_lower=json.dumps(se_result.ci_lower.tolist()),
               ci_upper=json.dumps(se_result.ci_upper.tolist()),
               t_stats=json.dumps(se_result.t_stats.tolist()))
    _append_row(OUTPUT_DIR / BOOTSTRAP_CSV, row)


def save_counterfactual(config, est, cf_result, n_mtas):
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "source_type": est.get("source_type", ""),
        "source_timestamp": est.get("timestamp", ""),
        "source_slurm_job_id": est.get("slurm_job_id", ""),
        "quadratic_regressors": json.dumps(est["quadratic_regressors"]),
        "continental_only": est["continental_only"],
        "error_seed": est["error_seed"],
        "n_mtas": n_mtas,
        "theta_hat_mta": json.dumps(cf_result.theta_hat.tolist()),
        "converged": cf_result.converged,
        "num_iterations": cf_result.num_iterations,
        "final_objective": cf_result.final_objective,
        "total_time": cf_result.total_time,
    }
    _append_row(OUTPUT_DIR / COUNTERFACTUAL_CSV, row)


# ── Load ──────────────────────────────────────────────────────────────

def load_result(source="point_estimate", run_idx=-1):
    import pandas as pd

    csv_name = POINT_ESTIMATE_CSV if source == "point_estimate" else BOOTSTRAP_CSV
    row = pd.read_csv(OUTPUT_DIR / csv_name).iloc[run_idx]

    theta_hat = np.array(json.loads(
        row["theta_hat"] if source == "point_estimate" else row["theta_mean"]))

    result = {
        "theta_hat": theta_hat, "source_type": source,
        "timestamp": str(row.get("timestamp", "")),
        "slurm_job_id": str(row.get("slurm_job_id", "")),
        "winners_only": bool(row["winners_only"]),
        "continental_only": bool(row["continental_only"]),
        "rescale_features": bool(row["rescale_features"]),
        "error_seed": int(row["error_seed"]),
        "modular_regressors": json.loads(row["modular_regressors"]),
        "quadratic_regressors": json.loads(row["quadratic_regressors"]),
        "quadratic_id_regressors": json.loads(row["quadratic_id_regressors"]),
        "n_obs": int(row["n_obs"]),
        "n_items": int(row["n_items"]),
        "n_covariates": int(row.get("n_covariates", row.get("n_features", 0))),
        "mod_bounds": json.loads(row["mod_bounds"]),
        "quad_bounds": json.loads(row["quad_bounds"]),
        "quad_id_bounds": json.loads(row["quad_id_bounds"]),
        "pop_dominates_travel": bool(row.get("pop_dominates_travel", False)),
    }

    if source == "point_estimate":
        result.update(converged=bool(row.get("converged", True)),
                      num_iterations=int(row.get("num_iterations", 0)),
                      total_time=float(row.get("total_time", 0)))
    else:
        result.update(
            se=np.array(json.loads(row["se"])),
            ci_lower=np.array(json.loads(row["ci_lower"])),
            ci_upper=np.array(json.loads(row["ci_upper"])),
            t_stats=np.array(json.loads(row["t_stats"])),
            confidence=float(row["confidence"]))

    return result
