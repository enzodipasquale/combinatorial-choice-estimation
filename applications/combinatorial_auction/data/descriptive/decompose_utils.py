"""Raw welfare decomposition in utility units (no 2SLS).
Usage: python -m applications.combinatorial_auction.data.descriptive.decompose_utils SPEC
"""
import json, sys, yaml, numpy as np
from pathlib import Path

from ..prepare import prepare
from ...scripts.second_stage.compute import _xbar

APP = Path(__file__).resolve().parent.parent.parent


def run(spec):
    r = json.load(open(APP / "results" / spec / "point_estimate" / "result.json"))
    m, n_sim = r["meta"], r["config"]["dimensions"]["n_simulations"]
    th, u = np.asarray(r["theta_hat"]), np.asarray(r["u_hat"])
    if r.get("xbar") is None:
        app = yaml.safe_load(open(APP / "configs" / f"{spec}.yaml"))["application"]
        d, _ = prepare(app.get("modular_regressors", []),
                       app.get("quadratic_regressors", []),
                       app.get("quadratic_id_regressors", []),
                       winners_only=app.get("winners_only", False),
                       capacity_source=app.get("capacity_source", "initial"),
                       upper_triangular_quadratic=app.get("upper_triangular_quadratic", False))
        xb = _xbar(d, d["id_data"]["obs_bundles"].astype(float))
    else:
        xb = np.asarray(r["xbar"])
    names = {int(k): v for k, v in m["covariate_names"].items()}
    fe = slice(m["n_id_mod"], m["n_id_mod"] + m["n_btas"])
    contrib = th * xb
    surplus = u.reshape(m["n_obs"], n_sim).mean(1).sum()
    delta   = -th[fe].sum()                          # Σδ = -Σθ_fe
    entropy = surplus - sum(contrib[i] for i in names) - delta
    print(f"{spec}   surplus = {surplus:+.4f}   (utils)")
    for i, n in sorted(names.items()): print(f"  {n:<24} {contrib[i]:+.4f}")
    print(f"  {'Σδ':<24} {delta:+.4f}")
    print(f"  {'entropy':<24} {entropy:+.4f}")


if __name__ == "__main__":
    run(sys.argv[1])
