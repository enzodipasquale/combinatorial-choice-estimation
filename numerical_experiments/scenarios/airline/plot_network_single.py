"""Plot a single airline's route network (used in slides)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import generate_data
from plot_networks import plot_airline_network


def main(airline_rank="median"):
    """airline_rank: 'min', 'median', 'max', or an integer firm index."""
    with open(Path(__file__).resolve().parent / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    theta_star, obs_bundles, dgp_data, _ = generate_data(
        cfg["dgp"], cfg["healthy_dgp"], cfg["seeds"], verbose=False)
    C = cfg["dgp"]["C"]
    N = cfg["dgp"]["N"]

    locations = dgp_data["locations"]
    populations = dgp_data["populations"]
    hubs = dgp_data["hubs"]
    origin_of = dgp_data["origin_of"]
    dest_of = dgp_data["dest_of"]
    M = dgp_data["M"]

    sizes = obs_bundles.sum(axis=1)
    order = np.argsort(sizes)
    if airline_rank == "min":
        idx = int(order[0])
    elif airline_rank == "max":
        idx = int(order[-1])
    elif airline_rank == "median":
        idx = int(order[N // 2])
    else:
        idx = int(airline_rank)

    fig, ax = plt.subplots(figsize=(7, 7))
    plot_airline_network(ax, locations, populations, hubs[idx],
                         obs_bundles[idx], origin_of, dest_of, C,
                         idx, edges_color="#1f77b4")
    ax.set_title("")  # no caption for slides

    out_path = Path(__file__).resolve().parent / "network_single_seed42.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved to {out_path}")
    plt.close()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rank", default="median",
                   help="'min', 'median', 'max', or firm index")
    args = p.parse_args()
    main(args.rank)
