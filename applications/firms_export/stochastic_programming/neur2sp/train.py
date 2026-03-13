"""Train a ReLU neural network for the Neur2SP value-function approximation.

Loads the dataset produced by ``generate_data.py``, trains a small ReLU net
to predict the expected period-2 value from (b_1, theta), and saves the
weights with input/output normalization folded in so the MIP can work with
raw (unnormalized) inputs directly.

Usage:
    python -m neur2sp.train                  (from stochastic_programming/)
    python -m neur2sp.train --hidden 64 --epochs 800
"""
import argparse
import numpy as np
import torch
import torch.nn as nn


# ── Network definition ──────────────────────────────────────────────
class ValueNet(nn.Module):
    """Feed-forward ReLU network with a scalar output (no output activation)."""

    def __init__(self, input_dim, hidden_dim=32, n_hidden=2):
        super().__init__()
        layers = []
        d_in = input_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d_in, hidden_dim), nn.ReLU()]
            d_in = hidden_dim
        layers.append(nn.Linear(d_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Weight extraction with folded normalization ─────────────────────
def extract_weights(model, x_mean, x_std, y_mean, y_std):
    """Return (weights, biases) lists with normalization folded in.

    After folding, the network computes  y = f(x)  directly on the
    original (unnormalized) scale, so the MIP needs no extra scaling.
    """
    params = list(model.net.parameters())
    n_layers = len(params) // 2
    x_m = x_mean.numpy()
    x_s = x_std.numpy()
    y_m = float(y_mean)
    y_s = float(y_std)

    weights, biases = [], []
    for i in range(n_layers):
        W = params[2 * i].detach().numpy().copy()
        b = params[2 * i + 1].detach().numpy().copy()

        if i == 0:
            # fold input normalisation:  z = (x - mu) / sigma
            # h = W z + b  =  (W / sigma) x  + (b - (W / sigma) mu)
            W = W / x_s[None, :]          # broadcast over rows
            b = b - W @ x_m

        if i == n_layers - 1:
            # fold output de-normalisation:  y = y_norm * y_std + y_mean
            W = W * y_s
            b = b * y_s + y_m

        weights.append(W)
        biases.append(b)
    return weights, biases


# ── Training loop ───────────────────────────────────────────────────
def train(data_path, save_path, hidden_dim=32, n_hidden=2,
          lr=1e-3, epochs=500, batch_size=256, val_frac=0.1, seed=42):
    data = np.load(data_path, allow_pickle=True)
    X = torch.tensor(data["inputs"], dtype=torch.float32)
    y = torch.tensor(data["labels"], dtype=torch.float32)
    M = int(data["M"])
    n_rev = int(data["n_rev"])

    # train / validation split
    n = len(X)
    perm = np.random.default_rng(seed).permutation(n)
    n_val = max(int(n * val_frac), 1)
    X_val, y_val = X[perm[:n_val]], y[perm[:n_val]]
    X_tr, y_tr = X[perm[n_val:]], y[perm[n_val:]]

    # normalisation stats (from training set)
    x_mean, x_std = X_tr.mean(0), X_tr.std(0)
    x_std[x_std < 1e-8] = 1.0
    y_mean, y_std = y_tr.mean(), y_tr.std()
    if y_std < 1e-8:
        y_std = torch.tensor(1.0)

    input_dim = X.shape[1]
    net = ValueNet(input_dim, hidden_dim, n_hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None

    for ep in range(epochs):
        net.train()
        idx = torch.randperm(len(X_tr))
        ep_loss, nb = 0.0, 0
        for i in range(0, len(X_tr), batch_size):
            bx = (X_tr[idx[i:i + batch_size]] - x_mean) / x_std
            by = (y_tr[idx[i:i + batch_size]] - y_mean) / y_std
            loss = loss_fn(net(bx), by)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1

        net.eval()
        with torch.no_grad():
            vl = loss_fn(net((X_val - x_mean) / x_std),
                         (y_val - y_mean) / y_std).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in net.state_dict().items()}

        if ep % 50 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:4d}  train={ep_loss/nb:.6f}  val={vl:.6f}")

    net.load_state_dict(best_state)

    # extract MIP-ready weights
    weights, biases = extract_weights(net, x_mean, x_std, y_mean, y_std)

    # R^2 on validation set (original scale)
    net.eval()
    with torch.no_grad():
        pred = net((X_val - x_mean) / x_std) * y_std + y_mean
        ss_res = ((pred - y_val) ** 2).sum().item()
        ss_tot = ((y_val - y_val.mean()) ** 2).sum().item()
        r2 = 1 - ss_res / max(ss_tot, 1e-12)
    print(f"  R^2 (val, original scale) = {r2:.4f}")

    # theta bounds (saved alongside data)
    theta_lb = np.array(
        list(data["theta_bounds_rev"][:1]) * n_rev    # low end per rev dim
        + [float(data["theta_bounds_s"][0])]
        + [float(data["theta_bounds_c"][0])]
    )
    theta_ub = np.array(
        list(data["theta_bounds_rev"][1:2]) * n_rev
        + [float(data["theta_bounds_s"][1])]
        + [float(data["theta_bounds_c"][1])]
    )

    torch.save({
        "weights": weights,
        "biases": biases,
        "hidden_dim": hidden_dim,
        "n_hidden": n_hidden,
        "input_dim": input_dim,
        "M": M,
        "K": int(data["K"]),
        "n_rev": n_rev,
        "beta": float(data["beta"]),
        "theta_lb": theta_lb,
        "theta_ub": theta_ub,
    }, save_path)
    print(f"  Saved model  ->  {save_path}")


# ── CLI ─────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="neur2sp/data.npz")
    ap.add_argument("--out", default="neur2sp/model.pt")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--n_hidden", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(f"Training  (data={args.data}  hidden={args.hidden}x{args.n_hidden}  "
          f"epochs={args.epochs})")
    train(args.data, args.out,
          hidden_dim=args.hidden, n_hidden=args.n_hidden,
          lr=args.lr, epochs=args.epochs, batch_size=args.batch,
          seed=args.seed)


if __name__ == "__main__":
    main()
