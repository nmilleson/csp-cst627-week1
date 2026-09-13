"""
Model-wise double descent replication on a synthetic modular-arithmetic task.

Task: (a, b) -> (a + b) mod p, p-way classification. Inputs are encoded as
[cos, sin](2*pi*a/p) concat [cos, sin](2*pi*b/p) rather than one-hot(a),
one-hot(b): one-hot inputs are mutually orthogonal for every distinct pair,
so an MLP can only generalize by discovering the addition structure from
scratch during training -- the "grokking" phenomenon, where generalization
is a sudden phase transition that can take orders of magnitude longer than
interpolation. The trig encoding exposes the additive structure directly
(via the angle-sum identity), so a shallow ReLU net can fit the clean rule
through ordinary gradient descent, letting a WIDTH-driven (model-wise, not
epoch-wise) double descent curve show up within a normal epoch budget.

A configurable fraction of TRAINING labels are corrupted to a uniform random
class (test labels stay clean) -- this label noise is what makes the
interpolation-threshold penalty visible.

Sweeps hidden width on a one-layer MLP (Linear -> ReLU -> Linear) across
several seeds, training each run to convergence (loss plateau or floor, or
an epoch cap), and writes width/seed/epochs/train_error/test_error to a CSV.
"""

import csv
import math
import torch
import torch.nn as nn

# --- dataset ---
P = 97                  # modulus / number of classes
NOISE_RATIO = 0.4       # fraction of TRAIN labels replaced with a random class
TRAIN_FRACTION = 0.42   # fraction of the p*p domain used for training (matches
                         # the ~150/361 coverage from the original p=19 run, so
                         # domain coverage stays comparable as p changes)
DATA_SEED = 123         # fixed so the noisy dataset itself doesn't vary across runs

# --- sweep ---
WIDTHS = [4, 8, 16, 32, 64, 128, 256, 512]
SEEDS = [0, 1, 2]
MAX_EPOCHS = 20000
PATIENCE = 500          # epochs of no improvement before declaring a plateau
MIN_DELTA = 1e-6
LOSS_FLOOR = 1e-4       # "converged" once cross-entropy loss drops below this
LR = 1e-3


def make_modular_dataset(p=P, noise_ratio=NOISE_RATIO, train_frac=TRAIN_FRACTION, seed=DATA_SEED):
    g = torch.Generator().manual_seed(seed)

    a_all = torch.arange(p * p) // p
    b_all = torch.arange(p * p) % p
    labels_all = (a_all + b_all) % p

    train_size = round(train_frac * p * p)
    perm = torch.randperm(p * p, generator=g)
    train_idx = perm[:train_size]
    test_idx = perm[train_size:]

    def trig_pairs(a, b, p):
        angle_a = 2 * math.pi * a.float() / p
        angle_b = 2 * math.pi * b.float() / p
        return torch.stack(
            [torch.cos(angle_a), torch.sin(angle_a), torch.cos(angle_b), torch.sin(angle_b)],
            dim=1,
        )

    X_train = trig_pairs(a_all[train_idx], b_all[train_idx], p)
    X_test = trig_pairs(a_all[test_idx], b_all[test_idx], p)
    y_train = labels_all[train_idx].clone()
    y_test = labels_all[test_idx].clone()

    noise_mask = torch.rand(train_size, generator=g) < noise_ratio
    random_labels = torch.randint(0, p, (train_size,), generator=g)
    y_train[noise_mask] = random_labels[noise_mask]

    return X_train, y_train, X_test, y_test


class OneLayerMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def train_one(width, seed, X_train, y_train, X_test, y_test, p,
              max_epochs=MAX_EPOCHS, lr=LR, device="cpu"):
    torch.manual_seed(seed)
    model = OneLayerMLP(X_train.shape[1], width, p).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)  # no weight decay: no regularization
    loss_fn = nn.CrossEntropyLoss()

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    best_loss = float("inf")
    plateau_count = 0
    epochs_used = max_epochs

    for epoch in range(1, max_epochs + 1):
        model.train()
        opt.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

        loss_val = loss.item()

        # "Converged" = loss driven near zero, or stopped improving -- NOT merely
        # 0/1 train error hitting zero (that's just first interpolation; the
        # width-dependent generalization effect we're after emerges from
        # continued training past that point).
        if loss_val < LOSS_FLOOR:
            epochs_used = epoch
            break

        if loss_val < best_loss - MIN_DELTA:
            best_loss = loss_val
            plateau_count = 0
        else:
            plateau_count += 1
            if plateau_count >= PATIENCE:
                epochs_used = epoch
                break

    model.eval()
    with torch.no_grad():
        train_err = (model(X_train).argmax(1) != y_train).float().mean().item()
        test_err = (model(X_test).argmax(1) != y_test).float().mean().item()

    return epochs_used, train_err, test_err


def run_sweep(out_csv="double_descent_results.csv"):
    X_train, y_train, X_test, y_test = make_modular_dataset()
    print(f"p={P}, train_size={X_train.shape[0]}, test_size={X_test.shape[0]}, "
          f"noise_ratio={NOISE_RATIO}\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for width in WIDTHS:
        for seed in SEEDS:
            epochs_used, train_err, test_err = train_one(
                width, seed, X_train, y_train, X_test, y_test, P, device=device
            )
            rows.append((width, seed, epochs_used, train_err, test_err))
            print(f"width={width:4d} seed={seed} epochs={epochs_used:4d} "
                  f"train_err={train_err:.4f} test_err={test_err:.4f}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["width", "seed", "num_epochs", "train_error", "test_error"])
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    run_sweep()
