"""
Model-wise double descent replication on scikit-learn's digits dataset.

Unlike the synthetic modular-arithmetic task in double_descent.py, real image
data gives an MLP genuine shared structure between examples (similar digits
share pixel patterns), so the network should fit the clean signal quickly via
ordinary gradient descent -- no delayed "grokking"-style transition needed.
That makes the width-driven interpolation-threshold effect (from label noise)
expected to show up cleanly within a modest epoch budget, much like the
classical MNIST/CIFAR double-descent demonstrations.

A configurable fraction of TRAINING labels are corrupted to a uniform random
class (test labels stay clean). Sweeps hidden width on a one-layer MLP
(Linear -> ReLU -> Linear) across several seeds, training each run to
convergence (loss plateau or floor, or an epoch cap), and writes
width/seed/epochs/train_error/test_error to a CSV.
"""

import csv
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# --- dataset ---
NOISE_RATIO = 0.25    # fraction of TRAIN labels replaced with a random class
TRAIN_SIZE = 1000     # out of 1797 total digit images (8x8, 10 classes) --
                       # large enough that a small hidden layer can't memorize
                       # the noisy labels, so the interpolation threshold
                       # lands inside the width sweep instead of below it
DATA_SEED = 123       # fixed so the noisy split doesn't vary across runs

# --- sweep ---
WIDTHS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
SEEDS = [0, 1, 2]
MAX_EPOCHS = 12000
PATIENCE = 500          # epochs of no improvement before declaring a plateau
MIN_DELTA = 1e-6
LOSS_FLOOR = 1e-4       # "converged" once cross-entropy loss drops below this
LR = 1e-3


def make_digits_dataset(noise_ratio=NOISE_RATIO, train_size=TRAIN_SIZE, seed=DATA_SEED):
    data = load_digits()
    X = data.data.astype("float32") / 16.0   # pixel intensities are 0..16
    y = data.target.astype("int64")
    n_classes = int(y.max()) + 1

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=train_size, random_state=seed, stratify=y
    )

    X_train = torch.tensor(X_train)
    X_test = torch.tensor(X_test)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)

    g = torch.Generator().manual_seed(seed)
    noise_mask = torch.rand(len(y_train), generator=g) < noise_ratio
    random_labels = torch.randint(0, n_classes, (len(y_train),), generator=g)
    y_train_noisy = y_train.clone()
    y_train_noisy[noise_mask] = random_labels[noise_mask]

    return X_train, y_train_noisy, X_test, y_test, n_classes


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


def train_one(width, seed, X_train, y_train, X_test, y_test, n_classes,
              max_epochs=MAX_EPOCHS, lr=LR, device="cpu"):
    torch.manual_seed(seed)
    model = OneLayerMLP(X_train.shape[1], width, n_classes).to(device)
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


def run_sweep(out_csv="double_descent_scikit_results.csv"):
    X_train, y_train, X_test, y_test, n_classes = make_digits_dataset()
    print(f"train_size={X_train.shape[0]}, test_size={X_test.shape[0]}, "
          f"n_classes={n_classes}, noise_ratio={NOISE_RATIO}\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = []
    for width in WIDTHS:
        for seed in SEEDS:
            epochs_used, train_err, test_err = train_one(
                width, seed, X_train, y_train, X_test, y_test, n_classes, device=device
            )
            rows.append((width, seed, epochs_used, train_err, test_err))
            print(f"width={width:4d} seed={seed} epochs={epochs_used:5d} "
                  f"train_err={train_err:.4f} test_err={test_err:.4f}")

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["width", "seed", "num_epochs", "train_error", "test_error"])
        writer.writerows(rows)

    return rows


if __name__ == "__main__":
    run_sweep()
