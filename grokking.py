"""
Grokking replication on synthetic modular addition.

Task: (a, b) -> (a + b) mod p, one-hot(a) concat one-hot(b) as input, p-way
classification, CLEAN labels (no noise -- this is a delayed-generalization
study, distinct from the label-noise double-descent study in
double_descent.py).

One-hot inputs are mutually orthogonal for every distinct (a,b) pair, so the
network can only generalize by discovering the additive structure from
scratch during training. With strong weight decay, this produces the
grokking signature: training accuracy saturates to ~100% almost immediately
while test accuracy lags near chance for a long stretch, then rises sharply
to ~100% much later in training.

Logs train/test accuracy at fixed epoch intervals to a CSV and plots the
grokking curve (accuracy vs. epoch, log-x) to a PNG. Runs on GPU if
available -- intended to be run from grokking.ipynb on Colab, or directly
via `python grokking.py`.
"""

import csv
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --- dataset ---
P = 97                  # modulus / number of classes (matches Power et al. 2022)
TRAIN_FRACTION = 0.5    # fraction of the p*p domain used for training
DATA_SEED = 0

# --- model / optimization ---
HIDDEN = 128
SEED = 0
LR = 1e-3
WEIGHT_DECAY = 1.0      # strong weight decay is what makes grokking tractable
MAX_EPOCHS = 100_000
LOG_EVERY = 50
GROK_PATIENCE_CHECKS = 20    # keep logging this many more checkpoints after grokking, then stop
GROK_THRESHOLD = 0.99


def make_dataset(p=P, train_frac=TRAIN_FRACTION, seed=DATA_SEED, device="cpu"):
    g = torch.Generator().manual_seed(seed)

    a_all = torch.arange(p * p) // p
    b_all = torch.arange(p * p) % p
    labels_all = (a_all + b_all) % p

    def onehot_pairs(a, b, p):
        n = a.shape[0]
        X = torch.zeros(n, 2 * p)
        X[torch.arange(n), a] = 1.0
        X[torch.arange(n), p + b] = 1.0
        return X

    train_size = round(train_frac * p * p)
    perm = torch.randperm(p * p, generator=g)
    train_idx, test_idx = perm[:train_size], perm[train_size:]

    X_all = onehot_pairs(a_all, b_all, p)
    X_train, y_train = X_all[train_idx].to(device), labels_all[train_idx].to(device)
    X_test, y_test = X_all[test_idx].to(device), labels_all[test_idx].to(device)
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


def accuracy(logits, y):
    return (logits.argmax(dim=1) == y).float().mean().item()


def run_grokking(out_csv="grokking_results.csv", out_png="grokking_curve.png"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    X_train, y_train, X_test, y_test = make_dataset(device=device)
    print(f"p={P}, train_size={X_train.shape[0]}, test_size={X_test.shape[0]}")

    torch.manual_seed(SEED)
    model = OneLayerMLP(X_train.shape[1], HIDDEN, P).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    log = []  # (epoch, train_acc, test_acc, train_loss)
    grok_checks_remaining = None

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        logits = model(X_train)
        loss = loss_fn(logits, y_train)
        loss.backward()
        opt.step()

        if epoch % LOG_EVERY == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                train_acc = accuracy(logits, y_train)
                test_acc = accuracy(model(X_test), y_test)
            log.append((epoch, train_acc, test_acc, loss.item()))
            print(f"epoch={epoch:7d} loss={loss.item():.5f} "
                  f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}")

            if test_acc >= GROK_THRESHOLD:
                if grok_checks_remaining is None:
                    grok_checks_remaining = GROK_PATIENCE_CHECKS
                grok_checks_remaining -= 1
                if grok_checks_remaining <= 0:
                    break

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_accuracy", "test_accuracy", "train_loss"])
        writer.writerows(log)

    epochs, train_accs, test_accs, _ = zip(*log)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_accs, label="train accuracy")
    plt.plot(epochs, test_accs, label="test accuracy")
    plt.xscale("log")
    plt.xlabel("epoch (log scale)")
    plt.ylabel("accuracy")
    plt.title(f"Grokking: modular addition mod {P}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved {out_csv} and {out_png}")

    return log


if __name__ == "__main__":
    run_grokking()
