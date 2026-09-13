"""Plot the double-descent width sweep: mean train/test error vs. hidden width."""

import csv
from collections import defaultdict

import matplotlib.pyplot as plt

IN_CSV = "double_descent_results.csv"
OUT_PNG = "double_descent_curve.png"

TRAIN_COLOR = "#0072B2"  # Okabe-Ito blue
TEST_COLOR = "#E69F00"   # Okabe-Ito orange


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "width": int(r["width"]),
                "seed": int(r["seed"]),
                "train_error": float(r["train_error"]),
                "test_error": float(r["test_error"]),
            })
    return rows


def main():
    rows = load_rows(IN_CSV)
    widths = sorted({r["width"] for r in rows})

    train_by_width = defaultdict(list)
    test_by_width = defaultdict(list)
    for r in rows:
        train_by_width[r["width"]].append(r["train_error"])
        test_by_width[r["width"]].append(r["test_error"])

    train_mean = [sum(train_by_width[w]) / len(train_by_width[w]) for w in widths]
    test_mean = [sum(test_by_width[w]) / len(test_by_width[w]) for w in widths]

    fig, ax = plt.subplots(figsize=(7, 5))

    # individual seeds, faint, so the mean lines don't hide per-seed variance
    train_seed_x = [r["width"] for r in rows]
    train_seed_y = [r["train_error"] for r in rows]
    test_seed_y = [r["test_error"] for r in rows]
    ax.scatter(train_seed_x, train_seed_y, color=TRAIN_COLOR, alpha=0.25, s=20, zorder=2)
    ax.scatter(train_seed_x, test_seed_y, color=TEST_COLOR, alpha=0.25, s=20, zorder=2)

    ax.plot(widths, train_mean, color=TRAIN_COLOR, marker="o", linewidth=2,
            label="train error (mean over seeds)", zorder=3)
    ax.plot(widths, test_mean, color=TEST_COLOR, marker="o", linewidth=2,
            label="test error (mean over seeds)", zorder=3)

    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_xlabel("hidden width (log scale)")
    ax.set_ylabel("error rate")
    ax.set_title("Model-wise double descent: modular addition (p=97, 40% label noise)")
    ax.grid(True, which="both", axis="both", color="#dddddd", linewidth=0.8, zorder=1)
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
