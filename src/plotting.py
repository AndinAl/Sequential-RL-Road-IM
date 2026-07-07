from __future__ import annotations

from pathlib import Path

from .utils import ensure_dir


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_claim_comparison(rows: list[dict[str, float | str]], out_path: str | Path) -> None:
    plt = _import_matplotlib()
    labels = [str(row["dataset"]) for row in rows]
    means = [float(row["gap"]) for row in rows]
    lows = [float(row["ci_low"]) for row in rows]
    highs = [float(row["ci_high"]) for row in rows]
    yerr = [[mean - low for mean, low in zip(means, lows)], [high - mean for mean, high in zip(means, highs)]]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = list(range(len(rows)))
    ax.errorbar(means, positions, xerr=yerr, fmt="o", color="#1f4e79", ecolor="#2f2f2f", capsize=4)
    ax.axvline(0.0, color="#7a7a7a", linewidth=1.0, linestyle="--")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Paired gap: RL - Improved Re-rank")
    ax.set_title("RS, MG, and synthetic claim comparison")
    fig.tight_layout()
    target = Path(out_path)
    ensure_dir(target.parent)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def plot_interruption_panels(rows: list[dict[str, object]], out_path: str | Path) -> None:
    plt = _import_matplotlib()
    fig, axes = plt.subplots(1, len(rows), figsize=(13, 4), sharey=True)
    if len(rows) == 1:
        axes = [axes]

    for ax, row in zip(axes, rows):
        values = [float(value) for value in row["interruptions"]]
        mean_confidence = float(row["mean_confidence"])
        ax.hist(values, bins=12, color=str(row.get("color", "#1f4e79")), edgecolor="white", alpha=0.9)
        ax.set_title(str(row["dataset"]))
        ax.set_xlabel("Interruption fraction")
        ax.grid(alpha=0.2, axis="y")
        ax.text(
            0.03,
            0.95,
            f"n={len(values)}\nmean conf={mean_confidence:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
        )

    axes[0].set_ylabel("Count")
    fig.suptitle("Representative interruption profiles across RS, MG, and dense synthetic subsets")
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    target = Path(out_path)
    ensure_dir(target.parent)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def plot_reward_runtime_tradeoff(rows: list[dict[str, float | str]], out_path: str | Path) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(8, 5))
    for row in rows:
        runtime = float(row["runtime_ms"])
        reward = float(row["mean_reward"])
        label = str(row["label"])
        color = str(row.get("color", "#1f4e79"))
        ax.scatter(runtime, reward, s=60, color=color)
        ax.text(runtime, reward, f" {label}", fontsize=8, va="center")
    ax.set_xlabel("Runtime (ms)")
    ax.set_ylabel("Mean reward")
    ax.set_title("Reward-runtime tradeoff")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    target = Path(out_path)
    ensure_dir(target.parent)
    fig.savefig(target, dpi=150)
    plt.close(fig)
