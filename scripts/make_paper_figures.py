from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plotting import plot_claim_comparison, plot_interruption_panels
from src.utils import ensure_dir


def _read_interruptions(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    interruptions = [float(row["interruption_fraction"]) for row in rows]
    confidences = [float(row["confidence_score"]) for row in rows]
    dataset = Path(path).name.replace("_imputed_subset.csv", "").upper()
    return {
        "dataset": dataset,
        "interruptions": interruptions,
        "mean_confidence": sum(confidences) / max(len(confidences), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready figures.")
    parser.add_argument("--all", action="store_true", help="Ignored flag for compatibility with the README examples.")
    parser.parse_args()

    figures_dir = ensure_dir("figures/paper")
    interruption_rows = [
        {**_read_interruptions("data/imputed_subset/rs_imputed_subset.csv"), "color": "#1f4e79"},
        {**_read_interruptions("data/imputed_subset/mg_imputed_subset.csv"), "color": "#c0504d"},
        {**_read_interruptions("data/imputed_subset/synthetic_imputed_subset.csv"), "color": "#2e8b57"},
    ]
    plot_interruption_panels(interruption_rows, Path(figures_dir) / "fig_networks_three_panel_interruptions.png")

    claim_rows = [
        {"dataset": "RS", "gap": 2.18, "ci_low": 0.66, "ci_high": 3.76},
        {"dataset": "MG", "gap": -5.26, "ci_low": -6.90, "ci_high": -3.62},
        {"dataset": "Synthetic", "gap": 45.40, "ci_low": 32.86, "ci_high": 58.06},
    ]
    plot_claim_comparison(claim_rows, Path(figures_dir) / "fig_rs_mg_methodology_replication.png")
    print("Paper figures written to figures/paper/")


if __name__ == "__main__":
    main()
