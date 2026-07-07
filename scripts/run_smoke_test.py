from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import TinyRoadEnvironment, load_smoke_dataset
from src.evaluate import evaluate_policies, write_smoke_outputs
from src.imputation import impute_matrix
from src.shortlists import build_deterministic_shortlist, shortlist_manifest
from src.train_ddqn import train_smoke_ddqn
from src.utils import ensure_dir, load_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the lightweight end-to-end smoke pipeline.")
    parser.add_argument("--config", default="configs/synthetic_small.yaml", help="Path to the smoke-test config.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset = load_smoke_dataset(config["dataset"]["processed_dir"])
    env = TinyRoadEnvironment(dataset)
    output_dir = ensure_dir(config["dataset"]["output_dir"])

    strength_matrix = np.asarray(
        [
            [dataset.snapshots[t][node]["current_strength"] for t in range(dataset.n_snapshots)]
            for node in dataset.nodes
        ],
        dtype=float,
    )
    masked = strength_matrix.copy()
    masked[1, 1] = np.nan
    masked[4, 2] = np.nan
    neighbor_fill = np.tile(np.nanmean(strength_matrix, axis=0, keepdims=True), (strength_matrix.shape[0], 1))
    static_fill = np.nanmean(strength_matrix, axis=1)
    imputed = impute_matrix(masked, neighbor_values=neighbor_fill, static_values=static_fill)
    write_json(
        Path(output_dir) / "imputation_summary.json",
        {
            "fallback_rate": round(imputed.fallback_rate, 6),
            "confidence_min": float(np.min(imputed.confidence)),
            "confidence_max": float(np.max(imputed.confidence)),
            "missing_after_imputation": int(np.isnan(imputed.values).sum()),
        },
    )

    shortlists = []
    for start in range(int(config["evaluation"]["n_test_starts"])):
        state = env.reset(start % dataset.n_snapshots)
        candidates, features, feature_names = env.candidate_features(state, horizon=int(config["shortlist"]["h"]))
        shortlists.append(
            build_deterministic_shortlist(
                candidates,
                features,
                feature_names,
                shortlist_size=int(config["shortlist"]["B"]),
            )
        )
    write_json(Path(output_dir) / "shortlist_manifest.json", shortlist_manifest(shortlists))

    artifacts = train_smoke_ddqn(
        env,
        output_dir=Path(output_dir) / "training",
        shortlist_size=int(config["shortlist"]["B"]),
        horizon=int(config["shortlist"]["h"]),
        hidden_dim=int(config["rl"]["hidden_dim"]),
        learning_rate=float(config["rl"]["learning_rate"]),
        gamma=float(config["rl"]["gamma"]),
        batch_size=int(config["rl"]["batch_size"]),
        max_episodes=int(config["rl"]["max_episodes"]),
        seed=int(config["rl"]["seed"]),
    )

    starts = [idx % dataset.n_snapshots for idx in range(int(config["evaluation"]["n_test_starts"]))]
    evaluation = evaluate_policies(
        env,
        starts=starts,
        shortlist_size=int(config["shortlist"]["B"]),
        horizon=int(config["shortlist"]["h"]),
        rl_checkpoint=artifacts.checkpoint_path,
        hidden_dim=int(config["rl"]["hidden_dim"]),
        seed=int(config["rl"]["seed"]),
    )
    report = write_smoke_outputs(
        output_dir=output_dir,
        dataset_name=dataset.name,
        evaluation=evaluation,
        shortlist_size=int(config["shortlist"]["B"]),
        horizon=int(config["shortlist"]["h"]),
        budget=int(config["shortlist"]["K"]),
        checkpoint_path=artifacts.checkpoint_path,
        bootstrap=int(config["evaluation"]["bootstrap"]),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
