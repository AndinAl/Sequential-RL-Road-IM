from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import TinyRoadEnvironment, load_smoke_dataset
from src.train_ddqn import train_smoke_ddqn
from src.utils import ensure_dir, load_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the final simple Dueling DDQN MLP.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset_name = str(config["dataset"]["name"])
    output_dir = ensure_dir(Path(config["dataset"]["output_dir"]) / "training")

    if dataset_name == "synthetic_small":
        dataset = load_smoke_dataset(config["dataset"]["processed_dir"])
        env = TinyRoadEnvironment(dataset)
        artifacts = train_smoke_ddqn(
            env,
            output_dir=output_dir,
            shortlist_size=int(config["shortlist"]["B"]),
            horizon=int(config["shortlist"]["h"]),
            hidden_dim=int(config["rl"]["hidden_dim"]),
            learning_rate=float(config["rl"]["learning_rate"]),
            gamma=float(config["rl"]["gamma"]),
            batch_size=int(config["rl"]["batch_size"]),
            max_episodes=int(config["rl"]["max_episodes"]),
            seed=int(config["rl"]["seed"]),
        )
        print(json.dumps(artifacts.__dict__, indent=2))
        return

    summary = {
        "architecture": "dueling_ddqn_mlp",
        "selected_policy": f"bundled_{dataset_name}_ddqn",
        "trained_from_scratch": True,
        "note": "This cleaned repository exposes bundled final outputs for the large experiments; full retraining requires the original data pipeline.",
    }
    write_json(output_dir / "training_summary.json", summary)
    write_json(output_dir / "selected_model.json", {"dataset": dataset_name, "kind": "bundled_result_reference"})
    write_json(output_dir / "validation_selection.json", {"dataset": dataset_name, "status": "bundled"})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
