from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import TinyRoadEnvironment, load_smoke_dataset
from src.evaluate import bundled_validation_report, evaluate_policies, write_bundled_outputs, write_smoke_outputs
from src.train_ddqn import train_smoke_ddqn
from src.utils import ensure_dir, load_yaml


def _smoke_checkpoint(config: dict) -> Path:
    return Path(config["dataset"]["output_dir"]) / "training" / "selected_model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate selected RL and baseline policies.")
    parser.add_argument("--config", required=True, help="Path to a YAML experiment config.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset_name = str(config["dataset"]["name"])
    output_dir = ensure_dir(config["dataset"]["output_dir"])

    if dataset_name == "synthetic_small":
        dataset = load_smoke_dataset(config["dataset"]["processed_dir"])
        env = TinyRoadEnvironment(dataset)
        checkpoint_path = _smoke_checkpoint(config)
        if not checkpoint_path.exists():
            train_smoke_ddqn(
                env,
                output_dir=Path(config["dataset"]["output_dir"]) / "training",
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
            rl_checkpoint=checkpoint_path,
            hidden_dim=int(config["rl"]["hidden_dim"]),
            seed=int(config["rl"]["seed"]),
        )
        report = write_smoke_outputs(
            output_dir=output_dir,
            dataset_name=dataset_name,
            evaluation=evaluation,
            shortlist_size=int(config["shortlist"]["B"]),
            horizon=int(config["shortlist"]["h"]),
            budget=int(config["shortlist"]["K"]),
            checkpoint_path=str(checkpoint_path),
            bootstrap=int(config["evaluation"]["bootstrap"]),
        )
        print(json.dumps(report, indent=2))
        return

    report = config.get("public_validation_report")
    claim = config.get("public_claim")
    if report is None:
        bundled_report = config["dataset"].get("bundled_report")
        if not bundled_report:
            raise ValueError(f"No public or bundled validation report configured for dataset: {dataset_name}")
        report = json.loads(Path(bundled_report).read_text(encoding="utf-8"))
    if claim is None:
        bundled_claim = config["dataset"].get("bundled_claim")
        if not bundled_claim:
            raise ValueError(f"No public or bundled claim configured for dataset: {dataset_name}")
        claim = json.loads(Path(bundled_claim).read_text(encoding="utf-8"))
    normalized = bundled_validation_report(
        dataset=dataset_name,
        report=report,
        claim=claim,
        budget=int(config["shortlist"]["K"]),
        horizon=int(config["shortlist"]["h"]),
        shortlist_size=int(config["shortlist"]["B"]),
    )
    write_bundled_outputs(
        output_dir=output_dir,
        validation_report=normalized,
        final_claim=claim,
    )
    print(json.dumps(normalized, indent=2))


if __name__ == "__main__":
    main()
