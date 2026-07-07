from __future__ import annotations

from pathlib import Path
from typing import Any

from .environment import load_smoke_dataset
from .utils import ensure_dir, write_json


def convert_configured_dataset(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    name = str(dataset_cfg.get("name", "unknown"))
    output_dir = ensure_dir(dataset_cfg.get("output_dir", f"experiments/{name}"))
    if name == "synthetic_small":
        dataset = load_smoke_dataset(dataset_cfg["processed_dir"])
        payload = {
            "dataset": name,
            "n_nodes": dataset.n_nodes,
            "n_edges": dataset.n_edges,
            "n_snapshots": dataset.n_snapshots,
            "status": "ready",
        }
    else:
        payload = {
            "dataset": name,
            "raw_dir": dataset_cfg.get("raw_dir"),
            "processed_dir": dataset_cfg.get("processed_dir"),
            "status": "external_data_required",
        }
    write_json(Path(output_dir) / "conversion_manifest.json", payload)
    return payload
