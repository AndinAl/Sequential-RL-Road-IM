from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import ensure_dir, load_yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a lightweight road-style synthetic dataset.")
    parser.add_argument("--config", required=True, help="Path to a YAML config with a generation block.")
    parser.add_argument("--output", required=True, help="Directory where the generated dataset will be written.")
    parser.add_argument("--nodes", type=int, help="Optional override for the number of nodes.")
    parser.add_argument("--snapshots", type=int, help="Optional override for the number of snapshots.")
    parser.add_argument("--avg-degree", dest="avg_degree", type=float, help="Optional override for target average degree.")
    parser.add_argument("--seed", type=int, help="Optional override for the random seed.")
    return parser.parse_args()


def _resolve_params(config: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    generation = dict(config.get("generation", {}))
    dataset = dict(config.get("dataset", {}))
    split = dict(config.get("split", {}))
    shortlist = dict(config.get("shortlist", {}))
    rl = dict(config.get("rl", {}))

    params = {
        "name": str(dataset.get("name", "synthetic")),
        "description": str(generation.get("description", "Road-style synthetic benchmark generated from public config.")),
        "nodes": int(generation.get("nodes", 24)),
        "snapshots": int(generation.get("snapshots", 16)),
        "avg_degree": float(generation.get("avg_degree", 4.0)),
        "seed": int(generation.get("seed", rl.get("seed", 7))),
        "budget": int(generation.get("budget", shortlist.get("K", 3))),
        "width_km": float(generation.get("width_km", 100.0)),
        "height_km": float(generation.get("height_km", 70.0)),
        "corridor_count": int(generation.get("corridor_count", 3)),
        "capacity_floor": float(generation.get("capacity_floor", 110.0)),
        "capacity_scale": float(generation.get("capacity_scale", 55.0)),
        "base_missing_rate": float(generation.get("base_missing_rate", 0.04)),
        "disruption_missing_rate": float(generation.get("disruption_missing_rate", 0.10)),
        "train_fraction": float(split.get("train_fraction", 0.6)),
        "val_fraction": float(split.get("val_fraction", 0.2)),
        "test_fraction": float(split.get("test_fraction", 0.2)),
    }
    if args.nodes is not None:
        params["nodes"] = int(args.nodes)
    if args.snapshots is not None:
        params["snapshots"] = int(args.snapshots)
    if args.avg_degree is not None:
        params["avg_degree"] = float(args.avg_degree)
    if args.seed is not None:
        params["seed"] = int(args.seed)
    return params


def _build_coordinates(n_nodes: int, width_km: float, height_km: float, corridor_count: int, rng: np.random.Generator) -> np.ndarray:
    corridor_count = max(2, corridor_count)
    corridor_y = np.linspace(0.15 * height_km, 0.85 * height_km, corridor_count)
    coordinates = np.zeros((n_nodes, 2), dtype=float)
    for node_id in range(n_nodes):
        corridor_id = node_id % corridor_count
        progress = (node_id + 0.5) / n_nodes
        x_coord = progress * width_km + rng.normal(0.0, width_km * 0.025)
        x_coord = float(np.clip(x_coord, 0.0, width_km))
        y_coord = corridor_y[corridor_id] + rng.normal(0.0, height_km * 0.04)
        y_coord = float(np.clip(y_coord, 0.0, height_km))
        coordinates[node_id] = (x_coord, y_coord)
    return coordinates


def _distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    deltas = coordinates[:, None, :] - coordinates[None, :, :]
    return np.sqrt(np.sum(deltas * deltas, axis=2))


def _build_graph(coordinates: np.ndarray, target_avg_degree: float, rng: np.random.Generator) -> nx.Graph:
    n_nodes = coordinates.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))

    distances = _distance_matrix(coordinates)
    np.fill_diagonal(distances, np.inf)

    local_k = max(2, int(round(target_avg_degree / 2.0)))
    for node_id in range(n_nodes):
        nearest = np.argpartition(distances[node_id], local_k)[:local_k]
        for neighbor_id in nearest:
            graph.add_edge(int(node_id), int(neighbor_id))

    ordering = np.argsort(coordinates[:, 0] + 0.1 * coordinates[:, 1])
    for left, right in zip(ordering[:-1], ordering[1:]):
        graph.add_edge(int(left), int(right))

    tri_i, tri_j = np.triu_indices(n_nodes, k=1)
    pair_distances = distances[tri_i, tri_j]
    noise = rng.uniform(0.0, max(1.0, float(pair_distances.max()) * 1.0e-4), size=len(pair_distances))
    pair_order = np.argsort(pair_distances + noise)
    target_edges = max(n_nodes - 1, int(round(n_nodes * target_avg_degree / 2.0)))
    for pair_index in pair_order:
        if graph.number_of_edges() >= target_edges:
            break
        graph.add_edge(int(tri_i[pair_index]), int(tri_j[pair_index]))

    return graph


def _split_labels(n_snapshots: int, train_fraction: float, val_fraction: float) -> list[str]:
    n_train = max(1, int(round(n_snapshots * train_fraction)))
    n_val = max(1, int(round(n_snapshots * val_fraction)))
    if n_train + n_val >= n_snapshots:
        n_train = max(1, n_snapshots - 2)
        n_val = 1
    labels: list[str] = []
    for snapshot_idx in range(n_snapshots):
        if snapshot_idx < n_train:
            labels.append("train")
        elif snapshot_idx < n_train + n_val:
            labels.append("val")
        else:
            labels.append("test")
    return labels


def _generate_snapshot_rows(
    graph: nx.Graph,
    coordinates: np.ndarray,
    params: dict[str, object],
    rng: np.random.Generator,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    n_nodes = int(params["nodes"])
    n_snapshots = int(params["snapshots"])
    width_km = float(params["width_km"])
    height_km = float(params["height_km"])
    capacity_floor = float(params["capacity_floor"])
    capacity_scale = float(params["capacity_scale"])
    base_missing_rate = float(params["base_missing_rate"])
    disruption_missing_rate = float(params["disruption_missing_rate"])

    degrees = np.asarray([graph.degree(node_id) for node_id in range(n_nodes)], dtype=float)
    degree_norm = degrees / max(1.0, float(degrees.max()))
    center_distance = np.sqrt(
        ((coordinates[:, 0] - width_km / 2.0) / max(width_km / 2.0, 1.0)) ** 2
        + ((coordinates[:, 1] - height_km / 2.0) / max(height_km / 2.0, 1.0)) ** 2
    )
    center_bonus = 1.0 - np.clip(center_distance, 0.0, 1.0)
    spatial_bonus = 0.15 + 0.35 * degree_norm + 0.20 * center_bonus
    capacities = capacity_floor + capacity_scale * (0.45 + degree_norm + 0.35 * center_bonus)

    noise_state = rng.normal(0.0, 0.12, size=n_nodes)
    split_labels = _split_labels(
        n_snapshots,
        train_fraction=float(params["train_fraction"]),
        val_fraction=float(params["val_fraction"]),
    )
    disruption_start = max(1, n_snapshots // 3)
    disruption_end = min(n_snapshots, disruption_start + max(2, n_snapshots // 5))

    rows: list[dict[str, object]] = []
    disruption_snapshots = 0
    missing_rows = 0
    low_confidence_rows = 0

    for snapshot_idx in range(n_snapshots):
        seasonal = 0.18 * math.sin(2.0 * math.pi * snapshot_idx / max(4, n_snapshots))
        seasonal += 0.08 * math.sin(2.0 * math.pi * snapshot_idx / max(3, n_snapshots // 2) + 0.6)
        disruption_flag = 1 if disruption_start <= snapshot_idx < disruption_end else 0
        if disruption_flag:
            disruption_snapshots += 1

        noise_state = 0.82 * noise_state + rng.normal(0.0, 0.08, size=n_nodes)
        for node_id in range(n_nodes):
            centrality_signal = 0.65 + 0.55 * degree_norm[node_id] + 0.25 * center_bonus[node_id]
            raw_flow = capacities[node_id] * max(0.08, centrality_signal + seasonal + noise_state[node_id])
            raw_flow *= 1.0 - 0.18 * disruption_flag * (0.4 + center_bonus[node_id])
            raw_flow = float(max(8.0, raw_flow))

            utilization = float(np.clip(raw_flow / capacities[node_id], 0.05, 1.60))
            capacity_pressure = float(np.clip((utilization - 0.72) / 0.45, 0.0, 1.0))

            interruption_probability = 0.02 + 0.04 * (1.0 - center_bonus[node_id]) + 0.10 * disruption_flag
            interrupted = rng.random() < interruption_probability
            interruption_fraction = float(rng.uniform(0.05, 0.35) if interrupted else 0.0)

            missing_probability = base_missing_rate + disruption_missing_rate * disruption_flag
            missing_probability += 0.04 * interruption_fraction
            is_missing = rng.random() < missing_probability
            if is_missing:
                missing_rows += 1
                if rng.random() < 0.75:
                    confidence = 0.60
                    imputation_method = "neighbor_mean"
                else:
                    confidence = 0.30
                    imputation_method = "static_fallback"
                low_confidence_rows += 1
            else:
                confidence = 1.00
                imputation_method = "observed"

            current_strength = raw_flow / capacities[node_id]
            current_strength = float(np.clip(current_strength + 0.35 * spatial_bonus[node_id], 0.10, 3.50))

            rows.append(
                {
                    "snapshot": snapshot_idx,
                    "node": node_id,
                    "current_strength": round(current_strength, 4),
                    "capacity_pressure": round(capacity_pressure, 4),
                    "interruption_fraction": round(interruption_fraction, 4),
                    "confidence": round(float(confidence), 4),
                    "spatial_bonus": round(float(spatial_bonus[node_id]), 4),
                    "flow": round(raw_flow, 4),
                    "capacity": round(float(capacities[node_id]), 4),
                    "utilization": round(utilization, 4),
                    "split": split_labels[snapshot_idx],
                    "disruption_flag": disruption_flag,
                    "imputation_method": imputation_method,
                }
            )

    metadata = {
        "split_counts": {
            "train": split_labels.count("train"),
            "val": split_labels.count("val"),
            "test": split_labels.count("test"),
        },
        "disruption_snapshots": disruption_snapshots,
        "missing_fraction": round(missing_rows / max(1, len(rows)), 4),
        "low_confidence_fraction": round(low_confidence_rows / max(1, len(rows)), 4),
        "mean_capacity": round(float(np.mean(capacities)), 4),
    }
    return rows, metadata


def _write_edges(output_dir: Path, graph: nx.Graph, coordinates: np.ndarray) -> dict[str, object]:
    edge_rows: list[dict[str, object]] = []
    for edge_id, (source, target) in enumerate(sorted(graph.edges())):
        source_lon = -43.0 + coordinates[source, 0] / 111.0
        source_lat = -22.5 + coordinates[source, 1] / 111.0
        target_lon = -43.0 + coordinates[target, 0] / 111.0
        target_lat = -22.5 + coordinates[target, 1] / 111.0
        distance_km = float(np.linalg.norm(coordinates[source] - coordinates[target]))
        edge_rows.append(
            {
                "edge_id": edge_id,
                "source": source,
                "target": target,
                "distance_km": round(distance_km, 4),
                "source_lon": round(source_lon, 6),
                "source_lat": round(source_lat, 6),
                "target_lon": round(target_lon, 6),
                "target_lat": round(target_lat, 6),
            }
        )

    with (output_dir / "graph_edges.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "edge_id",
                "source",
                "target",
                "distance_km",
                "source_lon",
                "source_lat",
                "target_lon",
                "target_lat",
            ],
        )
        writer.writeheader()
        writer.writerows(edge_rows)

    avg_degree_actual = (2.0 * graph.number_of_edges()) / max(1, graph.number_of_nodes())
    return {
        "edges": graph.number_of_edges(),
        "avg_degree_actual": round(avg_degree_actual, 4),
    }


def _write_snapshots(output_dir: Path, rows: list[dict[str, object]]) -> None:
    with (output_dir / "snapshots.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "snapshot",
                "node",
                "current_strength",
                "capacity_pressure",
                "interruption_fraction",
                "confidence",
                "spatial_bonus",
                "flow",
                "capacity",
                "utilization",
                "split",
                "disruption_flag",
                "imputation_method",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(output_dir: Path, params: dict[str, object], metadata: dict[str, object]) -> None:
    readme = "\n".join(
        [
            f"# {params['name']}",
            "",
            str(params["description"]),
            "",
            f"- nodes: {params['nodes']}",
            f"- snapshots: {params['snapshots']}",
            f"- target average degree: {params['avg_degree']}",
            f"- actual average degree: {metadata['avg_degree_actual']}",
            f"- budget: {params['budget']}",
            "",
            "Generated with:",
            f"`python scripts/generate_synthetic_data.py --config configs/{params['name']}.yaml --output {output_dir.as_posix()}/`" if params["name"] in {"synthetic", "synthetic_small"} else "`python scripts/generate_synthetic_data.py --config <config> --output <dir>`",
            "",
            "Files:",
            "- `graph_edges.csv`: fixed road-style graph topology with spatial coordinates.",
            "- `snapshots.csv`: per-snapshot node signals, capacity, utilization, interruption, confidence, and split labels.",
            "- `metadata.json`: generation parameters and summary statistics.",
        ]
    ) + "\n"
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    config = load_yaml(args.config)
    params = _resolve_params(config, args)
    output_dir = ensure_dir(args.output)
    rng = np.random.default_rng(int(params["seed"]))

    coordinates = _build_coordinates(
        n_nodes=int(params["nodes"]),
        width_km=float(params["width_km"]),
        height_km=float(params["height_km"]),
        corridor_count=int(params["corridor_count"]),
        rng=rng,
    )
    graph = _build_graph(coordinates, float(params["avg_degree"]), rng)
    edge_summary = _write_edges(output_dir, graph, coordinates)
    snapshot_rows, snapshot_summary = _generate_snapshot_rows(graph, coordinates, params, rng)
    _write_snapshots(output_dir, snapshot_rows)

    metadata = {
        "name": params["name"],
        "description": params["description"],
        "seed": int(params["seed"]),
        "budget": int(params["budget"]),
        "nodes": int(params["nodes"]),
        "edges": int(edge_summary["edges"]),
        "avg_degree_target": float(params["avg_degree"]),
        "avg_degree_actual": float(edge_summary["avg_degree_actual"]),
        "snapshots": int(params["snapshots"]),
        "width_km": float(params["width_km"]),
        "height_km": float(params["height_km"]),
        "corridor_count": int(params["corridor_count"]),
        "train_snapshots": int(snapshot_summary["split_counts"]["train"]),
        "val_snapshots": int(snapshot_summary["split_counts"]["val"]),
        "test_snapshots": int(snapshot_summary["split_counts"]["test"]),
        "disruption_snapshots": int(snapshot_summary["disruption_snapshots"]),
        "missing_fraction": float(snapshot_summary["missing_fraction"]),
        "low_confidence_fraction": float(snapshot_summary["low_confidence_fraction"]),
        "mean_capacity": float(snapshot_summary["mean_capacity"]),
        "generated_by": "scripts/generate_synthetic_data.py",
        "output_files": ["graph_edges.csv", "snapshots.csv", "metadata.json", "README.md"],
    }
    (Path(output_dir) / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    _write_readme(Path(output_dir), params, metadata)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
