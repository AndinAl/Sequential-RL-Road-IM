from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from .reward import RewardBreakdown, compute_reward


@dataclass(frozen=True)
class SmokeDataset:
    name: str
    nodes: list[int]
    edges: list[tuple[int, int]]
    snapshots: list[dict[int, dict[str, float]]]
    budget: int
    metadata: dict[str, object]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_snapshots(self) -> int:
        return len(self.snapshots)


@dataclass
class EnvironmentState:
    start_snapshot_idx: int
    snapshot_idx: int
    step: int
    selected: list[int]
    active: set[int]
    total_reward: float = 0.0


def load_smoke_dataset(data_dir: str | Path) -> SmokeDataset:
    root = Path(data_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))

    edges: list[tuple[int, int]] = []
    with (root / "graph_edges.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            edges.append((int(row["source"]), int(row["target"])))

    snapshots_by_t: dict[int, dict[int, dict[str, float]]] = {}
    nodes: set[int] = set()
    with (root / "snapshots.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            snapshot_idx = int(row["snapshot"])
            node_id = int(row["node"])
            nodes.add(node_id)
            snapshots_by_t.setdefault(snapshot_idx, {})[node_id] = {
                "current_strength": float(row["current_strength"]),
                "capacity_pressure": float(row["capacity_pressure"]),
                "interruption_fraction": float(row["interruption_fraction"]),
                "confidence": float(row["confidence"]),
                "spatial_bonus": float(row["spatial_bonus"]),
            }

    snapshots = [snapshots_by_t[idx] for idx in sorted(snapshots_by_t)]
    return SmokeDataset(
        name=str(metadata.get("name", "synthetic_small")),
        nodes=sorted(nodes),
        edges=edges,
        snapshots=snapshots,
        budget=int(metadata.get("budget", 3)),
        metadata=metadata,
    )


class TinyRoadEnvironment:
    feature_names = [
        "current_strength",
        "future_mean_strength",
        "degree",
        "pagerank",
        "betweenness",
        "capacity_pressure",
        "interruption_fraction",
        "confidence",
        "spatial_bonus",
        "already_selected",
        "already_active",
        "step_fraction",
    ]

    def __init__(
        self,
        dataset: SmokeDataset,
        *,
        beta_coverage: float = 0.5,
        lambda_cost: float = 0.05,
        eta_capacity: float = 0.8,
        eta_interruption: float = 0.7,
        eta_low_confidence: float = 0.4,
    ) -> None:
        self.dataset = dataset
        self.graph = nx.Graph()
        self.graph.add_nodes_from(dataset.nodes)
        self.graph.add_edges_from(dataset.edges)
        self.degree = nx.degree_centrality(self.graph)
        self.pagerank = nx.pagerank(self.graph, alpha=0.85)
        self.betweenness = nx.betweenness_centrality(self.graph, normalized=True)
        self.beta_coverage = float(beta_coverage)
        self.lambda_cost = float(lambda_cost)
        self.eta_capacity = float(eta_capacity)
        self.eta_interruption = float(eta_interruption)
        self.eta_low_confidence = float(eta_low_confidence)

    def reset(self, start_snapshot_idx: int = 0) -> EnvironmentState:
        idx = max(0, min(int(start_snapshot_idx), self.dataset.n_snapshots - 1))
        return EnvironmentState(
            start_snapshot_idx=idx,
            snapshot_idx=idx,
            step=0,
            selected=[],
            active=set(),
            total_reward=0.0,
        )

    def legal_candidates(self, state: EnvironmentState) -> list[int]:
        selected = set(state.selected)
        return [node for node in self.dataset.nodes if node not in selected]

    def _node_payload(self, node_id: int, snapshot_idx: int) -> dict[str, float]:
        idx = min(snapshot_idx, self.dataset.n_snapshots - 1)
        return self.dataset.snapshots[idx][node_id]

    def _future_mean_strength(self, node_id: int, snapshot_idx: int, horizon: int) -> float:
        end = min(self.dataset.n_snapshots, snapshot_idx + max(1, horizon))
        values = [self.dataset.snapshots[idx][node_id]["current_strength"] for idx in range(snapshot_idx, end)]
        return float(np.mean(values))

    def candidate_features(self, state: EnvironmentState, *, horizon: int) -> tuple[list[int], np.ndarray, list[str]]:
        candidates = self.legal_candidates(state)
        rows: list[list[float]] = []
        for node_id in candidates:
            payload = self._node_payload(node_id, state.snapshot_idx)
            rows.append(
                [
                    payload["current_strength"],
                    self._future_mean_strength(node_id, state.snapshot_idx, horizon),
                    float(self.degree.get(node_id, 0.0)),
                    float(self.pagerank.get(node_id, 0.0)),
                    float(self.betweenness.get(node_id, 0.0)),
                    payload["capacity_pressure"],
                    payload["interruption_fraction"],
                    payload["confidence"],
                    payload["spatial_bonus"],
                    1.0 if node_id in state.selected else 0.0,
                    1.0 if node_id in state.active else 0.0,
                    float(state.step / max(1, self.dataset.budget)),
                ]
            )
        return candidates, np.asarray(rows, dtype=np.float32), list(self.feature_names)

    def raw_candidate_gains(self, state: EnvironmentState, candidates: list[int]) -> np.ndarray:
        gains = []
        for node_id in candidates:
            payload = self._node_payload(node_id, state.snapshot_idx)
            uncovered_neighbors = [nbr for nbr in self.graph.neighbors(node_id) if nbr not in state.active]
            activation_gain = payload["current_strength"] + 0.15 * len(uncovered_neighbors)
            coverage_gain = payload["spatial_bonus"] + 0.05 * len(uncovered_neighbors)
            breakdown = compute_reward(
                activation_gain=activation_gain,
                coverage_gain=coverage_gain,
                beta_coverage=self.beta_coverage,
                lambda_cost=self.lambda_cost,
                action_cost=1.0,
                eta_capacity=self.eta_capacity,
                capacity_pressure=payload["capacity_pressure"],
                eta_interruption=self.eta_interruption,
                interruption_fraction=payload["interruption_fraction"],
                eta_low_confidence=self.eta_low_confidence,
                confidence=payload["confidence"],
            )
            gains.append(breakdown.reward)
        return np.asarray(gains, dtype=np.float32)

    def step(self, state: EnvironmentState, action: int) -> tuple[EnvironmentState, RewardBreakdown, bool]:
        if action in state.selected:
            raise ValueError(f"Action {action} is already selected")
        payload = self._node_payload(action, state.snapshot_idx)
        newly_active = {action}
        newly_active.update(self.graph.neighbors(action))
        fresh_nodes = newly_active.difference(state.active)
        activation_gain = payload["current_strength"] + 0.15 * len(fresh_nodes)
        coverage_gain = payload["spatial_bonus"] + 0.10 * len(fresh_nodes)
        breakdown = compute_reward(
            activation_gain=activation_gain,
            coverage_gain=coverage_gain,
            beta_coverage=self.beta_coverage,
            lambda_cost=self.lambda_cost,
            action_cost=1.0,
            eta_capacity=self.eta_capacity,
            capacity_pressure=payload["capacity_pressure"],
            eta_interruption=self.eta_interruption,
            interruption_fraction=payload["interruption_fraction"],
            eta_low_confidence=self.eta_low_confidence,
            confidence=payload["confidence"],
        )
        next_active = set(state.active)
        next_active.update(fresh_nodes)
        next_step = state.step + 1
        next_snapshot = min(state.start_snapshot_idx + next_step, self.dataset.n_snapshots - 1)
        next_state = EnvironmentState(
            start_snapshot_idx=state.start_snapshot_idx,
            snapshot_idx=next_snapshot,
            step=next_step,
            selected=[*state.selected, int(action)],
            active=next_active,
            total_reward=float(state.total_reward + breakdown.reward),
        )
        done = next_step >= self.dataset.budget or len(next_state.selected) >= self.dataset.n_nodes
        return next_state, breakdown, done
