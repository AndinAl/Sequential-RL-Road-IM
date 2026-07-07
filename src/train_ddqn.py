from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .environment import TinyRoadEnvironment
from .shortlists import build_deterministic_shortlist
from .utils import ensure_dir, set_global_seed, write_csv, write_json


@dataclass(frozen=True)
class ReplayTransition:
    state_features: np.ndarray
    action_index: int
    reward: float
    next_features: np.ndarray | None
    done: bool


@dataclass(frozen=True)
class SmokeDDQNConfig:
    input_dim: int
    hidden_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.99
    batch_size: int = 16
    tau: float = 0.01


@dataclass(frozen=True)
class TrainingArtifacts:
    checkpoint_path: str
    training_summary_path: str
    validation_selection_path: str
    selected_policy: str


class _ReplayBuffer:
    def __init__(self, capacity: int = 5000) -> None:
        self.capacity = int(capacity)
        self.data: list[ReplayTransition] = []

    def __len__(self) -> int:
        return len(self.data)

    def add(self, transition: ReplayTransition) -> None:
        self.data.append(transition)
        if len(self.data) > self.capacity:
            self.data.pop(0)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[ReplayTransition]:
        indices = rng.choice(len(self.data), size=min(batch_size, len(self.data)), replace=False)
        return [self.data[int(idx)] for idx in indices]


class _DuelingMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
        )
        self.value_head = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), 1),
        )
        self.adv_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        pooled = hidden.mean(dim=1)
        value = self.value_head(pooled).unsqueeze(1)
        advantage = self.adv_head(hidden)
        return (value + advantage - advantage.mean(dim=1, keepdim=True)).squeeze(-1)


class SmokeDDQNAgent:
    def __init__(self, config: SmokeDDQNConfig) -> None:
        self.config = config
        self.online = _DuelingMLP(config.input_dim, config.hidden_dim)
        self.target = _DuelingMLP(config.input_dim, config.hidden_dim)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = _ReplayBuffer()

    def act(self, features: np.ndarray, *, epsilon: float = 0.0, rng: np.random.Generator | None = None) -> int:
        rng = rng or np.random.default_rng(0)
        if rng.random() < epsilon:
            return int(rng.integers(0, len(features)))
        with torch.no_grad():
            q = self.online(torch.as_tensor(features, dtype=torch.float32).unsqueeze(0))
        return int(q.squeeze(0).argmax().item())

    def add_transition(self, transition: ReplayTransition) -> None:
        self.replay.add(transition)

    def _polyak_update(self) -> None:
        with torch.no_grad():
            for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
                target_param.mul_(1.0 - self.config.tau).add_(online_param, alpha=self.config.tau)

    def update(self, rng: np.random.Generator) -> float:
        if len(self.replay) < max(4, self.config.batch_size):
            return 0.0
        batch = self.replay.sample(self.config.batch_size, rng)
        states = torch.as_tensor(np.stack([item.state_features for item in batch]), dtype=torch.float32)
        q_values = self.online(states)
        action_indices = torch.as_tensor([item.action_index for item in batch], dtype=torch.long)
        chosen_q = q_values.gather(1, action_indices[:, None]).squeeze(1)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32)

        targets = rewards.clone()
        non_terminal = [idx for idx, item in enumerate(batch) if not item.done and item.next_features is not None]
        if non_terminal:
            next_states = torch.as_tensor(
                np.stack([batch[idx].next_features for idx in non_terminal]),
                dtype=torch.float32,
            )
            with torch.no_grad():
                next_online_q = self.online(next_states)
                next_actions = next_online_q.argmax(dim=1)
                next_target_q = self.target(next_states)
                next_values = next_target_q.gather(1, next_actions[:, None]).squeeze(1)
            for idx, next_value in zip(non_terminal, next_values):
                targets[idx] = rewards[idx] + self.config.gamma * next_value

        loss = F.smooth_l1_loss(chosen_q, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 2.0)
        self.optimizer.step()
        self._polyak_update()
        return float(loss.detach().cpu())

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "config": {
                    "input_dim": self.config.input_dim,
                    "hidden_dim": self.config.hidden_dim,
                    "learning_rate": self.config.learning_rate,
                    "gamma": self.config.gamma,
                    "batch_size": self.config.batch_size,
                    "tau": self.config.tau,
                },
                "state_dict": self.online.state_dict(),
            },
            str(path),
        )

    @classmethod
    def load(cls, path: str | Path) -> "SmokeDDQNAgent":
        payload = torch.load(str(path), map_location="cpu")
        agent = cls(SmokeDDQNConfig(**payload["config"]))
        agent.online.load_state_dict(payload["state_dict"])
        agent.target.load_state_dict(payload["state_dict"])
        return agent


def _epsilon(episode: int, max_episodes: int) -> float:
    ratio = 1.0 - (episode / max(1, max_episodes - 1))
    return float(max(0.05, 0.25 * ratio))


def train_smoke_ddqn(
    env: TinyRoadEnvironment,
    *,
    output_dir: str | Path,
    shortlist_size: int,
    horizon: int,
    hidden_dim: int,
    learning_rate: float,
    gamma: float,
    batch_size: int,
    max_episodes: int,
    seed: int,
) -> TrainingArtifacts:
    set_global_seed(seed)
    out_dir = ensure_dir(output_dir)
    state = env.reset(0)
    _, features, _ = env.candidate_features(state, horizon=horizon)
    agent = SmokeDDQNAgent(
        SmokeDDQNConfig(
            input_dim=int(features.shape[1]),
            hidden_dim=int(hidden_dim),
            learning_rate=float(learning_rate),
            gamma=float(gamma),
            batch_size=int(batch_size),
        )
    )
    rng = np.random.default_rng(seed)
    start_indices = list(range(min(3, env.dataset.n_snapshots)))
    episode_rows: list[dict[str, Any]] = []
    best_reward = float("-inf")
    best_checkpoint = out_dir / "selected_model.pt"

    for episode in range(int(max_episodes)):
        state = env.reset(start_indices[episode % len(start_indices)])
        total_reward = 0.0
        steps = 0
        last_loss = 0.0
        done = False
        while not done:
            candidates, feature_matrix, feature_names = env.candidate_features(state, horizon=horizon)
            shortlist = build_deterministic_shortlist(candidates, feature_matrix, feature_names, shortlist_size=shortlist_size)
            action_index = agent.act(shortlist.features, epsilon=_epsilon(episode, max_episodes), rng=rng)
            action_id = shortlist.candidates[action_index]
            next_state, breakdown, done = env.step(state, action_id)
            next_features = None
            if not done:
                next_candidates, next_feature_matrix, _ = env.candidate_features(next_state, horizon=horizon)
                next_shortlist = build_deterministic_shortlist(
                    next_candidates,
                    next_feature_matrix,
                    feature_names,
                    shortlist_size=shortlist_size,
                )
                next_features = next_shortlist.features
            agent.add_transition(
                ReplayTransition(
                    state_features=shortlist.features,
                    action_index=action_index,
                    reward=float(breakdown.reward),
                    next_features=next_features,
                    done=done,
                )
            )
            last_loss = agent.update(rng)
            state = next_state
            total_reward += breakdown.reward
            steps += 1

        episode_rows.append(
            {
                "episode": episode,
                "start_snapshot": start_indices[episode % len(start_indices)],
                "total_reward": round(total_reward, 6),
                "steps": steps,
                "loss": round(last_loss, 6),
            }
        )
        if total_reward >= best_reward:
            best_reward = total_reward
            agent.save(best_checkpoint)

    summary = {
        "architecture": "dueling_ddqn_mlp",
        "input_dim": int(features.shape[1]),
        "hidden_dim": int(hidden_dim),
        "learning_rate": float(learning_rate),
        "gamma": float(gamma),
        "batch_size": int(batch_size),
        "max_episodes": int(max_episodes),
        "seed": int(seed),
        "best_episode_reward": round(best_reward, 6),
        "trained_from_scratch": True,
        "selected_policy": f"smoke_ddqn_h{hidden_dim}_s{seed}",
    }
    training_summary_path = out_dir / "training_summary.json"
    validation_selection_path = out_dir / "validation_selection.csv"
    write_json(training_summary_path, summary)
    write_csv(validation_selection_path, episode_rows, ["episode", "start_snapshot", "total_reward", "steps", "loss"])

    return TrainingArtifacts(
        checkpoint_path=str(best_checkpoint),
        training_summary_path=str(training_summary_path),
        validation_selection_path=str(validation_selection_path),
        selected_policy=str(summary["selected_policy"]),
    )
