from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RewardBreakdown:
    activation_gain: float
    coverage_gain: float
    cost_penalty: float
    capacity_penalty: float
    interruption_penalty: float
    low_confidence_penalty: float
    reward: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def compute_reward(
    *,
    activation_gain: float,
    coverage_gain: float = 0.0,
    beta_coverage: float = 0.0,
    lambda_cost: float = 0.0,
    action_cost: float = 0.0,
    eta_capacity: float = 0.0,
    capacity_pressure: float = 0.0,
    eta_interruption: float = 0.0,
    interruption_fraction: float = 0.0,
    eta_low_confidence: float = 0.0,
    confidence: float = 1.0,
) -> RewardBreakdown:
    cost_penalty = max(0.0, lambda_cost * action_cost)
    capacity_penalty = max(0.0, eta_capacity * capacity_pressure)
    interruption_penalty = max(0.0, eta_interruption * interruption_fraction)
    low_confidence_penalty = max(0.0, eta_low_confidence * max(0.0, 1.0 - confidence))
    reward = (
        float(activation_gain)
        + float(beta_coverage) * float(coverage_gain)
        - cost_penalty
        - capacity_penalty
        - interruption_penalty
        - low_confidence_penalty
    )
    return RewardBreakdown(
        activation_gain=float(activation_gain),
        coverage_gain=float(coverage_gain),
        cost_penalty=float(cost_penalty),
        capacity_penalty=float(capacity_penalty),
        interruption_penalty=float(interruption_penalty),
        low_confidence_penalty=float(low_confidence_penalty),
        reward=float(reward),
    )
