"""Typed models for RationGuardEnv.

The models are intentionally compact and strictly structured so OpenEnv
evaluators can parse trajectories deterministically.
"""

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ActionType = Literal[
    "REQUEST_BENEFICIARY_AUDIT",
    "REQUEST_DEALER_LEDGER",
    "REQUEST_TRANSPORT_LOG",
    "FLAG_BENEFICIARY_FRAUD",
    "FLAG_DEALER_FRAUD",
    "FLAG_SUPPLY_CHAIN_FRAUD",
    "CLEAR_CLAIM",
]


class RationAction(BaseModel):
    """Single environment action."""

    action: ActionType


class RationObservation(BaseModel):
    """Structured observation returned every step."""

    task_level: Literal["easy", "medium", "hard"]
    task_id: str
    step_number: int
    max_steps: int
    expected_quota: int
    claimed_quantity: int
    dealer_stock: int
    transported_stock: int
    suspicion_score: float = Field(ge=0.0, le=1.0)
    revealed_checks: Dict[str, bool]
    revealed_indicators: Dict[str, float]
    allowed_actions: List[str]
    last_action: Optional[str] = None


class RewardBreakdown(BaseModel):
    base: float
    evidence_bonus: float
    penalty: float
    early_bonus: float
    final_reward: float


class StepInfo(BaseModel):
    action_valid: bool
    action_type: Literal["investigate", "final_decision", "invalid", "auto_timeout"]
    evaluation: Literal["correct", "partial", "incorrect"]
    last_action_error: Optional[str] = None
    reward_breakdown: RewardBreakdown
    ground_truth_fraud: Optional[Literal["beneficiary", "dealer", "supply_chain"]] = None
    expected_final_action: Optional[str] = None
