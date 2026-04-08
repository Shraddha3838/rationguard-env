"""RationGuardEnv: deterministic multi-step PDS fraud detection environment."""

from typing import Dict, List, Optional

from env.grader import FINAL_DECISION_ACTIONS, INVESTIGATION_ACTIONS, evaluate_step
from env.models import RationAction, RationObservation, RewardBreakdown, StepInfo
from env.tasks import generate_task, supported_task_levels


class RationGuardEnv:
    """OpenEnv-style environment with reset(), step(), and state()."""

    def __init__(self, default_level: str = "easy", max_steps: int = 5):
        self.default_level = default_level
        self.max_steps = max_steps

        self._task: Dict = {}
        self._done: bool = False
        self._step_number: int = 0
        self._suspicion_score: float = 0.10
        self._revealed_checks: Dict[str, bool] = {
            "beneficiary_audit": False,
            "dealer_ledger": False,
            "transport_log": False,
        }
        self._revealed_indicators: Dict[str, float] = {
            "beneficiary_gap_ratio": 0.0,
            "dealer_ledger_gap": 0.0,
            "transport_loss_ratio": 0.0,
        }
        self._executed_evidence_actions: List[str] = []
        self._last_action: Optional[str] = None
        self._last_error: Optional[str] = None

    def _clamp_01(self, value: float) -> float:
        return max(0.0, min(1.0, round(value, 4)))

    def _allowed_actions(self) -> List[str]:
        if self._done:
            return []
        return sorted(list(INVESTIGATION_ACTIONS | FINAL_DECISION_ACTIONS))

    def _build_observation(self) -> Dict:
        obs = RationObservation(
            task_level=self._task["task_level"],
            task_id=self._task["task_id"],
            step_number=self._step_number,
            max_steps=self.max_steps,
            expected_quota=self._task["expected_quota"],
            claimed_quantity=self._task["claimed_quantity"],
            dealer_stock=self._task["dealer_stock"],
            transported_stock=self._task["transported_stock"],
            suspicion_score=self._clamp_01(self._suspicion_score),
            revealed_checks=dict(self._revealed_checks),
            revealed_indicators={k: round(v, 4) for k, v in self._revealed_indicators.items()},
            allowed_actions=self._allowed_actions(),
            last_action=self._last_action,
        )
        return obs.model_dump()

    def reset(self, level: str = "easy") -> Dict:
        if level not in supported_task_levels():
            level = self.default_level

        self._task = generate_task(level)
        self._done = False
        self._step_number = 0
        self._suspicion_score = 0.10
        self._revealed_checks = {
            "beneficiary_audit": False,
            "dealer_ledger": False,
            "transport_log": False,
        }
        self._revealed_indicators = {
            "beneficiary_gap_ratio": 0.0,
            "dealer_ledger_gap": 0.0,
            "transport_loss_ratio": 0.0,
        }
        self._executed_evidence_actions = []
        self._last_action = None
        self._last_error = None

        return self._build_observation()

    def step(self, action: Dict):
        if not self._task:
            self.reset(self.default_level)

        if self._done:
            timeout_info = StepInfo(
                action_valid=False,
                action_type="invalid",
                evaluation="incorrect",
                last_action_error="Episode already completed. Call reset() to start a new episode.",
                reward_breakdown=RewardBreakdown(
                    base=0.0,
                    evidence_bonus=0.0,
                    penalty=0.0,
                    early_bonus=0.0,
                    final_reward=0.0,
                ),
                ground_truth_fraud=self._task["ground_truth_fraud"],
                expected_final_action=self._task["expected_final_action"],
            )
            return self._build_observation(), 0.0, True, timeout_info.model_dump()

        self._step_number += 1
        self._last_error = None

        action_model = None
        try:
            action_model = RationAction(**action)
        except Exception:
            self._last_error = "Invalid action payload. Expected {'action': <supported_action>}"

        action_str = action_model.action if action_model else "INVALID_ACTION"
        self._last_action = action_str

        executed_before = list(self._executed_evidence_actions)

        if action_str in INVESTIGATION_ACTIONS:
            effect = self._task["investigation_effects"][action_str]
            indicator_key = effect["indicator_key"]
            self._revealed_indicators[indicator_key] = effect["indicator_value"]

            if action_str == "REQUEST_BENEFICIARY_AUDIT":
                self._revealed_checks["beneficiary_audit"] = True
            elif action_str == "REQUEST_DEALER_LEDGER":
                self._revealed_checks["dealer_ledger"] = True
            elif action_str == "REQUEST_TRANSPORT_LOG":
                self._revealed_checks["transport_log"] = True

            self._suspicion_score = self._clamp_01(self._suspicion_score + effect["suspicion_delta"])
            if action_str not in self._executed_evidence_actions:
                self._executed_evidence_actions.append(action_str)

        elif action_str in FINAL_DECISION_ACTIONS:
            self._done = True

        elif self._last_error is None:
            self._last_error = "Unsupported action string."

        reward_parts = evaluate_step(
            action=action_str,
            expected_final_action=self._task["expected_final_action"],
            required_evidence_actions=self._task["required_evidence_actions"],
            executed_evidence_actions=executed_before,
            step_number=self._step_number,
            max_steps=self.max_steps,
        )

        if self._last_error:
            reward_parts["final_reward"] = 0.0
            reward_parts["penalty"] = self._clamp_01(reward_parts["penalty"] + 0.20)

        timeout_reached = False
        if not self._done and self._step_number >= self.max_steps:
            self._done = True
            timeout_reached = True

        info = StepInfo(
            action_valid=self._last_error is None,
            action_type=(
                "auto_timeout"
                if timeout_reached and action_str not in FINAL_DECISION_ACTIONS
                else "investigate"
                if action_str in INVESTIGATION_ACTIONS
                else "final_decision"
                if action_str in FINAL_DECISION_ACTIONS
                else "invalid"
            ),
            evaluation=str(reward_parts.get("evaluation", "incorrect")),
            last_action_error=self._last_error,
            reward_breakdown=RewardBreakdown(
                base=float(reward_parts["base"]),
                evidence_bonus=float(reward_parts["evidence_bonus"]),
                penalty=float(reward_parts["penalty"]),
                early_bonus=float(reward_parts["early_bonus"]),
                final_reward=float(reward_parts["final_reward"]),
            ),
            ground_truth_fraud=self._task["ground_truth_fraud"] if self._done else None,
            expected_final_action=self._task["expected_final_action"],
        )

        return self._build_observation(), reward_parts["final_reward"], self._done, info.model_dump()

    def state(self) -> Dict:
        """Return full current internal state for deterministic debugging."""

        return {
            "task": dict(self._task),
            "done": self._done,
            "step_number": self._step_number,
            "max_steps": self.max_steps,
            "suspicion_score": self._clamp_01(self._suspicion_score),
            "revealed_checks": dict(self._revealed_checks),
            "revealed_indicators": dict(self._revealed_indicators),
            "executed_evidence_actions": list(self._executed_evidence_actions),
            "last_action": self._last_action,
            "last_action_error": self._last_error,
        }

    def close(self) -> None:
        """Compatibility no-op for runners that expect a close() method."""

        return None
