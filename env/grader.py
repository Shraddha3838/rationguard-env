"""Deterministic, rule-based grading utilities for RationGuardEnv."""

from typing import Dict, List


INVESTIGATION_ACTIONS = {
    "REQUEST_BENEFICIARY_AUDIT",
    "REQUEST_DEALER_LEDGER",
    "REQUEST_TRANSPORT_LOG",
}

FINAL_DECISION_ACTIONS = {
    "FLAG_BENEFICIARY_FRAUD",
    "FLAG_DEALER_FRAUD",
    "FLAG_SUPPLY_CHAIN_FRAUD",
    "CLEAR_CLAIM",
}


def _clamp_01(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def evaluate_step(
    action: str,
    expected_final_action: str,
    required_evidence_actions: List[str],
    executed_evidence_actions: List[str],
    step_number: int,
    max_steps: int,
) -> Dict[str, float | str]:
    """Return deterministic reward components for a single step.

    Reward is always clamped to [0, 1].
    """

    base = 0.0
    evidence_bonus = 0.0
    penalty = 0.0
    early_bonus = 0.0

    evaluation = "incorrect"

    if action in INVESTIGATION_ACTIONS:
        base = 0.08
        if action in required_evidence_actions and action not in executed_evidence_actions:
            evidence_bonus = 0.14
            evaluation = "partial"
        elif action in executed_evidence_actions:
            penalty = 0.05
            evaluation = "incorrect"
        else:
            penalty = 0.02
            evaluation = "partial"

    elif action in FINAL_DECISION_ACTIONS:
        coverage = 0.0
        if required_evidence_actions:
            completed = sum(1 for x in required_evidence_actions if x in executed_evidence_actions)
            coverage = completed / len(required_evidence_actions)

        if action == expected_final_action:
            if action == "CLEAR_CLAIM":
                # Explicit no-fraud edge case: clean claim should receive full score.
                base = 1.0
                evidence_bonus = 0.0
                early_bonus = 0.0
            else:
                base = 0.45
                evidence_bonus = 0.25 * coverage
                if step_number <= max_steps - 1:
                    early_bonus = 0.20
            evaluation = "correct"
        else:
            # Partial score for wrong final decision if evidence collection is strong
            base = 0.15 * coverage
            penalty = 0.20
            evaluation = "partial" if coverage >= 0.66 else "incorrect"

    else:
        penalty = 0.18
        evaluation = "incorrect"

    final_reward = _clamp_01(base + evidence_bonus + early_bonus - penalty)
    return {
        "base": _clamp_01(base),
        "evidence_bonus": _clamp_01(evidence_bonus),
        "penalty": _clamp_01(penalty),
        "early_bonus": _clamp_01(early_bonus),
        "final_reward": final_reward,
        "evaluation": evaluation,
    }
