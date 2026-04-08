"""Deterministic task bank for easy/medium/hard episodes."""

from copy import deepcopy
from typing import Dict, List


_TASKS: Dict[str, Dict] = {
    "easy": {
        "task_id": "pds-easy-beneficiary",
        "task_level": "easy",
        "expected_quota": 20,
        "claimed_quantity": 44,
        "dealer_stock": 500,
        "transported_stock": 500,
        "ground_truth_fraud": "beneficiary",
        "expected_final_action": "FLAG_BENEFICIARY_FRAUD",
        "required_evidence_actions": ["REQUEST_BENEFICIARY_AUDIT"],
        "investigation_effects": {
            "REQUEST_BENEFICIARY_AUDIT": {
                "indicator_key": "beneficiary_gap_ratio",
                "indicator_value": 1.20,
                "suspicion_delta": 0.50,
            },
            "REQUEST_DEALER_LEDGER": {
                "indicator_key": "dealer_ledger_gap",
                "indicator_value": 0.00,
                "suspicion_delta": 0.05,
            },
            "REQUEST_TRANSPORT_LOG": {
                "indicator_key": "transport_loss_ratio",
                "indicator_value": 0.00,
                "suspicion_delta": 0.05,
            },
        },
    },
    "medium": {
        "task_id": "pds-medium-dealer",
        "task_level": "medium",
        "expected_quota": 25,
        "claimed_quantity": 25,
        "dealer_stock": 452,
        "transported_stock": 480,
        "ground_truth_fraud": "dealer",
        "expected_final_action": "FLAG_DEALER_FRAUD",
        "required_evidence_actions": ["REQUEST_DEALER_LEDGER", "REQUEST_TRANSPORT_LOG"],
        "investigation_effects": {
            "REQUEST_BENEFICIARY_AUDIT": {
                "indicator_key": "beneficiary_gap_ratio",
                "indicator_value": 0.00,
                "suspicion_delta": 0.05,
            },
            "REQUEST_DEALER_LEDGER": {
                "indicator_key": "dealer_ledger_gap",
                "indicator_value": 0.48,
                "suspicion_delta": 0.32,
            },
            "REQUEST_TRANSPORT_LOG": {
                "indicator_key": "transport_loss_ratio",
                "indicator_value": 0.06,
                "suspicion_delta": 0.20,
            },
        },
    },
    "hard": {
        "task_id": "pds-hard-supply-chain",
        "task_level": "hard",
        "expected_quota": 30,
        "claimed_quantity": 30,
        "dealer_stock": 468,
        "transported_stock": 495,
        "ground_truth_fraud": "supply_chain",
        "expected_final_action": "FLAG_SUPPLY_CHAIN_FRAUD",
        "required_evidence_actions": [
            "REQUEST_DEALER_LEDGER",
            "REQUEST_TRANSPORT_LOG",
            "REQUEST_BENEFICIARY_AUDIT",
        ],
        "investigation_effects": {
            "REQUEST_BENEFICIARY_AUDIT": {
                "indicator_key": "beneficiary_gap_ratio",
                "indicator_value": 0.00,
                "suspicion_delta": 0.08,
            },
            "REQUEST_DEALER_LEDGER": {
                "indicator_key": "dealer_ledger_gap",
                "indicator_value": 0.09,
                "suspicion_delta": 0.18,
            },
            "REQUEST_TRANSPORT_LOG": {
                "indicator_key": "transport_loss_ratio",
                "indicator_value": 0.27,
                "suspicion_delta": 0.42,
            },
        },
    },
    "no_fraud": {
        "task_id": "pds-edge-clean-claim",
        "task_level": "easy",
        "expected_quota": 24,
        "claimed_quantity": 24,
        "dealer_stock": 500,
        "transported_stock": 500,
        "ground_truth_fraud": None,
        "expected_final_action": "CLEAR_CLAIM",
        "required_evidence_actions": ["REQUEST_BENEFICIARY_AUDIT"],
        "investigation_effects": {
            "REQUEST_BENEFICIARY_AUDIT": {
                "indicator_key": "beneficiary_gap_ratio",
                "indicator_value": 0.00,
                "suspicion_delta": 0.02,
            },
            "REQUEST_DEALER_LEDGER": {
                "indicator_key": "dealer_ledger_gap",
                "indicator_value": 0.00,
                "suspicion_delta": 0.02,
            },
            "REQUEST_TRANSPORT_LOG": {
                "indicator_key": "transport_loss_ratio",
                "indicator_value": 0.00,
                "suspicion_delta": 0.02,
            },
        },
    },
}


def available_task_levels() -> List[str]:
    return ["easy", "medium", "hard"]


def supported_task_levels() -> List[str]:
    return ["easy", "medium", "hard", "no_fraud"]


def generate_task(level: str = "easy") -> Dict:
    if level not in _TASKS:
        raise ValueError(f"Unsupported task level '{level}'. Expected one of {supported_task_levels()}.")
    return deepcopy(_TASKS[level])
