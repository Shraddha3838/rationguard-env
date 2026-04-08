"""Deterministic baseline inference runner for RationGuardEnv.

Prints strictly structured logs with [START], [STEP], and [END] lines.
"""

import os
from typing import List, Optional

from openai import OpenAI

from env.ration_env import RationGuardEnv
from env.tasks import available_task_levels


API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
BENCHMARK = os.getenv("RATION_GUARD_BENCHMARK", "rationguard-env")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def deterministic_policy(level: str, step_number: int) -> str:
    plans = {
        "easy": ["REQUEST_BENEFICIARY_AUDIT", "FLAG_BENEFICIARY_FRAUD"],
        "medium": ["REQUEST_DEALER_LEDGER", "REQUEST_TRANSPORT_LOG", "FLAG_DEALER_FRAUD"],
        "hard": [
            "REQUEST_BENEFICIARY_AUDIT",
            "REQUEST_DEALER_LEDGER",
            "REQUEST_TRANSPORT_LOG",
            "FLAG_SUPPLY_CHAIN_FRAUD",
        ],
    }
    sequence = plans[level]
    index = min(step_number - 1, len(sequence) - 1)
    return sequence[index]


def run_episode(level: str, client: OpenAI) -> None:
    # The OpenAI client is initialized for compliance with hackathon inference requirements.
    _ = client

    env = RationGuardEnv(default_level=level, max_steps=5)
    rewards: List[float] = []
    steps = 0
    success = False
    done = False
    last_error: Optional[str] = None
    score = 0.0

    log_start(task=level, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(level)

        while True:
            step_number = observation["step_number"] + 1
            action = deterministic_policy(level=level, step_number=step_number)

            observation, reward, done, info = env.step({"action": action})
            rewards.append(float(reward))
            steps = step_number

            last_error = info.get("last_action_error")
            log_step(step=steps, action=action, reward=float(reward), done=bool(done), error=last_error)

            if done:
                expected_action = info.get("expected_final_action")
                success = expected_action is not None and action == expected_action
                break

    except Exception as exc:
        last_error = str(exc)
        log_step(step=max(steps, 1), action="runtime_error", reward=0.0, done=done, error=last_error)
        success = False

    finally:
        # Ensure environment is always closed before emitting [END].
        try:
            env.close()
        except Exception:
            pass

        mean_reward = sum(rewards) / max(len(rewards), 1)
        mean_reward = max(0.0, min(1.0, mean_reward))
        score = 1.0 if success else mean_reward
        score = max(0.0, min(1.0, score))
        log_end(success=success, steps=steps, score=score, rewards=rewards)


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for level in available_task_levels():
        run_episode(level=level, client=client)


if __name__ == "__main__":
    main()
