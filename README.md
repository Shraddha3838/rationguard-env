# RationGuardEnv

**RationGuardEnv** is a deterministic OpenEnv-compatible simulation for fraud detection in India’s Public Distribution System (PDS), a system where leakage has been estimated at **~₹40,000 crore** in some analyses.  
The environment is designed for multi-step AI-agent reasoning: the agent must request evidence, update suspicion, and then make a final fraud decision.

---

## Why this environment matters

PDS fraud is a real governance problem with multiple failure points:

- **Beneficiary fraud**: inflated claims versus entitlement
- **Dealer fraud**: ledger-level mismatch between recorded and available stock
- **Supply chain fraud**: divergence between transported and distributed stock

RationGuardEnv converts this into a clean, deterministic benchmark suitable for reliable agent evaluation.

---

## OpenEnv compliance

The environment implements the required interface:

- `reset(level) -> observation`
- `step(action) -> (observation, reward, done, info)`
- `state() -> current_state`

Implementation file: `env/ration_env.py`

---

## Environment design

### Observation space (structured JSON-like dict)

Each step returns fields including:

- `task_level`, `task_id`
- `step_number`, `max_steps`
- `claimed_quantity`, `expected_quota`
- `dealer_stock`, `transported_stock`
- `suspicion_score` (0.0 to 1.0)
- `revealed_checks` (which investigations have been run)
- `revealed_indicators` (numeric evidence values)
- `allowed_actions`
- `last_action`

### Action space

Investigation actions:

- `REQUEST_BENEFICIARY_AUDIT`
- `REQUEST_DEALER_LEDGER`
- `REQUEST_TRANSPORT_LOG`

Final decision actions:

- `FLAG_BENEFICIARY_FRAUD`
- `FLAG_DEALER_FRAUD`
- `FLAG_SUPPLY_CHAIN_FRAUD`
- `CLEAR_CLAIM`

### Multi-step transitions

- Suspicion begins at a deterministic baseline.
- Investigation actions reveal new indicators and increase suspicion.
- Episode ends on final decision or max-step timeout.
- No randomness is used anywhere.

---

## Reward logic (all rewards in [0, 1])

Reward shaping is deterministic and interpretable:

- **Partial rewards** for useful investigations
- **Penalties** for repeated/irrelevant/invalid actions
- **Final-decision reward** for correct fraud classification
- **Evidence bonus** based on required investigation coverage
- **Early bonus** for correct early final decision

Every step returns a `reward_breakdown` in `info`:

- `base`
- `evidence_bonus`
- `penalty`
- `early_bonus`
- `final_reward`

---

## Task levels (easy → medium → hard)

All tasks are deterministic and stored in `env/tasks.py`.

1. **Easy (beneficiary fraud)**  
   Obvious over-claim mismatch; minimal evidence needed.

2. **Medium (dealer fraud)**  
   Subtle dealer-level mismatch; requires combining ledger + transport signals.

3. **Hard (supply chain fraud)**  
   Multi-variable anomaly with hidden pattern; requires broader evidence synthesis.

---

## Grading quality

Grading is fully rule-based in `env/grader.py`:

- Deterministic action-to-score mapping
- Partial correctness scoring
- No LLM-based grading ambiguity

---

## Project structure

```text
env/
  __init__.py
  models.py
  tasks.py
  grader.py
  ration_env.py
inference.py
openenv.yaml
Dockerfile
requirements.txt
README.md
```

---

## Run locally

```bash
python -m pip install -r requirements.txt
python inference.py
```

Inference script requirements satisfied:

- uses `OpenAI` client
- prints strict `[START]`, `[STEP]`, `[END]` log lines
- deterministic policy for reproducible baseline

Optional env vars:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` (or `API_KEY`)

---

## Run with Docker (Mac M1 + HF Spaces friendly)

```bash
docker build -t rationguard-env .
docker run --rm rationguard-env
```

Notes:

- Uses `python:3.11-slim` (multi-arch compatible, including ARM64 / Apple Silicon)
- Minimal dependencies for fast build/startup

---

## Determinism guarantee

- No random module usage
- Fixed task bank
- Fixed rule-based transitions and grader
- Same action sequence always gives same trajectory/reward

---

## Submission readiness checklist

- ✅ OpenEnv interface methods present (`reset`, `step`, `state`)
- ✅ 3 tasks with increasing difficulty
- ✅ Rewards normalized to [0, 1]
- ✅ Deterministic and reproducible
- ✅ Structured observations for LLM evaluator
- ✅ Dockerized baseline runner
