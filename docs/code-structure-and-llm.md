# Code Structure and LLM Usage

This document explains the current RestBench agent codebase at a high level:

- how the main files are organized
- how a daily decision is produced
- how `agents.evaluate` runs games through the RestBench API
- where LLM calls happen
- how to run the agent with or without LLM mode

## 1. Main Entry Points

### `agents/team_agent.py`

This is the team's main agent entrypoint.

It supports two usage modes:

```powershell
python -m agents.team_agent --scenario baseline --seed 7
```

and the starter-kit evaluation interface:

```powershell
python -m agents.evaluate agents.team_agent --scenarios baseline --seeds 7
```

Important functions:

- `main()` is used when running `python -m agents.team_agent`.
- `strategy(observation, day)` is used by `agents.evaluate`.
- `configure_strategy(use_llm=True)` lets `agents.evaluate --llm` enable the LLM regime supervisor.
- `_prepare_llm_env()` checks that an LLM API key exists before LLM mode runs.

The default team name is read from:

```python
os.environ.get("RESTBENCH_TEAM", "pppp")
```

So `RESTBENCH_TEAM` can override it.

### `agents/evaluate.py`

This is the multi-scenario evaluation harness.

It:

1. imports a module such as `agents.team_agent`
2. loads its `strategy()` function
3. fetches scenarios from `/scenarios` unless scenarios are passed explicitly
4. runs every `(scenario, seed)` pair
5. prints a per-scenario and average-score report

Useful commands:

```powershell
python -m agents.evaluate agents.team_agent --scenarios baseline --seeds 7 --parallel 1
python -m agents.evaluate agents.team_agent --llm --scenarios baseline --seeds 7 --parallel 1
```

Current behavior:

- default API server: `http://52.48.183.209:8001`
- default non-LLM parallelism: `5`
- default LLM parallelism: `1`
- `--llm` calls `configure_strategy(use_llm=True)` if the agent supports it

### `agents/runner.py`

This file is the HTTP runner used by `agents.evaluate`.

It owns the game lifecycle:

1. `POST /games`
2. call local `strategy(observation, day)`
3. `POST /games/{game_id}/action` for each tool call
4. `POST /games/{game_id}/end-turn`
5. repeat until completed or bankrupt
6. `GET /games/{game_id}/score`

It also has `_request_json()`, which retries temporary API failures:

```python
RETRY_STATUSES = {429, 500, 502, 503, 504}
```

This is important because the RestBench server enforces rate limits.

## 2. Agent Decision Pipeline

The decision core is in:

```text
restbench/agent.py
```

The daily flow is:

```text
Observation
  -> BeliefEstimator.update()
  -> RegimeSupervisor.decide()
  -> SupplyController.propose()
  -> OperationsController.propose()
  -> PricingController.propose()
  -> Coordinator.compose()
  -> SafetyGate.filter()
  -> final actions
```

### `restbench/belief.py`

Maintains estimated state from observations:

- ingredient daily usage
- supplier reliability
- reputation estimate
- capacity-reduction memory
- demand and service signals

This lets the agent use cross-day information instead of reacting only to the current observation.

### `restbench/controllers/supply.py`

Responsible for replenishment.

It looks at:

- usable inventory
- pending orders
- ingredient usage estimates
- supplier lead time
- supplier delivery days
- shelf-life planning caps
- structural stockout risk

It proposes `place_order` actions.

### `restbench/controllers/operations.py`

Responsible for staffing.

It looks at:

- forecast covers
- staff throughput
- weekend effects
- walkouts and wait pressure

It proposes `set_staff_level`.

### `restbench/controllers/pricing.py`

Responsible for:

- menu selection
- price multiplier
- marketing spend
- happy hour
- daily special

It now filters the menu toward serviceable dishes when ingredients are tight. This helps avoid a wide menu that appears active but has no actually cookable dishes.

### `restbench/composer.py`

Combines controller proposals into one ordered action list.

It handles conflicts such as:

- only one staff setting per day
- only one marketing setting per day
- one order per `(supplier, ingredient)`
- stable action ordering before API submission

### `restbench/safety.py`

Final deterministic safety layer.

It:

- clamps illegal staff / marketing / price values
- rejects invalid menus
- removes duplicate pending orders
- enforces a cash reserve and order budget

This layer is deliberately not LLM-based.

## 3. Dashboard and README Page

### `dashboard_server.py`

Runs a small local HTTP server.

Routes:

- `/` serves the replay dashboard
- `/research-readme` serves the explanation page
- `/api/runs` lists replay files
- `/api/run` loads one replay file

Start it with:

```powershell
python dashboard_server.py
```

Then open:

```text
http://127.0.0.1:8765
http://127.0.0.1:8765/research-readme
```

### `dashboard/index.html`

Shows replay trends:

- cash
- final score
- covers and revenue
- walkouts and wait pressure
- inventory risk
- cost stack
- staff, menu, pending order quantity
- alerts and stockouts

### `dashboard/research-readme.html`

A simple team-facing explanation of the multi-agent + auto-research idea.

It is based on the Obsidian brief:

```text
D:\10 Work\Obsidian Vault\10 有明确目标、且近期会推进的事项\20260518 prosus hackthon\brief-idea-multi-agent-auto-research.md
```

## 4. Where LLM Is Called

There are two LLM-related code paths.

### Production path: `restbench/regime.py`

The main team agent only uses LLM through `LLMRegime`.

This is the intended LLM integration:

```python
r = litellm.completion(
    model=self.model,
    max_tokens=200,
    messages=[{"role": "user", "content": prompt}],
)
```

The LLM does not directly place orders or set prices.

It only chooses a high-level mode and optional parameter overrides:

```text
normal
supply_defensive
demand_surge
demand_slump
endgame
```

Allowed override keys are requested in the prompt:

```text
target_days
reliability_inflation
base_price_mult
marketing_slump
```

This design keeps risky operational logic deterministic while still letting the system adapt at a strategic level.

### Template path: `agents/llm_template.py`

This is a starter-kit style template.

It sends the full observation to the LLM and asks for raw tool calls.

This file is useful as an example, but it is not the main team-agent architecture.

The main team agent is more controlled because it keeps orders, cash safety, and menu validity in deterministic code.

## 5. How LLM Mode Is Enabled

### Single game

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:AGENT_MODEL="openai/gpt-4.1-mini"
python -m agents.team_agent --scenario baseline --seed 7 --llm
```

or:

```powershell
$env:ANTHROPIC_API_KEY="..."
$env:AGENT_MODEL="anthropic/claude-haiku-4-5"
python -m agents.team_agent --scenario baseline --seed 7 --llm
```

### Evaluation

```powershell
python -m agents.evaluate agents.team_agent --llm --scenarios baseline --seeds 7 --parallel 1
```

The `--llm` flag calls:

```python
configure_strategy(use_llm=True)
```

which prepares the LLM environment and makes the evaluate-compatible `strategy()` use `LLMRegime`.

## 6. LiteLLM Logging

LiteLLM can print noisy provider hints.

The helper below suppresses those logs:

```text
restbench/llm_silence.py
```

It is used by:

- `restbench/regime.py`
- `agents/llm_template.py`

## 7. Recommended Commands

### Fast baseline smoke test

```powershell
python -m agents.evaluate agents.team_agent --scenarios baseline --seeds 7 --parallel 1
```

### LLM smoke test

```powershell
python -m agents.evaluate agents.team_agent --llm --scenarios baseline --seeds 7 --parallel 1
```

### Multi-scenario non-LLM test

```powershell
python -m agents.evaluate agents.team_agent --scenarios baseline,supply_crisis,tourist_season --seeds 7 --parallel 3
```

### Local dashboard

```powershell
python dashboard_server.py
```

Then open:

```text
http://127.0.0.1:8765
```

## 8. Design Rationale

The code intentionally separates three responsibilities:

1. deterministic mechanics for safety-critical decisions
2. evaluation infrastructure for reproducible server runs
3. optional LLM supervision for high-level adaptation

This is safer than asking an LLM to directly produce every action because:

- the API action format is strict
- invalid orders or menus waste turns
- cash reserve violations can cause catastrophic failure
- delivery-calendar arithmetic should be deterministic
- rate limits and retries need predictable infrastructure

The LLM is useful as a strategic supervisor, but the execution layer remains code-first.
