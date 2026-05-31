# scienceflow-sft

Lightweight ScienceFlow pipeline for generating visually grounded SFT QA data from scientific figures.

The pipeline runs five stages:

1. `extract_evidence` extracts visual units and figure-level control signals.
2. `induce_rrels` induces relational reasoning links from the extracted evidence.
3. `compose_path` builds a reasoning path and a boundary-aware `question_spec`.
4. `generate_sft_qa` generates a question-answer pair under the `question_spec`.
5. `judge_quality` scores the QA and applies the final export gate.

Only attempts that pass both quality scoring and boundary checking are marked exportable.

## Repository Layout

```text
configs/                 Runtime configuration template
domain/                  Shared data models
operators/               Pipeline operators
prompts/                 Prompt builders for each stage
runner/                  End-to-end orchestration
tests/                   Contract and guardrail tests
tools/                   Utility scripts
run_scienceflow_sft.py   CLI entrypoint
```

Local run artifacts are intentionally excluded from git:

```text
runs/
codex.md
MIGRATION_GUIDE.md
__pycache__/
```

## Configuration

The default config is `configs/default.yaml`.

Required environment variables:

```bash
export LLM_API_KEY="..."
```

Optional environment variables:

```bash
export LLM_BASE_URL="https://coding.dashscope.aliyuncs.com/v1"
export LLM_MODEL="qwen3.5-plus"
export JUDGE_API_KEY="..."
export JUDGE_BASE_URL="https://coding.dashscope.aliyuncs.com/v1"
export JUDGE_MODEL="qwen3.5-plus"
```

If `JUDGE_API_KEY` is not set, the runner can fall back to the main LLM key without persisting resolved secrets in run metadata.

## Usage

Run one sample:

```bash
python3 run_scienceflow_sft.py run sample_000
```

Resume an existing run:

```bash
python3 run_scienceflow_sft.py resume sample_000_YYYYMMDD_HHMMSS sample_000
```

Run a batch:

```bash
python3 run_scienceflow_sft.py batch sample_000 sample_001 sample_002 -w 4
```

## Tests

```bash
python3 -m unittest discover -s tests
```

## Secret Hygiene

Before sharing any run artifacts outside the local workspace, redact stored metadata:

```bash
python3 tools/redact_run_secrets.py runs
```
