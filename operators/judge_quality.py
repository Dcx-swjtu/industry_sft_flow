"""judge_quality operator — LLM-as-judge for SFT training data quality."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

from dataflow.infra.model_router import ModelRouter
from dataflow.operators.common import invoke_prompt

from domain import (
    ComposePathResult,
    InduceRrelsResult,
    JudgeResult,
    SFTQAResult,
    SampleInput,
)
from prompts.judge_quality import JUDGE_QUALITY_PROMPT


def _build_figure_context(sample) -> dict:
    """Build lightweight figure context from sample metadata (replaces assess_figure)."""
    return {
        "title": sample.title or "",
        "caption": sample.caption or "",
        "raw_subject": sample.raw_subject or "",
    }

# Weights must match the prompt's rubric
_DIMENSION_WEIGHTS = {
    "factual_correctness": 0.25,
    "visual_grounding": 0.25,
    "reasoning_coherence": 0.20,
    "scientific_accuracy": 0.10,
    "answer_completeness": 0.10,
    "difficulty_appropriateness": 0.10,
}

_ACCEPT_THRESHOLD = 0.7
_DIMENSION_FLOOR = 0.4
_FACTUAL_CORRECTNESS_FLOOR = 0.8


def _validate_judge_contract(data: Dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(data.get("accepted"), bool):
        errors.append("accepted must be a boolean")
    if not isinstance(data.get("boundary_pass"), bool):
        errors.append("boundary_pass must be an explicit boolean")
    if not isinstance(data.get("boundary_violations"), list):
        errors.append("boundary_violations must be an array")

    scores = data.get("dimension_scores")
    if not isinstance(scores, dict):
        errors.append("dimension_scores must be an object")
        return errors
    for dimension in _DIMENSION_WEIGHTS:
        if dimension not in scores:
            errors.append(f"dimension_scores missing required dimension: {dimension}")
            continue
        value = scores[dimension]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"dimension_scores.{dimension} must be numeric")
            continue
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            errors.append(f"dimension_scores.{dimension} must be between 0 and 1")
    return errors


def run_judge_quality(
    sample: SampleInput,
    *,
    sft_qa: SFTQAResult,
    rrel_result: InduceRrelsResult,
    compose_result: ComposePathResult,
    router: ModelRouter,
    operator_dir: Path,
    prompt_snapshot_enabled: bool,
    accept_threshold: float = _ACCEPT_THRESHOLD,
    dimension_floor: float = _DIMENSION_FLOOR,
    factual_correctness_floor: float = _FACTUAL_CORRECTNESS_FLOOR,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> JudgeResult:
    payload: Dict[str, Any] = {
        "question": sft_qa.question,
        "answer": sft_qa.answer,
        "reasoning_path": compose_result.reasoning_path.to_dict(),
        "rrel_bank": [r.to_dict() for r in rrel_result.rrel_bank],
        "figure_context": _build_figure_context(sample),
        "question_spec": compose_result.question_spec,
    }

    raw = invoke_prompt(
        template=JUDGE_QUALITY_PROMPT,
        payload=payload,
        image_base64=sample.image_base64,
        operator_name="judge_quality",
        prompt_name="judge_quality",
        prompt_dir=operator_dir,
        router=router,
        prompt_snapshot_enabled=prompt_snapshot_enabled,
        max_retries=max_retries,
        base_delay=base_delay,
    )

    judge_data = raw.get("judge_output") or {}
    if not isinstance(judge_data, dict):
        judge_data = {}
    contract_errors = _validate_judge_contract(judge_data)
    result = JudgeResult.from_dict(judge_data)
    if contract_errors:
        result.accepted = False
        for error in contract_errors:
            reason = f"Judge output contract violation: {error}"
            if reason not in result.reject_reasons:
                result.reject_reasons.append(reason)

    # Post-processing: enforce thresholds
    result = _enforce_thresholds(
        result,
        accept_threshold,
        dimension_floor,
        factual_correctness_floor,
    )
    return result


def _enforce_thresholds(
    result: JudgeResult,
    accept_threshold: float,
    dimension_floor: float,
    factual_correctness_floor: float = _FACTUAL_CORRECTNESS_FLOOR,
) -> JudgeResult:
    """Enforce acceptance thresholds as a hard post-processing guard."""
    # Boundary gate - the judge must explicitly emit the check and pass it.
    if not result.boundary_checked:
        result.accepted = False
        reason = "Boundary check missing or malformed in judge output"
        if reason not in result.reject_reasons:
            result.reject_reasons.append(reason)
    elif not result.boundary_pass:
        result.accepted = False
        if "Answer violates question information boundary" not in result.reject_reasons:
            result.reject_reasons.append("Answer violates question information boundary")
        for v in result.boundary_violations[:3]:
            msg = f"Boundary violation: {v}"
            if msg not in result.reject_reasons:
                result.reject_reasons.append(msg)

    # Recompute overall_score from dimension_scores using canonical weights
    weighted_sum = 0.0
    for dim, weight in _DIMENSION_WEIGHTS.items():
        weighted_sum += weight * result.dimension_scores.get(dim, 0.0)

    result.overall_score = round(weighted_sum, 4)

    # Check floor violations
    floor_violations = [
        dim for dim in _DIMENSION_WEIGHTS
        if result.dimension_scores.get(dim, 0.0) < dimension_floor
    ]

    # Force rejection if thresholds not met
    if result.overall_score < accept_threshold:
        result.accepted = False
        if f"Overall score {result.overall_score} below threshold {accept_threshold}" not in result.reject_reasons:
            result.reject_reasons.append(
                f"Overall score {result.overall_score} below threshold {accept_threshold}"
            )

    if floor_violations:
        result.accepted = False
        for dim in floor_violations:
            msg = f"Dimension {dim} score {result.dimension_scores[dim]} below floor {dimension_floor}"
            if msg not in result.reject_reasons:
                result.reject_reasons.append(msg)

    factual_score = result.dimension_scores.get("factual_correctness", 0.0)
    if factual_score < factual_correctness_floor:
        result.accepted = False
        msg = (
            f"Dimension factual_correctness score {factual_score} "
            f"below critical floor {factual_correctness_floor}"
        )
        if msg not in result.reject_reasons:
            result.reject_reasons.append(msg)

    # Build feedback if rejected
    if not result.accepted and not result.feedback:
        result.feedback = "; ".join(result.reject_reasons)

    return result
