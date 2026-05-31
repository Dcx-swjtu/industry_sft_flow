"""generate_sft_qa operator — convert reasoning path into free-form Q&A for SFT."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

from dataflow.infra.model_router import ModelRouter
from dataflow.operators.common import invoke_prompt

from domain import ComposePathResult, InduceRrelsResult, SFTQAResult, SampleInput
from prompts.generate_sft_qa import GENERATE_SFT_QA_PROMPT


def _build_figure_context(sample) -> dict:
    """Build lightweight figure context from sample metadata (replaces assess_figure)."""
    return {
        "title": sample.title or "",
        "caption": sample.caption or "",
        "raw_subject": sample.raw_subject or "",
    }


_VISUAL_ANCHOR_PATTERNS = re.compile(
    r"(?:panel\s+[a-z]|bar\s|axis\s|label|scale|trace|plot|histogram|heatmap|"
    r"scatter|curve|arrow|line\s+graph|dot\s+plot|box\s+plot|violin|"
    r"[0-9]+[\.\d]*\s*(?:%|Å|nm|μm|mm|cm|m|kg|mg|ml|°C|K|pA|nA|μA|mV|V))",
    re.IGNORECASE,
)
_INTERNAL_ARTIFACT_PATTERNS = re.compile(r"\brrel_[a-z0-9_]+\b|\bS\d+\s*(?:->|=>)", re.IGNORECASE)
_STRUCTURAL_MODEL_CONTEXT_PATTERN = re.compile(
    r"\b(pocket|ligand|active[\s-]+site|conformation\w*|molecular[\s-]+model\w*)\b",
    re.IGNORECASE,
)
_STRONG_CAUSAL_PATTERN = re.compile(
    r"\b(force[sd]?|dictate[sd]?|necessitat(?:e|es|ed|ing))\b",
    re.IGNORECASE,
)
_PREMISE_STOPWORDS = {
    "a", "an", "and", "as", "are", "be", "for", "in", "is", "of", "the", "to",
    "that", "this", "these", "those", "refers", "represent", "represents",
}
_TEXT_DEPENDENCY_LEVEL_BY_SOURCE_MODE = {
    "image_only": "none",
    "image_plus_caption": "caption",
    "image_plus_context": "context",
}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _salient_tokens(text: str) -> set[str]:
    normalized = text.lower()
    normalized = re.sub(r"\bdefin(?:e|es|ed|ing|ition|itions)\b", "define", normalized)
    normalized = re.sub(r"\black(?:s|ed|ing)?\b", "lack", normalized)
    tokens = {token for token in re.findall(r"[a-z0-9]+", normalized)}
    return {token for token in tokens if token not in _PREMISE_STOPWORDS}


def _premise_is_covered(premise: str, question: str) -> bool:
    required = _salient_tokens(premise)
    return bool(required) and required.issubset(_salient_tokens(question))


def _apply_deterministic_generation_metadata(
    result: SFTQAResult,
    question_spec: Dict[str, Any],
) -> SFTQAResult:
    source_mode = str(question_spec.get("source_mode") or "").strip()
    dependency_level = _TEXT_DEPENDENCY_LEVEL_BY_SOURCE_MODE.get(source_mode)
    if dependency_level is not None:
        result.difficulty_signals = dict(result.difficulty_signals or {})
        result.difficulty_signals["text_dependency_level"] = dependency_level
        result.generation_summary = dict(result.generation_summary or {})
        result.generation_summary["text_dependency_level_derived_from_question_spec"] = True
    return result


def run_generate_sft_qa(
    sample: SampleInput,
    *,
    rrel_result: InduceRrelsResult,
    compose_result: ComposePathResult,
    router: ModelRouter,
    operator_dir: Path,
    prompt_snapshot_enabled: bool,
    judge_feedback: str = "",
    max_retries: int = 3,
    base_delay: float = 2.0,
    local_max_retries: int = 2,
) -> SFTQAResult:
    base_payload: Dict[str, Any] = {
        "rrel_bank": [r.to_dict() for r in rrel_result.rrel_bank],
        "reasoning_path": compose_result.reasoning_path.to_dict(),
        "figure_context": _build_figure_context(sample),
        "judge_feedback": judge_feedback,
        "question_spec": compose_result.question_spec,
    }

    local_feedback = judge_feedback
    for local_attempt in range(local_max_retries + 1):
        payload = dict(base_payload)
        payload["judge_feedback"] = local_feedback
        prompt_dir = operator_dir if local_attempt == 0 else operator_dir / f"local_retry_{local_attempt}"
        raw = invoke_prompt(
            template=GENERATE_SFT_QA_PROMPT,
            payload=payload,
            image_base64=sample.image_base64,
            operator_name="generate_sft_qa",
            prompt_name="generate_sft_qa",
            prompt_dir=prompt_dir,
            router=router,
            prompt_snapshot_enabled=prompt_snapshot_enabled,
            max_retries=max_retries,
            base_delay=base_delay,
        )

        sft_data = raw.get("sft_qa_output") or {}
        result = SFTQAResult.from_dict(sft_data)
        validation = _validate_sft_qa(result, compose_result.question_spec)
        validation["local_attempt"] = local_attempt
        _write_json(operator_dir / f"local_validation_{local_attempt}.json", validation)
        if not validation["hard_errors"]:
            result = _apply_deterministic_generation_metadata(result, compose_result.question_spec)
            result.generation_summary = dict(result.generation_summary or {})
            result.generation_summary["local_validation_pass"] = True
            result.generation_summary["local_regeneration_count"] = local_attempt
            return result

        _write_json(operator_dir / f"local_invalid_result_{local_attempt}.json", result.to_dict())
        local_feedback = (
            f"{judge_feedback}\n\n" if judge_feedback else ""
        ) + "Local generation validation failed. Revise the QA to fix all of these issues: " + "; ".join(
            validation["hard_errors"]
        )

    raise ValueError(
        "generate_sft_qa failed local hard validation after "
        f"{local_max_retries + 1} generations"
    )


def _validate_sft_qa(result: SFTQAResult, question_spec: Dict[str, Any]) -> Dict[str, list[str]]:
    """Validate deterministic requirements before spending a judge attempt."""
    hard_errors: list[str] = []
    soft_warnings: list[str] = []

    if not result.question or len(result.question.strip()) < 20:
        hard_errors.append("question is empty or shorter than 20 characters")

    if not result.answer or len(result.answer.strip()) < 50:
        hard_errors.append("answer is empty or shorter than 50 characters")

    if not _VISUAL_ANCHOR_PATTERNS.search(result.answer):
        hard_errors.append("answer lacks an explicit panel, label, plot, or visible-value anchor")

    for premise in question_spec.get("must_include") or []:
        if isinstance(premise, str) and premise.strip() and not _premise_is_covered(premise, result.question):
            hard_errors.append(f"question does not cover required premise: {premise}")

    forbidden_terms = question_spec.get("forbidden_terms") or []
    combined_text = f"{result.question}\n{result.answer}".lower()
    for term in forbidden_terms:
        if isinstance(term, str) and term.strip() and term.lower() in combined_text:
            hard_errors.append(f"question or answer contains forbidden literal term: {term}")

    if _INTERNAL_ARTIFACT_PATTERNS.search(f"{result.question}\n{result.answer}"):
        hard_errors.append("question or answer exposes internal reasoning identifiers")

    qa_text = f"{result.question}\n{result.answer}"
    if _STRUCTURAL_MODEL_CONTEXT_PATTERN.search(qa_text) and _STRONG_CAUSAL_PATTERN.search(qa_text):
        hard_errors.append(
            "structural modeling QA uses unsupported strong causal wording; "
            "prefer supports, is consistent with, or would favor"
        )

    if len(result.answer.split()) > 450:
        soft_warnings.append("answer is unusually long for a focused SFT response")

    if soft_warnings:
        print(f"[generate_sft_qa] Validation warnings: {'; '.join(soft_warnings)}")
    return {"hard_errors": hard_errors, "soft_warnings": soft_warnings}
