"""compose_path operator — fuse RREL assets into a pure reasoning spine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Set
import re

from dataflow.infra.model_router import ModelRouter
from dataflow.operators.common import invoke_prompt

from domain import ComposePathResult, InduceRrelsResult, SampleInput
from prompts.compose_path import COMPOSE_PATH_PROMPT


ALLOWED_STEP_KINDS = {
    "read",
    "compare",
    "compute",
    "bridge",
    "boundary",
    "synthesis",
}

ALLOWED_OBSERVABILITY = {
    "exact_labeled",
    "axis_estimated",
    "relational",
    "text_dependent",
}

ALLOWED_TXT_DEPENDENCY = {
    "none",
    "caption",
    "context",
    "mixed",
}

ALLOWED_SOURCE_MODES = {
    "image_only",
    "image_plus_caption",
    "image_plus_context",
}

DEPENDENCY_RANK = {
    "none": 0,
    "caption": 1,
    "context": 2,
    "mixed": 2,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

MINIMAL_GIVEN_BANNED_PATTERNS = (
    r"\d",
    r"%",
    r"\b(panel|figure|plot|chart|trace|bar|axis)\b",
    r"\b(increase|increased|decrease|decreased|stable|unchanged|higher|lower|greater|less|majority)\b",
    r"\b(shows|showed|observed|confirms|confirmed|suggests|suggested|implies|implied|demonstrates|demonstrated|indicates|indicated)\b",
    r"\b(ratio|burden|latency|amplitude|pressure|trend|compare|comparison)\b",
)

ANSWER_REVEALING_PREMISE_PATTERNS = (
    r"\b(balanced|evenly distributed|uniform)\b",
    r"\b(lacks?|lacking|without|no)\b[^.]{0,60}\b(hotspot|hotspots|cluster|clustering)\b",
)


def _validate_minimal_givens(minimal_givens: list[str]) -> None:
    if len(minimal_givens) > 3:
        print(f"[compose_path] Warning: minimal_givens has {len(minimal_givens)} items (max 3), truncating")
        del minimal_givens[3:]
    for item in minimal_givens:
        text = str(item or "").strip()
        if not text:
            raise ValueError("compose_path minimal_givens must not contain empty items")
        lowered = text.lower()
        is_identity_mapping = bool(
            re.search(r"\b(is|denotes|represents|corresponds to|refers to)\b|=", lowered)
        )
        for pattern_index, pattern in enumerate(MINIMAL_GIVEN_BANNED_PATTERNS):
            if is_identity_mapping and pattern_index in {0, 2}:
                # Panel/color-to-identity mappings may legitimately include labels or identifiers.
                continue
            if re.search(pattern, lowered):
                print(f"[compose_path] Warning: minimal_given may not be glossary-style: {text}")
                break


def _validate_non_revealing_premises(premises: list[str], *, field_name: str) -> None:
    for premise in premises:
        lowered = str(premise or "").strip().lower()
        for pattern in ANSWER_REVEALING_PREMISE_PATTERNS:
            if re.search(pattern, lowered):
                raise ValueError(
                    f"compose_path {field_name} contains an answer-revealing visual conclusion: {premise}"
                )


def _infer_observability(step) -> str:
    if step.observability:
        return step.observability
    claim = step.step_claim.lower()
    if step.txt_dependency != "none":
        return "text_dependent"
    if any(token in claim for token in ("~", "approximately", "approx", "about ", "roughly", "around ")):
        return "axis_estimated"
    if any(token in claim for token in ("read", "measure", "value", "latency", "reversal", "amplitude")):
        return "exact_labeled"
    return "relational"


def _infer_step_kind(step) -> str:
    if step.step_kind in ALLOWED_STEP_KINDS:
        return step.step_kind
    claim = step.step_claim.lower()
    legacy_kernel = str(getattr(step, "kernel", "") or "").strip()
    if legacy_kernel == "numeric_fill":
        if any(token in claim for token in ("difference", "increase by", "decrease by", "delta", "shift by")):
            return "compute"
        return "read"
    if any(token in claim for token in ("therefore", "together", "supports", "indicates that", "best explained by")):
        return "synthesis"
    if any(token in claim for token in ("higher than", "lower than", "greater than", "less than", "compared with", "versus")):
        return "compare"
    if any(token in claim for token in ("boundary", "preserved", "unchanged", "still", "selective", "not all")):
        return "boundary"
    if step.txt_dependency != "none":
        return "bridge"
    return "read"


def _contains_pseudo_precision(step_claim: str) -> bool:
    lowered = step_claim.lower()
    if "calibrat" in lowered or "reported in the" in lowered:
        return True
    return "using the reported" in lowered or "aligns with the reported" in lowered


def _normalize_compose_result(result: ComposePathResult) -> ComposePathResult:
    reasoning_path = result.reasoning_path
    compose_summary = dict(result.compose_summary or {})
    question_spec = dict(result.question_spec or {})
    used_rrel_ids = []
    seen_ids: Set[str] = set()
    step_sources: dict[str, list[str]] = {}
    for idx, step in enumerate(reasoning_path.reasoning_path, start=1):
        inferred: list[str] = []
        if not step.source_rrels and step.depends_on:
            inferred_seen: Set[str] = set()
            for dep in step.depends_on:
                for rrel_id in step_sources.get(dep, []):
                    normalized_id = str(rrel_id)
                    if normalized_id and normalized_id not in inferred_seen:
                        inferred_seen.add(normalized_id)
                        inferred.append(normalized_id)
        if inferred:
            step.source_rrels = inferred
        step.step_kind = _infer_step_kind(step)
        step.observability = _infer_observability(step)
        if not step.txt_dependency:
            step.txt_dependency = "none"
        normalized_sources = []
        normalized_seen: Set[str] = set()
        for rrel_id in step.source_rrels:
            rrel_id = str(rrel_id)
            if rrel_id and rrel_id not in normalized_seen:
                normalized_seen.add(rrel_id)
                normalized_sources.append(rrel_id)
            if rrel_id and rrel_id not in seen_ids:
                seen_ids.add(rrel_id)
                used_rrel_ids.append(rrel_id)
        step.source_rrels = normalized_sources
        step_sources[step.step_id] = normalized_sources

    current_selected = [str(x) for x in (compose_summary.get("selected_rrel_ids") or []) if str(x)]
    if used_rrel_ids and current_selected != used_rrel_ids:
        compose_summary["selected_rrel_ids"] = used_rrel_ids
        notes = str(compose_summary.get("notes") or "").strip()
        fix_note = "selected_rrel_ids normalized to the exact union of reasoning step source_rrels"
        compose_summary["notes"] = f"{notes} | {fix_note}" if notes else fix_note
    result.reasoning_path = reasoning_path
    result.compose_summary = compose_summary
    result.question_spec = question_spec
    return result


def _join_dependencies(values: list[str]) -> str:
    dependencies = {value for value in values if value and value != "none"}
    if not dependencies:
        return "none"
    if "mixed" in dependencies or {"caption", "context"}.issubset(dependencies):
        return "mixed"
    if "context" in dependencies:
        return "context"
    return "caption"


def _dependency_from_rrel(asset) -> str:
    dependency = str((asset.knowledge or {}).get("txt_required") or "none").strip().lower()
    if dependency not in ALLOWED_TXT_DEPENDENCY:
        raise ValueError(f"compose_path rrel {asset.id} has invalid knowledge.txt_required: {dependency}")
    return dependency


def _derive_source_mode(steps: list) -> str:
    dependencies = [step.txt_dependency for step in steps]
    if any(dependency in {"context", "mixed"} for dependency in dependencies):
        return "image_plus_context"
    if any(dependency == "caption" for dependency in dependencies):
        return "image_plus_caption"
    return "image_only"


def _repair_dependency_contract(result: ComposePathResult, rrel_result: InduceRrelsResult) -> ComposePathResult:
    bank = {asset.id: asset for asset in rrel_result.rrel_bank if asset.id}
    summary = dict(result.compose_summary or {})
    spec = dict(result.question_spec or {})
    repairs = list(summary.get("dependency_repairs") or [])

    for step in result.reasoning_path.reasoning_path:
        declared = str(step.txt_dependency or "none").strip().lower()
        if declared not in ALLOWED_TXT_DEPENDENCY:
            raise ValueError(
                f"compose_path step {step.step_id} has invalid txt_dependency before repair: {declared}"
            )
        source_dependencies = [
            _dependency_from_rrel(bank[rrel_id])
            for rrel_id in step.source_rrels
            if rrel_id in bank
        ]
        required = _join_dependencies(source_dependencies)
        needs_repair = (
            DEPENDENCY_RANK[declared] < DEPENDENCY_RANK[required]
            or (required == "mixed" and declared != "mixed")
        )
        if needs_repair:
            repairs.append({
                "field": "txt_dependency",
                "step_id": step.step_id,
                "from": declared,
                "to": required,
                "reason": "direct source_rrels require at least this text dependency",
            })
            step.txt_dependency = required
        if required in {"context", "mixed"} and step.observability != "text_dependent":
            repairs.append({
                "field": "observability",
                "step_id": step.step_id,
                "from": step.observability,
                "to": "text_dependent",
                "reason": "context-dependent source_rrels cannot remain purely visually observable",
            })
            step.observability = "text_dependent"

    derived_source_mode = _derive_source_mode(result.reasoning_path.reasoning_path)
    emitted_source_mode = str(spec.get("source_mode") or "").strip()
    if emitted_source_mode in ALLOWED_SOURCE_MODES and emitted_source_mode != derived_source_mode:
        repairs.append({
            "field": "question_spec.source_mode",
            "from": emitted_source_mode,
            "to": derived_source_mode,
            "reason": "derived from repaired reasoning step dependencies",
        })
        spec["source_mode"] = derived_source_mode

    if repairs:
        summary["dependency_repairs"] = repairs
    summary["derived_source_mode"] = derived_source_mode
    result.compose_summary = summary
    result.question_spec = spec
    return result


def _validate_question_spec(result: ComposePathResult) -> None:
    spec = result.question_spec or {}
    source_mode = str(spec.get("source_mode") or "").strip()
    if source_mode not in ALLOWED_SOURCE_MODES:
        raise ValueError(f"compose_path question_spec has invalid source_mode: {source_mode or '<missing>'}")
    for field_name in ("must_include", "must_not_use"):
        value = spec.get(field_name)
        if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError(f"compose_path question_spec.{field_name} must be a list of non-empty strings")

    derived_source_mode = _derive_source_mode(result.reasoning_path.reasoning_path)
    if source_mode != derived_source_mode:
        raise ValueError(
            "compose_path question_spec.source_mode does not match final reasoning step dependencies: "
            f"emitted={source_mode} derived={derived_source_mode}"
        )
    if source_mode != "image_only" and not spec["must_include"]:
        raise ValueError(
            "compose_path text-dependent paths must place required premises in question_spec.must_include"
        )
    _validate_non_revealing_premises(spec["must_include"], field_name="question_spec.must_include")
    result.compose_summary["question_spec_validated"] = True


def _validate_compose_output(result: ComposePathResult, rrel_result: InduceRrelsResult) -> None:
    reasoning_path = result.reasoning_path
    compose_summary = result.compose_summary or {}
    selected_path_id = str(compose_summary.get("selected_path_id") or "")
    selected_rrel_ids = {
        str(x) for x in (compose_summary.get("selected_rrel_ids") or []) if str(x)
    }
    candidate_path_ids = {path.path_id for path in rrel_result.candidate_paths if path.path_id}
    bank_ids = {asset.id for asset in rrel_result.rrel_bank if asset.id}

    if not reasoning_path.latent_target.strip():
        raise ValueError("compose_path must emit a non-empty latent_target")
    if not reasoning_path.reasoning.strip():
        raise ValueError("compose_path must emit a non-empty reasoning summary")
    if not reasoning_path.reasoning_path:
        raise ValueError("compose_path must emit a non-empty reasoning_path")
    if len(reasoning_path.reasoning_path) > 6:
        raise ValueError("compose_path must not emit more than 6 reasoning steps")
    if selected_path_id and selected_path_id not in candidate_path_ids:
        raise ValueError(f"compose_path selected_path_id not found in candidate_paths: {selected_path_id}")
    if not selected_rrel_ids:
        raise ValueError("compose_path must emit non-empty selected_rrel_ids")
    if not selected_rrel_ids.issubset(bank_ids):
        missing = sorted(selected_rrel_ids - bank_ids)
        raise ValueError(f"compose_path selected_rrel_ids not found in rrel_bank: {missing}")

    step_ids: Set[str] = set()
    used_rrel_ids: Set[str] = set()
    non_root_steps = 0
    prefinal_non_root_steps = 0
    for idx, step in enumerate(reasoning_path.reasoning_path, start=1):
        if not step.step_id.strip():
            raise ValueError(f"compose_path step {idx} has empty step_id")
        if step.step_id in step_ids:
            raise ValueError(f"compose_path step_id must be unique, got duplicate: {step.step_id}")
        step_ids.add(step.step_id)
        if not step.source_rrels:
            raise ValueError(f"compose_path step {step.step_id} must have non-empty source_rrels")
        source_rrels = {str(x) for x in step.source_rrels if str(x)}
        if not source_rrels.issubset(bank_ids):
            missing = sorted(source_rrels - bank_ids)
            raise ValueError(f"compose_path step {step.step_id} references unknown rrels: {missing}")
        used_rrel_ids.update(source_rrels)
        if not step.step_claim.strip():
            raise ValueError(f"compose_path step {step.step_id} must have a non-empty step_claim")
        if _contains_pseudo_precision(step.step_claim):
            raise ValueError(
                f"compose_path step {step.step_id} must not sharpen image evidence with reported/text-derived values"
            )
        if not step.step_kind.strip():
            raise ValueError(f"compose_path step {step.step_id} must have a non-empty step_kind")
        if step.step_kind not in ALLOWED_STEP_KINDS:
            raise ValueError(f"compose_path step {step.step_id} has invalid step_kind: {step.step_kind}")
        if not step.observability.strip():
            raise ValueError(f"compose_path step {step.step_id} must declare observability")
        if step.observability not in ALLOWED_OBSERVABILITY:
            raise ValueError(
                f"compose_path step {step.step_id} has invalid observability: {step.observability}"
            )
        if step.txt_dependency not in ALLOWED_TXT_DEPENDENCY:
            raise ValueError(
                f"compose_path step {step.step_id} has invalid txt_dependency: {step.txt_dependency}"
            )
        if any(dep not in step_ids for dep in step.depends_on):
            raise ValueError(
                f"compose_path step {step.step_id} depends on unknown or future steps: {step.depends_on}"
            )
        if step.depends_on:
            non_root_steps += 1
            if idx < len(reasoning_path.reasoning_path):
                prefinal_non_root_steps += 1

    if used_rrel_ids != selected_rrel_ids:
        raise ValueError(
            "compose_path selected_rrel_ids must exactly match the union of reasoning step source_rrels"
        )
    if len(reasoning_path.reasoning_path) > 1 and non_root_steps == 0:
        print("[compose_path] Warning: multi-step path has no real dependency edges (all roots)")
    if len(reasoning_path.reasoning_path) >= 4 and prefinal_non_root_steps == 0:
        print("[compose_path] Warning: no non-final dependency edges before terminal synthesis")
    if any(step.txt_dependency != "none" for step in reasoning_path.reasoning_path):
        if not reasoning_path.minimal_givens:
            raise ValueError(
                "compose_path must populate minimal_givens when any step has non-none txt_dependency"
            )
    if reasoning_path.minimal_givens:
        _validate_minimal_givens(reasoning_path.minimal_givens)
        _validate_non_revealing_premises(reasoning_path.minimal_givens, field_name="minimal_givens")
    _validate_question_spec(result)


def normalize_and_validate_compose_result(
    result: ComposePathResult,
    rrel_result: InduceRrelsResult,
) -> ComposePathResult:
    result = _normalize_compose_result(result)
    result = _repair_dependency_contract(result, rrel_result)
    _validate_compose_output(result, rrel_result)
    return result


def run_compose_path(
    sample: SampleInput,
    *,
    rrel_result: InduceRrelsResult,
    router: ModelRouter,
    operator_dir: Path,
    prompt_snapshot_enabled: bool,
    max_retries: int = 3,
    base_delay: float = 2.0,
    local_max_retries: int = 2,
) -> ComposePathResult:
    base_payload = {
        "rrel_bank": [x.to_dict() for x in rrel_result.rrel_bank],
        "candidate_paths": [x.to_dict() for x in rrel_result.candidate_paths],
        "induction_summary": rrel_result.induction_summary,
        "context": sample.context,
    }
    validation_feedback = ""
    for local_attempt in range(local_max_retries + 1):
        payload = dict(base_payload)
        payload["validation_feedback"] = validation_feedback
        prompt_dir = operator_dir if local_attempt == 0 else operator_dir / f"local_retry_{local_attempt}"
        raw = invoke_prompt(
            template=COMPOSE_PATH_PROMPT,
            payload=payload,
            image_base64=sample.image_base64,
            operator_name="compose_path",
            prompt_name="compose_path",
            prompt_dir=prompt_dir,
            router=router,
            prompt_snapshot_enabled=prompt_snapshot_enabled,
            max_retries=max_retries,
            base_delay=base_delay,
        )
        path_output = raw.get("path_output") or {}
        try:
            if not isinstance(path_output, dict):
                raise ValueError("compose_path path_output must be an object")
            result = ComposePathResult.from_dict(path_output)
            result = normalize_and_validate_compose_result(result, rrel_result)
        except (TypeError, ValueError) as exc:
            _write_json(operator_dir / f"local_invalid_result_{local_attempt}.json", path_output)
            _write_json(operator_dir / f"local_validation_{local_attempt}.json", {
                "hard_errors": [str(exc)],
                "local_attempt": local_attempt,
            })
            validation_feedback = (
                "Previous compose output failed deterministic validation. Correct the next output: "
                + str(exc)
            )
            continue
        result.compose_summary = dict(result.compose_summary or {})
        result.compose_summary["local_validation_pass"] = True
        result.compose_summary["local_regeneration_count"] = local_attempt
        _write_json(operator_dir / f"local_validation_{local_attempt}.json", {
            "hard_errors": [],
            "local_attempt": local_attempt,
        })
        return result

    raise ValueError(
        "compose_path failed local hard validation after "
        f"{local_max_retries + 1} generations"
    )
