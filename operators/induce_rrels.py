"""induce_rrels operator."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Set

from dataflow.infra.model_router import ModelRouter
from dataflow.operators.common import invoke_prompt

from domain import ExtractEvidenceResult, InduceRrelsResult, SampleInput
from prompts.induce_rrels import INDUCE_RRELS_PROMPT


def _build_figure_context(sample) -> dict:
    """Build lightweight figure context from sample metadata (replaces assess_figure)."""
    return {
        "title": sample.title or "",
        "caption": sample.caption or "",
        "raw_subject": sample.raw_subject or "",
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


TEXT_REQUIRED = "text_required"
IMAGE_PLUS_TEXT_CLARIFICATION = "image_plus_text_clarification"
ALLOWED_DEPENDENCY_LEVELS = {
    "image_sufficient",
    IMAGE_PLUS_TEXT_CLARIFICATION,
    TEXT_REQUIRED,
}
ALLOWED_TXT_REQUIRED = {"none", "caption", "context", "mixed"}
OBSERVATIONAL_ROOT_KINDS = {"read", "compare"}


def _path_ids(paths: Iterable[Dict[str, object]]) -> Set[str]:
    return {str(path.get("path_id") or "") for path in paths if str(path.get("path_id") or "")}


def _coerce_score(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value or "").strip().lower()
    if not text:
        return 0.0
    mapping = {
        "very high": 0.95,
        "high": 0.85,
        "medium": 0.65,
        "med": 0.65,
        "low": 0.35,
        "very low": 0.15,
    }
    if text in mapping:
        return mapping[text]
    try:
        return float(text)
    except ValueError:
        return 0.0


def _normalize_rrel_output_payload(payload: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(payload)

    bank = payload.get("rrel_bank") or []
    if isinstance(bank, list):
        normalized_bank = []
        for item in bank:
            if isinstance(item, dict):
                row = dict(item)
                row["score"] = _coerce_score(row.get("score"))
                support = row.get("support")
                if isinstance(support, dict):
                    normalized_support = dict(support)
                    if "extract_rel_ids" in normalized_support and "relation_ids" not in normalized_support:
                        normalized_support["relation_ids"] = normalized_support.get("extract_rel_ids")
                    for key in ("evidence_ids", "relation_ids", "extract_rel_ids", "control_targets"):
                        values = normalized_support.get(key)
                        if isinstance(values, list):
                            deduped = []
                            seen = set()
                            for value in values:
                                normalized_id = str(value or "").strip()
                                if normalized_id and normalized_id not in seen:
                                    seen.add(normalized_id)
                                    deduped.append(normalized_id)
                            normalized_support[key] = deduped
                    row["support"] = normalized_support
                normalized_bank.append(row)
            else:
                normalized_bank.append(item)
        normalized["rrel_bank"] = normalized_bank

    paths = payload.get("candidate_paths") or []
    if isinstance(paths, list):
        normalized_paths = []
        for item in paths:
            if isinstance(item, dict):
                row = dict(item)
                try:
                    row["current_depth"] = int(row.get("current_depth") or 0)
                except (TypeError, ValueError):
                    row["current_depth"] = 0
                normalized_paths.append(row)
            else:
                normalized_paths.append(item)
        normalized["candidate_paths"] = normalized_paths

    return normalized


def _normalize_claim_text(text: object) -> str:
    lowered = str(text or "").strip().lower()
    lowered = re.sub(r"\s+", " ", lowered)
    lowered = re.sub(r"[^a-z0-9\s]", "", lowered)
    return lowered.strip()


def _infer_dependency_level(item: Dict[str, object]) -> str:
    level = str(item.get("dependency_level") or "").strip()
    if level in ALLOWED_DEPENDENCY_LEVELS:
        return level
    txt_required = item.get("txt_required")
    txt_used = item.get("txt_used")
    if txt_required is True:
        return TEXT_REQUIRED
    if txt_used is True:
        return IMAGE_PLUS_TEXT_CLARIFICATION
    return "image_sufficient"


def _validate_rrel_output(
    result: InduceRrelsResult,
    extract_result: ExtractEvidenceResult,
) -> None:
    if not result.rrel_bank:
        raise ValueError("induce_rrels must emit a non-empty rrel_bank")
    if len(result.rrel_bank) > 6:
        raise ValueError(f"induce_rrels must not emit more than 6 rrels, got {len(result.rrel_bank)}")
    bank_ids = [asset.id for asset in result.rrel_bank if asset.id]
    if len(bank_ids) != len(set(bank_ids)):
        raise ValueError(f"induce_rrels rrel ids must be unique, got duplicates: {bank_ids}")
    normalized_claims = [_normalize_claim_text(asset.claim) for asset in result.rrel_bank if asset.claim.strip()]
    if len(normalized_claims) != len(set(normalized_claims)):
        raise ValueError("induce_rrels contains duplicated or near-duplicated claims")
    synthesis_count = sum(1 for asset in result.rrel_bank if asset.kind == "synthesis")
    if synthesis_count > 1:
        raise ValueError(f"induce_rrels must emit at most one synthesis rrel, got {synthesis_count}")
    candidate_path_ids = {path.path_id for path in result.candidate_paths if path.path_id}
    if len(candidate_path_ids) != len(result.candidate_paths):
        raise ValueError("induce_rrels candidate_path ids must be unique and non-empty")
    summary = result.induction_summary or {}
    best_path_id = str(summary.get("best_path_id") or "")
    summary_path_ids = _path_ids(summary.get("path_summaries") or [])
    if best_path_id and best_path_id not in candidate_path_ids:
        raise ValueError(f"induce_rrels best_path_id not found in candidate_paths: {best_path_id}")
    if summary_path_ids and not summary_path_ids.issubset(candidate_path_ids):
        raise ValueError(
            f"induce_rrels path_summaries reference unknown candidate paths: "
            f"candidate={sorted(candidate_path_ids)} summary={sorted(summary_path_ids)}"
        )
    if best_path_id and summary_path_ids and best_path_id not in summary_path_ids:
        raise ValueError(
            f"induce_rrels best_path_id must appear in path_summaries: {best_path_id}"
        )
    has_reasoning_asset = any(
        asset.kind in {"compare", "compute", "bridge", "boundary", "synthesis"}
        for asset in result.rrel_bank
    )
    if not has_reasoning_asset:
        raise ValueError("induce_rrels must emit at least one comparative or transformed reasoning rrel")
    evidence_ids = {
        str(item.get("id") or "")
        for item in extract_result.evidence_graph.vis
        if str(item.get("id") or "")
    }
    relation_ids = {
        str(item.get("id") or "")
        for item in extract_result.evidence_graph.rels
        if str(item.get("id") or "")
    }
    dependency_by_target = {
        str(item.get("target_id") or ""): _infer_dependency_level(item)
        for item in extract_result.control_layer.items
        if str(item.get("target_id") or "")
    }
    observational_root_ids = []
    for asset in result.rrel_bank:
        support = asset.support or {}
        supported_evidence = [str(x) for x in (support.get("evidence_ids") or []) if str(x)]
        supported_relations = [str(x) for x in (support.get("relation_ids") or []) if str(x)]
        missing_evidence = [item_id for item_id in supported_evidence if item_id not in evidence_ids]
        missing_relations = [item_id for item_id in supported_relations if item_id not in relation_ids]
        if missing_evidence:
            raise ValueError(
                f"induce_rrels asset {asset.id} references unknown evidence ids: {missing_evidence}"
            )
        if missing_relations:
            hint = ""
            if all(item_id.startswith("rrel_") for item_id in missing_relations):
                hint = " (looks like rrel ids were placed into support.relation_ids; use extract rel_* ids only)"
            raise ValueError(
                f"induce_rrels asset {asset.id} references unknown relation ids: {missing_relations}{hint}"
            )
        if asset.kind in OBSERVATIONAL_ROOT_KINDS and supported_evidence:
            observational_root_ids.append(asset.id)
        dependency_levels = {
            dependency_by_target[item_id]
            for item_id in [*supported_evidence, *supported_relations]
            if item_id in dependency_by_target
        }
        txt_required = str((asset.knowledge or {}).get("txt_required") or "none")
        if txt_required not in ALLOWED_TXT_REQUIRED:
            raise ValueError(
                f"induce_rrels asset {asset.id} has invalid knowledge.txt_required: {txt_required}"
            )
        source = str((asset.knowledge or {}).get("source") or "").strip()
        if TEXT_REQUIRED in dependency_levels:
            if txt_required == "none":
                raise ValueError(
                    f"induce_rrels asset {asset.id} hides text-required support behind knowledge.txt_required=none"
                )
            if not source:
                raise ValueError(
                    f"induce_rrels asset {asset.id} must provide knowledge.source for text-required support"
                )
    if not observational_root_ids:
        raise ValueError(
            "induce_rrels must emit at least one observational root: "
            "a read or compare rrel directly supported by visual evidence"
        )
    bank_id_set = set(bank_ids)
    for path in result.candidate_paths:
        if not path.path_id.strip():
            raise ValueError("induce_rrels candidate_path must have non-empty path_id")
        if len(path.step_rrel_ids) != len(set(path.step_rrel_ids)):
            raise ValueError(f"induce_rrels candidate_path {path.path_id} contains duplicate rrel ids")
        missing = [rrel_id for rrel_id in path.step_rrel_ids if rrel_id not in bank_id_set]
        if missing:
            raise ValueError(
                f"induce_rrels candidate_path {path.path_id} references unknown rrels: {missing}"
            )
        if path.current_depth != len(path.step_rrel_ids):
            raise ValueError(
                f"induce_rrels candidate_path {path.path_id} current_depth must match step_rrel_ids length"
            )


def _repair_rrel_contract(result: InduceRrelsResult) -> InduceRrelsResult:
    """Repair fields whose only valid value is derivable from emitted content."""
    summary = dict(result.induction_summary or {})
    repairs = list(summary.get("contract_repairs") or [])
    for path in result.candidate_paths:
        expected_depth = len(path.step_rrel_ids)
        if path.current_depth != expected_depth:
            repairs.append({
                "field": f"candidate_paths.{path.path_id}.current_depth",
                "from": path.current_depth,
                "to": expected_depth,
                "reason": "derived from step_rrel_ids length",
            })
            path.current_depth = expected_depth
    if repairs:
        summary["contract_repairs"] = repairs
    result.induction_summary = summary
    return result


def normalize_and_validate_rrel_result(
    result: InduceRrelsResult,
    extract_result: ExtractEvidenceResult,
) -> InduceRrelsResult:
    result = _repair_rrel_contract(result)
    _validate_rrel_output(result, extract_result)
    return result


def run_induce_rrels(
    sample: SampleInput,
    *,
    extract_result: ExtractEvidenceResult,
    router: ModelRouter,
    operator_dir: Path,
    prompt_snapshot_enabled: bool,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> InduceRrelsResult:
    raw = invoke_prompt(
        template=INDUCE_RRELS_PROMPT,
        payload={
            "figure_context": _build_figure_context(sample),
            "evidence_graph": extract_result.evidence_graph.to_dict(),
            "control_layer": extract_result.control_layer.to_dict(),
            "context": sample.context,
        },
        image_base64=sample.image_base64,
        operator_name="induce_rrels",
        prompt_name="induce_rrels",
        prompt_dir=operator_dir,
        router=router,
        prompt_snapshot_enabled=prompt_snapshot_enabled,
        max_retries=max_retries,
        base_delay=base_delay,
    )
    payload = _normalize_rrel_output_payload(raw.get("rrel_output") or {})
    try:
        result = InduceRrelsResult.from_dict(payload)
        result = normalize_and_validate_rrel_result(result, extract_result)
    except (TypeError, ValueError) as exc:
        _write_json(operator_dir / "local_invalid_result_0.json", payload)
        _write_json(operator_dir / "local_validation_0.json", {
            "hard_errors": [str(exc)],
            "local_attempt": 0,
        })
        raise
    _write_json(operator_dir / "local_validation_0.json", {
        "hard_errors": [],
        "local_attempt": 0,
        "contract_repairs": result.induction_summary.get("contract_repairs", []),
    })
    return result
