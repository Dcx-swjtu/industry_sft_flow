"""extract_evidence operator."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Set

from dataflow.infra.model_router import ModelRouter
from dataflow.operators.common import invoke_prompt

from domain import ExtractEvidenceResult, SampleInput
from prompts.extract_evidence import EXTRACT_EVIDENCE_PROMPT


def _build_figure_context(sample) -> dict:
    """Build lightweight figure context from sample metadata (replaces assess_figure)."""
    return {
        "title": sample.title or "",
        "caption": sample.caption or "",
        "raw_subject": sample.raw_subject or "",
    }


ALLOWED_DEPENDENCY_LEVELS = {
    "image_sufficient",
    "image_plus_text_clarification",
    "text_required",
}


def _collect_ids(items: Iterable[Dict[str, object]]) -> Set[str]:
    return {str(item.get("id") or "") for item in items if str(item.get("id") or "")}


def _find_duplicate_ids(items: Iterable[Dict[str, object]]) -> Set[str]:
    seen: Set[str] = set()
    duplicates: Set[str] = set()
    for item in items:
        item_id = str(item.get("id") or "")
        if not item_id:
            continue
        if item_id in seen:
            duplicates.add(item_id)
        seen.add(item_id)
    return duplicates


def _coerce_declared_count(value: object, field_name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"extract_evidence {field_name} must be an integer, got {value!r}") from exc


def _infer_dependency_level(item: Dict[str, object]) -> str:
    level = str(item.get("dependency_level") or "").strip()
    if level in ALLOWED_DEPENDENCY_LEVELS:
        return level
    txt_required = item.get("txt_required")
    txt_used = item.get("txt_used")
    if txt_required is True:
        return "text_required"
    if txt_used is True:
        return "image_plus_text_clarification"
    return "image_sufficient"


def _validate_extract_output(result: ExtractEvidenceResult) -> None:
    vis_duplicates = _find_duplicate_ids(result.evidence_graph.vis)
    rel_duplicates = _find_duplicate_ids(result.evidence_graph.rels)
    if vis_duplicates:
        raise ValueError(f"extract_evidence vis ids must be unique, got duplicates: {sorted(vis_duplicates)}")
    if rel_duplicates:
        raise ValueError(f"extract_evidence rel ids must be unique, got duplicates: {sorted(rel_duplicates)}")
    vis_ids = _collect_ids(result.evidence_graph.vis)
    rel_ids = _collect_ids(result.evidence_graph.rels)
    declared_vis_count = result.extraction_summary.get("vis_count")
    declared_rel_count = result.extraction_summary.get("rel_count")
    if declared_vis_count is not None and _coerce_declared_count(declared_vis_count, "vis_count") != len(result.evidence_graph.vis):
        raise ValueError(
            f"extract_evidence vis_count must match evidence_graph.vis length: "
            f"declared={declared_vis_count} actual={len(result.evidence_graph.vis)}"
        )
    if declared_rel_count is not None and _coerce_declared_count(declared_rel_count, "rel_count") != len(result.evidence_graph.rels):
        raise ValueError(
            f"extract_evidence rel_count must match evidence_graph.rels length: "
            f"declared={declared_rel_count} actual={len(result.evidence_graph.rels)}"
        )
    recommended_vis_ids = [
        str(x) for x in (result.extraction_summary.get("recommended_vis_ids") or []) if str(x)
    ]
    recommended_rel_ids = [
        str(x) for x in (result.extraction_summary.get("recommended_rel_ids") or []) if str(x)
    ]
    if len(recommended_vis_ids) != len(set(recommended_vis_ids)):
        raise ValueError("extract_evidence recommended_vis_ids must not contain duplicates")
    if len(recommended_rel_ids) != len(set(recommended_rel_ids)):
        raise ValueError("extract_evidence recommended_rel_ids must not contain duplicates")
    cleaned_vis_ids = [item_id for item_id in recommended_vis_ids if item_id in vis_ids]
    cleaned_rel_ids = [item_id for item_id in recommended_rel_ids if item_id in rel_ids]
    if cleaned_vis_ids != recommended_vis_ids:
        result.extraction_summary["recommended_vis_ids"] = cleaned_vis_ids
        dropped = sorted(set(recommended_vis_ids) - set(cleaned_vis_ids))
        notes = str(result.extraction_summary.get("notes") or "")
        suffix = f" [auto-cleaned invalid recommended_vis_ids: {', '.join(dropped)}]"
        result.extraction_summary["notes"] = f"{notes}{suffix}".strip()
    if cleaned_rel_ids != recommended_rel_ids:
        result.extraction_summary["recommended_rel_ids"] = cleaned_rel_ids
        dropped = sorted(set(recommended_rel_ids) - set(cleaned_rel_ids))
        notes = str(result.extraction_summary.get("notes") or "")
        suffix = f" [auto-cleaned invalid recommended_rel_ids: {', '.join(dropped)}]"
        result.extraction_summary["notes"] = f"{notes}{suffix}".strip()
    if result.evidence_graph.vis and not result.extraction_summary.get("recommended_vis_ids"):
        raise ValueError("extract_evidence must recommend at least one vis id when vis items exist")
    if result.evidence_graph.rels and not result.extraction_summary.get("recommended_rel_ids"):
        raise ValueError("extract_evidence must recommend at least one rel id when rel items exist")
    for item in result.control_layer.items:
        target_id = str(item.get("target_id") or "")
        if target_id and target_id not in vis_ids and target_id not in rel_ids:
            raise ValueError(f"extract_evidence control_layer target_id must reference a real vis/rel id: {target_id}")
        dependency_level = _infer_dependency_level(item)
        item["dependency_level"] = dependency_level
        if dependency_level == "image_sufficient":
            item["txt_used"] = bool(item.get("txt_used", False))
            item["txt_required"] = False
        elif dependency_level == "image_plus_text_clarification":
            item["txt_used"] = True
            item["txt_required"] = False
        else:
            item["txt_used"] = True
            item["txt_required"] = True
        txt_used = item.get("txt_used")
        txt_required = item.get("txt_required")
        if txt_used is not None and not isinstance(txt_used, bool):
            raise ValueError("extract_evidence control_layer.txt_used must be boolean when present")
        if txt_required is not None and not isinstance(txt_required, bool):
            raise ValueError("extract_evidence control_layer.txt_required must be boolean when present")
        if dependency_level == "text_required" and not item.get("txt_used_sources"):
            raise ValueError(
                "extract_evidence control_layer text_required items must provide txt_used_sources"
            )


def run_extract_evidence(
    sample: SampleInput,
    *,
    router: ModelRouter,
    operator_dir: Path,
    prompt_snapshot_enabled: bool,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> ExtractEvidenceResult:
    raw = invoke_prompt(
        template=EXTRACT_EVIDENCE_PROMPT,
        payload={
            "figure_context": _build_figure_context(sample),
            "caption": sample.caption,
            "raw_caption": sample.raw_caption,
            "context": sample.context,
            "title": sample.title,
            "subfigure_infos": sample.subfigure_infos,
            "raw_subject": sample.raw_subject,
        },
        image_base64=sample.image_base64,
        operator_name="extract_evidence",
        prompt_name="extract_evidence",
        prompt_dir=operator_dir,
        router=router,
        prompt_snapshot_enabled=prompt_snapshot_enabled,
        max_retries=max_retries,
        base_delay=base_delay,
    )
    result = ExtractEvidenceResult.from_dict(raw.get("extract_output") or {})
    _validate_extract_output(result)
    return result
