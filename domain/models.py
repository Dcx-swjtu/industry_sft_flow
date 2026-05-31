"""Dataclass models for the scienceflow-sft pipeline (SFT-focused, no RLVR artifacts)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional


def _coerce_dict(value: Any, *, fallback_key: str) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, list):
        return {fallback_key: value}
    if value is None:
        return {}
    return {fallback_key: value}


def _precision_to_observability(value: Any) -> str:
    text = str(value or "").strip().lower()
    mapping = {
        "explicit": "exact_labeled",
        "estimated": "axis_estimated",
    }
    return mapping.get(text, "")


def _coerce_finite_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


class DictSerializable:
    """Small mixin for explicit JSON-friendly state objects."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Shared base types
# ---------------------------------------------------------------------------

@dataclass
class SampleInput(DictSerializable):
    sample_id: str
    image_path: str
    image_base64: Optional[str]
    caption: str
    raw_caption: str
    context: List[Any]
    title: str
    raw_subject: List[Any]
    subfigure_infos: List[Any]
    raw_record: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SampleInput":
        return cls(
            sample_id=str(data.get("sample_id") or data.get("id") or ""),
            image_path=str(data.get("image_path") or ""),
            image_base64=data.get("image_base64"),
            caption=str(data.get("caption") or ""),
            raw_caption=str(data.get("raw_caption") or ""),
            context=list(data.get("context") or []),
            title=str(data.get("title") or ""),
            raw_subject=list(data.get("raw_subject") or []),
            subfigure_infos=list(data.get("subfigure_infos") or []),
            raw_record=dict(data.get("raw_record") or {}),
        )


@dataclass
class FigureProfile(DictSerializable):
    should_generate: bool
    why: str
    figure_family: str
    shortcut_risk: str
    visual_hinge_density: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FigureProfile":
        return cls(
            should_generate=bool(data.get("should_generate", False)),
            why=str(data.get("why") or ""),
            figure_family=str(data.get("figure_family") or ""),
            shortcut_risk=str(data.get("shortcut_risk") or ""),
            visual_hinge_density=str(data.get("visual_hinge_density") or ""),
        )


@dataclass
class EvidenceGraph(DictSerializable):
    vis: List[Dict[str, Any]] = field(default_factory=list)
    rels: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceGraph":
        return cls(
            vis=list(data.get("vis") or []),
            rels=list(data.get("rels") or []),
        )


@dataclass
class ControlLayer(DictSerializable):
    items: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Any) -> "ControlLayer":
        if isinstance(data, dict):
            items = list(data.get("items") or [])
        elif isinstance(data, list):
            items = list(data)
        else:
            items = []
        return cls(items=items)


@dataclass
class RRELAsset(DictSerializable):
    id: str
    claim: str
    support: Dict[str, Any]
    reasoning: str
    kind: str
    score: float
    knowledge: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RRELAsset":
        return cls(
            id=str(data.get("id") or ""),
            claim=str(data.get("claim") or ""),
            support=_coerce_dict(data.get("support"), fallback_key="items"),
            reasoning=str(data.get("reasoning") or ""),
            kind=str(data.get("kind") or ""),
            score=float(data.get("score") or 0.0),
            knowledge=_coerce_dict(data.get("knowledge"), fallback_key="content"),
        )


@dataclass
class PartialPath(DictSerializable):
    path_id: str
    step_rrel_ids: List[str]
    missing_hinges: List[str]
    current_depth: int
    shortcut_risk: str

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PartialPath":
        return cls(
            path_id=str(data.get("path_id") or ""),
            step_rrel_ids=[str(x) for x in (data.get("step_rrel_ids") or [])],
            missing_hinges=[str(x) for x in (data.get("missing_hinges") or [])],
            current_depth=int(data.get("current_depth") or 0),
            shortcut_risk=str(data.get("shortcut_risk") or ""),
        )


# ---------------------------------------------------------------------------
# ReasoningStep: compose_path output — pure reasoning spine (no kernel/answer_format)
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStep(DictSerializable):
    step_id: str
    source_rrels: List[str]
    step_claim: str
    depends_on: List[str]
    step_kind: str = ""            # "read" | "compare" | "compute" | "bridge" | "boundary" | "synthesis"
    observability: str = ""        # "exact_labeled" | "axis_estimated" | "relational" | "text_dependent"
    txt_dependency: str = "none"   # "none" | "caption" | "context" | "mixed"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningStep":
        return cls(
            step_id=str(data.get("step_id") or ""),
            source_rrels=[str(x) for x in (data.get("source_rrels") or [])],
            step_claim=str(data.get("step_claim") or ""),
            depends_on=[str(x) for x in (data.get("depends_on") or [])],
            step_kind=str(data.get("step_kind") or data.get("kind") or data.get("reasoning_kind") or ""),
            observability=str(
                data.get("observability")
                or _precision_to_observability(data.get("precision"))
                or ""
            ),
            txt_dependency=str(data.get("txt_dependency") or data.get("text_dependency") or "none"),
        )


@dataclass
class ReasoningPath(DictSerializable):
    latent_target: str
    minimal_givens: List[str]
    reasoning: str
    reasoning_path: List[ReasoningStep]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReasoningPath":
        return cls(
            latent_target=str(data.get("latent_target") or ""),
            minimal_givens=[str(x) for x in (data.get("minimal_givens") or [])],
            reasoning=str(data.get("reasoning") or ""),
            reasoning_path=[ReasoningStep.from_dict(x) for x in (data.get("reasoning_path") or [])],
        )


# ---------------------------------------------------------------------------
# Pipeline stage result wrappers (stages 1-4, shared with RLVR pipeline)
# ---------------------------------------------------------------------------

@dataclass
class AssessFigureResult(DictSerializable):
    figure_profile: FigureProfile
    admission_summary: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssessFigureResult":
        return cls(
            figure_profile=FigureProfile.from_dict(data.get("figure_profile") or {}),
            admission_summary=dict(data.get("admission_summary") or {}),
        )


@dataclass
class ExtractEvidenceResult(DictSerializable):
    evidence_graph: EvidenceGraph
    control_layer: ControlLayer
    extraction_summary: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtractEvidenceResult":
        return cls(
            evidence_graph=EvidenceGraph.from_dict(data.get("evidence_graph") or {}),
            control_layer=ControlLayer.from_dict(data.get("control_layer") or []),
            extraction_summary=dict(data.get("extraction_summary") or {}),
        )


@dataclass
class InduceRrelsResult(DictSerializable):
    rrel_bank: List[RRELAsset] = field(default_factory=list)
    candidate_paths: List[PartialPath] = field(default_factory=list)
    induction_summary: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InduceRrelsResult":
        return cls(
            rrel_bank=[RRELAsset.from_dict(x) for x in (data.get("rrel_bank") or [])],
            candidate_paths=[PartialPath.from_dict(x) for x in (data.get("candidate_paths") or [])],
            induction_summary=dict(data.get("induction_summary") or {}),
        )


@dataclass
class ComposePathResult(DictSerializable):
    reasoning_path: ReasoningPath = field(
        default_factory=lambda: ReasoningPath(
            latent_target="",
            minimal_givens=[],
            reasoning="",
            reasoning_path=[],
        )
    )
    compose_summary: Dict[str, Any] = field(default_factory=dict)
    question_spec: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComposePathResult":
        return cls(
            reasoning_path=ReasoningPath.from_dict(data.get("reasoning_path") or data.get("reasoning_path_output") or {}),
            compose_summary=dict(data.get("compose_summary") or {}),
            question_spec=dict(data.get("question_spec") or {}),
        )


# ---------------------------------------------------------------------------
# SFT-specific result types (stages 5-6)
# ---------------------------------------------------------------------------

@dataclass
class SFTQAResult(DictSerializable):
    question: str
    answer: str
    difficulty_signals: Dict[str, Any] = field(default_factory=dict)
    internal_path: Dict[str, Any] = field(default_factory=dict)
    generation_summary: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SFTQAResult":
        return cls(
            question=str(data.get("question") or ""),
            answer=str(data.get("answer") or ""),
            difficulty_signals=dict(data.get("difficulty_signals") or {}),
            internal_path=dict(data.get("internal_path") or {}),
            generation_summary=dict(data.get("generation_summary") or {}),
        )


@dataclass
class JudgeResult(DictSerializable):
    accepted: bool
    overall_score: float
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    feedback: str = ""
    reject_reasons: List[str] = field(default_factory=list)
    judge_summary: Dict[str, Any] = field(default_factory=dict)
    boundary_pass: bool = False
    boundary_checked: bool = False
    boundary_violations: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JudgeResult":
        raw_scores = data.get("dimension_scores")
        dimension_scores = raw_scores if isinstance(raw_scores, dict) else {}
        raw_violations = data.get("boundary_violations")
        boundary_violations = raw_violations if isinstance(raw_violations, list) else []
        raw_reasons = data.get("reject_reasons")
        reject_reasons = raw_reasons if isinstance(raw_reasons, list) else []
        raw_summary = data.get("judge_summary")
        judge_summary = raw_summary if isinstance(raw_summary, dict) else {}
        return cls(
            accepted=data.get("accepted") is True,
            overall_score=_coerce_finite_float(data.get("overall_score")),
            dimension_scores={
                str(k): _coerce_finite_float(v)
                for k, v in dimension_scores.items()
            },
            feedback=str(data.get("feedback") or ""),
            reject_reasons=[str(x) for x in reject_reasons],
            judge_summary=dict(judge_summary),
            boundary_pass=data.get("boundary_pass") is True,
            boundary_checked=isinstance(data.get("boundary_pass"), bool),
            boundary_violations=[str(x) for x in boundary_violations],
        )


@dataclass
class GenerateSFTQAResult(DictSerializable):
    sft_qa: SFTQAResult = field(default_factory=SFTQAResult)
    judge_result: Optional[JudgeResult] = None
    retry_count: int = 0
    final_summary: Dict[str, Any] = field(default_factory=dict)
    export_to_sft: bool = False

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerateSFTQAResult":
        judge_data = data.get("judge_result")
        return cls(
            sft_qa=SFTQAResult.from_dict(data.get("sft_qa") or {}),
            judge_result=JudgeResult.from_dict(judge_data) if judge_data else None,
            retry_count=int(data.get("retry_count") or 0),
            final_summary=dict(data.get("final_summary") or {}),
            export_to_sft=bool(data.get("export_to_sft", False)),
        )
