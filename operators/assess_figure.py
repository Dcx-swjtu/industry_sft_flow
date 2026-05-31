"""assess_figure operator."""

from __future__ import annotations

from pathlib import Path

from dataflow.infra.model_router import ModelRouter
from dataflow.operators.common import invoke_prompt

from domain import AssessFigureResult, SampleInput
from prompts.assess_figure import ASSESS_FIGURE_PROMPT


def run_assess_figure(
    sample: SampleInput,
    *,
    router: ModelRouter,
    operator_dir: Path,
    prompt_snapshot_enabled: bool,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> AssessFigureResult:
    raw = invoke_prompt(
        template=ASSESS_FIGURE_PROMPT,
        payload={
            "caption": sample.caption,
            "raw_caption": sample.raw_caption,
            "context": sample.context,
            "title": sample.title,
            "subfigure_infos": sample.subfigure_infos,
            "raw_subject": sample.raw_subject,
        },
        image_base64=sample.image_base64,
        operator_name="assess_figure",
        prompt_name="assess_figure",
        prompt_dir=operator_dir,
        router=router,
        prompt_snapshot_enabled=prompt_snapshot_enabled,
        max_retries=max_retries,
        base_delay=base_delay,
    )
    return AssessFigureResult.from_dict(raw.get("assess_output") or {})

