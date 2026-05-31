"""SFT pipeline prompts."""

from .compose_path import COMPOSE_PATH_PROMPT
from .extract_evidence import EXTRACT_EVIDENCE_PROMPT
from .generate_sft_qa import GENERATE_SFT_QA_PROMPT
from .induce_rrels import INDUCE_RRELS_PROMPT
from .judge_quality import JUDGE_QUALITY_PROMPT

__all__ = [
    "COMPOSE_PATH_PROMPT",
    "EXTRACT_EVIDENCE_PROMPT",
    "GENERATE_SFT_QA_PROMPT",
    "INDUCE_RRELS_PROMPT",
    "JUDGE_QUALITY_PROMPT",
]
