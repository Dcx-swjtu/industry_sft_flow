"""Runner for scienceflow-sft — SFT training data pipeline with generate→judge→retry loop."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dataflow.infra import ModelRouter, load_config, load_sample

from domain import (
    ComposePathResult,
    ExtractEvidenceResult,
    GenerateSFTQAResult,
    InduceRrelsResult,
    JudgeResult,
    SFTQAResult,
)
from operators.compose_path import normalize_and_validate_compose_result, run_compose_path
from operators.extract_evidence import run_extract_evidence
from operators.generate_sft_qa import run_generate_sft_qa
from operators.induce_rrels import normalize_and_validate_rrel_result, run_induce_rrels
from operators.judge_quality import run_judge_quality


SCIENCEFLOW_SFT_OPERATOR_DIRS = {
    "extract_evidence": "01_extract_evidence",
    "induce_rrels": "02_induce_rrels",
    "compose_path": "03_compose_path",
    "generate_sft_qa": "04_generate_sft_qa",
    "judge_quality": "05_judge_quality",
}


_SENSITIVE_CONFIG_KEYS = {"api_key", "token", "secret", "authorization", "password"}


def _redact_sensitive_values(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, child in value.items():
            normalized = str(key).strip().lower()
            is_secret = normalized in _SENSITIVE_CONFIG_KEYS or any(
                normalized.endswith(f"_{suffix}") for suffix in _SENSITIVE_CONFIG_KEYS
            )
            cleaned[key] = "***REDACTED***" if is_secret else _redact_sensitive_values(child)
        return cleaned
    if isinstance(value, list):
        return [_redact_sensitive_values(child) for child in value]
    return value


def _redact_exception_message(message: str) -> str:
    return re.sub(
        r"(?i)\b(api[_ -]?key|authorization|bearer|token|secret)\b(\s*[:=]\s*|\s+)\S+",
        r"\1\2***REDACTED***",
        message,
    )


def _is_hard_pass(judge_result: JudgeResult) -> bool:
    return (
        judge_result.accepted
        and judge_result.boundary_checked
        and judge_result.boundary_pass
    )


def _sample_metadata_snapshot(sample: Dict[str, Any], *, include_image_base64: bool) -> Dict[str, Any]:
    snapshot = dict(sample)
    image_base64 = snapshot.get("image_base64")
    if include_image_base64 or not image_base64:
        return snapshot
    snapshot["image_base64"] = None
    raw_record = dict(snapshot.get("raw_record") or {})
    raw_record["image_base64_omitted"] = True
    raw_record["image_base64_chars"] = len(str(image_base64))
    snapshot["raw_record"] = raw_record
    return snapshot


class ScienceflowSFTRunStore:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def create_run_dir(self, sample_id: str, run_id: Optional[str] = None) -> Path:
        resolved_run_id = run_id or f"{sample_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = self.root_dir / resolved_run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "00_meta").mkdir(parents=True, exist_ok=True)
        for rel in SCIENCEFLOW_SFT_OPERATOR_DIRS.values():
            (run_dir / rel).mkdir(parents=True, exist_ok=True)
        return run_dir

    def resolve_run_dir(self, run_id: str) -> Path:
        run_dir = self.root_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    def operator_dir(self, run_dir: Path, operator_name: str) -> Path:
        return run_dir / SCIENCEFLOW_SFT_OPERATOR_DIRS[operator_name]

    def save_json(self, path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if is_dataclass(payload):
            data = asdict(payload)
        elif hasattr(payload, "to_dict"):
            data = payload.to_dict()
        else:
            data = payload
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_json(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    def save_meta(self, run_dir: Path, *, sample: Dict[str, Any], config: Dict[str, Any], routing: Dict[str, Any]) -> None:
        meta_dir = run_dir / "00_meta"
        self.save_json(meta_dir / "sample.json", sample)
        self.save_json(meta_dir / "config.json", _redact_sensitive_values(config))
        self.save_json(meta_dir / "model_routing.json", routing)

    def save_failure(self, run_dir: Path, *, stage: str, exc: Exception) -> None:
        self.save_json(run_dir / "00_meta" / "failure.json", {
            "stage": stage,
            "exception_type": type(exc).__name__,
            "message": _redact_exception_message(str(exc))[:500],
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    def redact_existing_config_snapshot(self, run_dir: Path) -> None:
        path = run_dir / "00_meta" / "config.json"
        existing = self.load_json(path)
        if existing is not None:
            self.save_json(path, _redact_sensitive_values(existing))

    def detect_resume_point(self, run_dir: Path) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for op_name in ["extract_evidence", "induce_rrels", "compose_path"]:
            result[op_name] = self.load_json(self.operator_dir(run_dir, op_name) / "result.json")
        # Check for completed generate_judge loop
        final_path = run_dir / "04_generate_sft_qa" / "final_result.json"
        if final_path.exists():
            result["generate_judge_complete"] = self.load_json(final_path)
        return result


class ScienceflowSFTRunner:
    """Execute the scienceflow-sft pipeline with generate→judge→retry loop."""

    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.config_path = str(Path(config_path).resolve())
        self.config_dir = Path(self.config["_meta"]["config_dir"])
        self.data_dir = (self.config_dir / self.config["init"]["data_dir"]).resolve()
        self.output_root = (self.config_dir / self.config["run"]["output_dir"]).resolve()
        init_cfg = self.config.get("init", {})
        self.prompt_snapshot_enabled = init_cfg.get("prompt_snapshot", True)
        self.snapshot_image_base64 = bool(init_cfg.get("snapshot_image_base64", False))
        self.router = ModelRouter(self.config)
        self.run_store = ScienceflowSFTRunStore(str(self.output_root))
        self.operator_names = list(SCIENCEFLOW_SFT_OPERATOR_DIRS.keys())

        # API retry config for stages 1-4
        self.api_max_retries = int(init_cfg.get("api_max_retries", 3))
        self.api_base_delay = float(init_cfg.get("api_base_delay", 2.0))

        # Judge config
        judge_cfg = self.config.get("judge", {})
        self.max_retries = int(judge_cfg.get("max_retries", 2))
        self.accept_threshold = float(judge_cfg.get("accept_threshold", 0.7))
        self.dimension_floor = float(judge_cfg.get("dimension_floor", 0.4))
        self.factual_correctness_floor = float(judge_cfg.get("factual_correctness_floor", 0.8))
        generation_cfg = self.config.get("generation", {})
        self.local_generation_max_retries = int(generation_cfg.get("local_max_retries", 2))
        compose_cfg = self.config.get("compose", {})
        self.local_compose_max_retries = int(compose_cfg.get("local_max_retries", 2))

    def _routing_snapshot(self) -> Dict[str, Any]:
        snapshot: Dict[str, Any] = {}
        for operator_name in self.operator_names:
            runtime = self.router.resolve(operator_name)
            snapshot[operator_name] = {
                "base_url": runtime.profile.base_url,
                "model": runtime.profile.model,
                "stream": runtime.stream,
                "allow_nonstream_fallback": runtime.allow_nonstream_fallback,
                "temperature": runtime.temperature,
                "max_tokens": runtime.max_tokens,
                "enable_thinking": runtime.enable_thinking,
            }
        return snapshot

    def _judge_model_independent(self) -> bool:
        generator = self.router.resolve("generate_sft_qa").profile
        judge = self.router.resolve("judge_quality").profile
        return (generator.base_url, generator.model) != (judge.base_url, judge.model)

    def run(self, sample_id: str, *, run_id: Optional[str] = None, resume: Optional[bool] = None) -> Dict[str, Any]:
        sample = load_sample(str(self.data_dir), sample_id)
        use_resume = self.config["run"].get("resume", True) if resume is None else bool(resume)
        run_dir = self.run_store.resolve_run_dir(run_id) if run_id else self.run_store.create_run_dir(sample_id)
        current_stage = "metadata"
        try:
            meta_dir = run_dir / "00_meta"
            if not (meta_dir / "sample.json").exists():
                self.run_store.save_meta(
                    run_dir,
                    sample=_sample_metadata_snapshot(
                        sample.to_dict(),
                        include_image_base64=self.snapshot_image_base64,
                    ),
                    config=self.config,
                    routing=self._routing_snapshot(),
                )
            else:
                self.run_store.redact_existing_config_snapshot(run_dir)

            resume_state = self.run_store.detect_resume_point(run_dir) if use_resume else {}

            # Stage 1-3: evidence -> rrels -> compose
            current_stage = "extract_evidence"
            extract_result = self._load_or_run_extract(run_dir, sample, resume_state if use_resume else {})
            current_stage = "induce_rrels"
            rrel_result = self._load_or_run_rrels(run_dir, sample, extract_result, resume_state if use_resume else {})
            current_stage = "compose_path"
            compose_result = self._load_or_run_compose(
                run_dir,
                sample,
                rrel_result,
                resume_state if use_resume else {},
            )

            # Stage 4+5: generate -> judge -> retry loop
            current_stage = "generate_judge_loop"
            final_result = self._run_generate_judge_loop(
                run_dir,
                sample,
                rrel_result,
                compose_result,
                resume_state if use_resume else {},
            )

            return {
                "run_id": run_dir.name,
                "extract_output": extract_result.to_dict(),
                "rrel_output": rrel_result.to_dict(),
                "path_output": compose_result.to_dict(),
                "sft_qa": final_result.sft_qa.to_dict(),
                "judge_result": final_result.judge_result.to_dict() if final_result.judge_result else None,
                "retry_count": final_result.retry_count,
                "final_summary": final_result.final_summary,
                "export_to_sft": final_result.export_to_sft,
            }
        except Exception as exc:
            self.run_store.save_failure(run_dir, stage=current_stage, exc=exc)
            raise

    # ------------------------------------------------------------------
    # Stages 1-3: load-or-run with resume + retry
    # ------------------------------------------------------------------

    def _load_or_run_extract(
        self,
        run_dir: Path,
        sample,
        resume_state: Dict[str, Any],
    ) -> ExtractEvidenceResult:
        path = self.run_store.operator_dir(run_dir, "extract_evidence") / "result.json"
        if resume_state.get("extract_evidence"):
            return ExtractEvidenceResult.from_dict(resume_state["extract_evidence"])
        result = run_extract_evidence(
            sample,
            router=self.router,
            operator_dir=self.run_store.operator_dir(run_dir, "extract_evidence"),
            prompt_snapshot_enabled=self.prompt_snapshot_enabled,
            max_retries=self.api_max_retries,
            base_delay=self.api_base_delay,
        )
        self.run_store.save_json(path, result)
        return result

    def _load_or_run_rrels(
        self,
        run_dir: Path,
        sample,
        extract_result: ExtractEvidenceResult,
        resume_state: Dict[str, Any],
    ) -> InduceRrelsResult:
        path = self.run_store.operator_dir(run_dir, "induce_rrels") / "result.json"
        if resume_state.get("induce_rrels"):
            loaded = InduceRrelsResult.from_dict(resume_state["induce_rrels"])
            result = normalize_and_validate_rrel_result(loaded, extract_result)
            if result.to_dict() != resume_state["induce_rrels"]:
                self.run_store.save_json(path, result)
            return result
        result = run_induce_rrels(
            sample,
            extract_result=extract_result,
            router=self.router,
            operator_dir=self.run_store.operator_dir(run_dir, "induce_rrels"),
            prompt_snapshot_enabled=self.prompt_snapshot_enabled,
            max_retries=self.api_max_retries,
            base_delay=self.api_base_delay,
        )
        self.run_store.save_json(path, result)
        return result

    def _load_or_run_compose(
        self,
        run_dir: Path,
        sample,
        rrel_result: InduceRrelsResult,
        resume_state: Dict[str, Any],
    ) -> ComposePathResult:
        path = self.run_store.operator_dir(run_dir, "compose_path") / "result.json"
        if resume_state.get("compose_path"):
            loaded = ComposePathResult.from_dict(resume_state["compose_path"])
            try:
                result = normalize_and_validate_compose_result(loaded, rrel_result)
            except ValueError as exc:
                self.run_store.save_json(
                    self.run_store.operator_dir(run_dir, "compose_path") / "resume_validation_error.json",
                    {"message": str(exc), "action": "regenerate_compose_path"},
                )
                resume_state.pop("compose_path", None)
                resume_state.pop("generate_judge_complete", None)
            else:
                if result.to_dict() != resume_state["compose_path"]:
                    self.run_store.save_json(path, result)
                    # Prior QA/judge output was based on a weaker contract and must not be reused.
                    resume_state.pop("generate_judge_complete", None)
                return result
        result = run_compose_path(
            sample,
            rrel_result=rrel_result,
            router=self.router,
            operator_dir=self.run_store.operator_dir(run_dir, "compose_path"),
            prompt_snapshot_enabled=self.prompt_snapshot_enabled,
            max_retries=self.api_max_retries,
            base_delay=self.api_base_delay,
            local_max_retries=self.local_compose_max_retries,
        )
        self.run_store.save_json(path, result)
        return result

    # ------------------------------------------------------------------
    # Stage 4+5: generate→judge→retry loop
    # ------------------------------------------------------------------

    def _run_generate_judge_loop(
        self,
        run_dir: Path,
        sample,
        rrel_result: InduceRrelsResult,
        compose_result: ComposePathResult,
        resume_state: Dict[str, Any],
    ) -> GenerateSFTQAResult:
        # Check if already completed
        if resume_state.get("generate_judge_complete"):
            return GenerateSFTQAResult.from_dict(resume_state["generate_judge_complete"])

        gen_dir = self.run_store.operator_dir(run_dir, "generate_sft_qa")
        judge_dir = self.run_store.operator_dir(run_dir, "judge_quality")

        best_attempt: Optional[Dict[str, Any]] = None
        best_score = -1.0
        best_hard_pass_attempt: Optional[Dict[str, Any]] = None
        best_hard_pass_score = -1.0
        hard_pass_attempts: list = []
        accepted = False
        last_judge_result: Optional[JudgeResult] = None
        last_sft_qa: Optional[SFTQAResult] = None
        judge_feedback = ""

        for attempt in range(self.max_retries + 1):
            attempt_label = f"attempt_{attempt}"
            attempt_gen_dir = gen_dir / attempt_label
            attempt_judge_dir = judge_dir / attempt_label
            attempt_gen_dir.mkdir(parents=True, exist_ok=True)
            attempt_judge_dir.mkdir(parents=True, exist_ok=True)

            print(f"[scienceflow-sft] Generate attempt {attempt}, feedback={'yes' if judge_feedback else 'none'}")

            # Generate, reusing completed attempt output when resuming after judge/API failures.
            gen_result_path = attempt_gen_dir / "result.json"
            existing_gen_result = self.run_store.load_json(gen_result_path)
            if existing_gen_result:
                sft_qa = SFTQAResult.from_dict(existing_gen_result)
            else:
                sft_qa = run_generate_sft_qa(
                    sample,
                    rrel_result=rrel_result,
                    compose_result=compose_result,
                    router=self.router,
                    operator_dir=attempt_gen_dir,
                    prompt_snapshot_enabled=self.prompt_snapshot_enabled,
                    judge_feedback=judge_feedback,
                    max_retries=self.api_max_retries,
                    base_delay=self.api_base_delay,
                    local_max_retries=self.local_generation_max_retries,
                )
                self.run_store.save_json(gen_result_path, sft_qa)
            last_sft_qa = sft_qa

            # Judge (uses separate model profile), also resumable per attempt.
            judge_result_path = attempt_judge_dir / "result.json"
            existing_judge_result = self.run_store.load_json(judge_result_path)
            if existing_judge_result:
                judge_result = JudgeResult.from_dict(existing_judge_result)
            else:
                judge_result = run_judge_quality(
                    sample,
                    sft_qa=sft_qa,
                    rrel_result=rrel_result,
                    compose_result=compose_result,
                    router=self.router,
                    operator_dir=attempt_judge_dir,
                    prompt_snapshot_enabled=self.prompt_snapshot_enabled,
                    accept_threshold=self.accept_threshold,
                    dimension_floor=self.dimension_floor,
                    factual_correctness_floor=self.factual_correctness_floor,
                    max_retries=self.api_max_retries,
                    base_delay=self.api_base_delay,
                )
                self.run_store.save_json(judge_result_path, judge_result)
            last_judge_result = judge_result

            # Track best attempt
            attempt_data = {
                "attempt": attempt,
                "sft_qa": sft_qa.to_dict(),
                "judge_result": judge_result.to_dict(),
            }

            hard_pass = _is_hard_pass(judge_result)
            if hard_pass:
                hard_pass_attempts.append(attempt_data)
                if judge_result.overall_score > best_hard_pass_score:
                    best_hard_pass_score = judge_result.overall_score
                    best_hard_pass_attempt = attempt_data

            if judge_result.overall_score > best_score:
                best_score = judge_result.overall_score
                best_attempt = attempt_data

            if hard_pass:
                accepted = True
                print(f"[scienceflow-sft] Accepted at attempt {attempt} (score={judge_result.overall_score:.2f})")
                break
            else:
                judge_feedback = judge_result.feedback
                print(
                    f"[scienceflow-sft] Rejected at attempt {attempt} "
                    f"(score={judge_result.overall_score:.2f}): "
                    f"{'; '.join(judge_result.reject_reasons[:3])}"
                )

        # Prefer an explicit boundary-checked hard pass; preserve best score only for debug.
        export_to_sft = False
        if best_hard_pass_attempt:
            chosen = best_hard_pass_attempt
            export_to_sft = True
        elif best_attempt:
            chosen = best_attempt
            export_to_sft = False
        else:
            chosen = None

        if chosen:
            final_sft_qa = SFTQAResult.from_dict(chosen["sft_qa"])
            final_judge = JudgeResult.from_dict(chosen["judge_result"])
        else:
            final_sft_qa = last_sft_qa or SFTQAResult(question="", answer="")
            final_judge = last_judge_result

        # Count total attempts from saved files
        total_attempts = 0
        for i in range(self.max_retries + 1):
            if (gen_dir / f"attempt_{i}" / "result.json").exists():
                total_attempts = i + 1

        best_score_attempt_index = best_attempt["attempt"] if best_attempt else None
        chosen_attempt_index = chosen["attempt"] if chosen else None
        if export_to_sft:
            failure_mode = None
            chosen_reason = "hard_pass"
        elif final_judge and not final_judge.boundary_checked:
            failure_mode = "boundary_not_checked"
            chosen_reason = "best_score_debug_only"
        elif final_judge and not final_judge.boundary_pass:
            failure_mode = "boundary_failed"
            chosen_reason = "best_score_debug_only"
        elif chosen:
            failure_mode = "no_hard_pass"
            chosen_reason = "best_score_debug_only"
        else:
            failure_mode = "no_attempt_completed"
            chosen_reason = None

        final_summary = {
            "accepted": accepted,
            "best_score": best_score,
            "best_score_attempt_index": best_score_attempt_index,
            "best_hard_pass_score": best_hard_pass_score if best_hard_pass_attempt else None,
            "best_attempt_index": chosen_attempt_index,
            "chosen_attempt_index": chosen_attempt_index,
            "chosen_reason": chosen_reason,
            "total_attempts": total_attempts,
            "rejected_after_retries": not accepted,
            "accept_threshold": self.accept_threshold,
            "dimension_floor": self.dimension_floor,
            "factual_correctness_floor": self.factual_correctness_floor,
            "hard_pass_count": len(hard_pass_attempts),
            "export_to_sft": export_to_sft,
            "boundary_pass_in_best": chosen.get("judge_result", {}).get("boundary_pass") if chosen else None,
            "boundary_pass_in_chosen": final_judge.boundary_pass if final_judge else None,
            "boundary_checked_in_chosen": final_judge.boundary_checked if final_judge else None,
            "failure_mode": failure_mode,
            "judge_model_independent": self._judge_model_independent(),
        }

        final_result = GenerateSFTQAResult(
            sft_qa=final_sft_qa,
            judge_result=final_judge,
            retry_count=max(0, total_attempts - 1),
            final_summary=final_summary,
            export_to_sft=export_to_sft,
        )

        # Save final result at top level
        self.run_store.save_json(gen_dir / "final_result.json", final_result)
        self.run_store.save_json(judge_dir / "final_result.json", {
            "accepted": accepted,
            "best_score": best_score,
            "best_score_attempt_index": best_score_attempt_index,
            "best_hard_pass_score": best_hard_pass_score if best_hard_pass_attempt else None,
            "best_attempt_index": chosen_attempt_index,
            "chosen_attempt_index": chosen_attempt_index,
            "chosen_reason": chosen_reason,
            "total_attempts": total_attempts,
            "hard_pass_count": len(hard_pass_attempts),
            "export_to_sft": export_to_sft,
            "factual_correctness_floor": self.factual_correctness_floor,
            "failure_mode": failure_mode,
            "judge_model_independent": self._judge_model_independent(),
        })

        return final_result


def run_scienceflow_sft(
    sample_id: str,
    *,
    config_path: str,
    run_id: Optional[str] = None,
    resume: Optional[bool] = None,
) -> Dict[str, Any]:
    runner = ScienceflowSFTRunner(config_path)
    return runner.run(sample_id, run_id=run_id, resume=resume)
