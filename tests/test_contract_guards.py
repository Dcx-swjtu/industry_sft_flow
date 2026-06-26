from __future__ import annotations

import json
import base64
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATAFLOW_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(DATAFLOW_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from domain import (  # noqa: E402
    ComposePathResult,
    ExtractEvidenceResult,
    InduceRrelsResult,
    JudgeResult,
    SFTQAResult,
    SampleInput,
)
from operators.compose_path import normalize_and_validate_compose_result, run_compose_path  # noqa: E402
from operators.generate_sft_qa import (  # noqa: E402
    _apply_deterministic_generation_metadata,
    _validate_sft_qa,
)
from operators.induce_rrels import _repair_rrel_contract, normalize_and_validate_rrel_result  # noqa: E402
from operators.judge_quality import _enforce_thresholds, _validate_judge_contract  # noqa: E402
from dataflow.infra.samples import load_sample  # noqa: E402
from dataflow.operators.common import (  # noqa: E402
    _clear_completion_url_cache,
    _completion_urls,
    _remember_completion_url,
    _snapshot_mode,
    invoke_prompt,
)
from runner.scienceflow_sft_runner import (  # noqa: E402
    ScienceflowSFTRunStore,
    _is_hard_pass,
    _sample_metadata_snapshot,
)
from run_scienceflow_sft import _resolve_worker_count  # noqa: E402


_TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADgQG"
    "BgeJ6WQAAAABJRU5ErkJggg=="
)


class JudgeContractTests(unittest.TestCase):
    def test_missing_boundary_pass_is_fail_closed(self) -> None:
        result = JudgeResult.from_dict({
            "accepted": True,
            "dimension_scores": {
                "factual_correctness": 1.0,
                "visual_grounding": 1.0,
                "reasoning_coherence": 1.0,
                "scientific_accuracy": 1.0,
                "answer_completeness": 1.0,
                "difficulty_appropriateness": 1.0,
            },
        })

        result = _enforce_thresholds(result, accept_threshold=0.7, dimension_floor=0.4)

        self.assertFalse(result.boundary_checked)
        self.assertFalse(result.boundary_pass)
        self.assertFalse(result.accepted)

    def test_missing_dimension_is_contract_error(self) -> None:
        errors = _validate_judge_contract({
            "accepted": True,
            "boundary_pass": True,
            "boundary_violations": [],
            "dimension_scores": {
                "factual_correctness": 0.9,
            },
        })

        self.assertTrue(any("visual_grounding" in error for error in errors))

    def test_export_gate_requires_explicit_boundary_check(self) -> None:
        unchecked = JudgeResult(accepted=True, overall_score=0.9, boundary_pass=True)
        checked = JudgeResult(
            accepted=True,
            overall_score=0.9,
            boundary_pass=True,
            boundary_checked=True,
        )

        self.assertFalse(_is_hard_pass(unchecked))
        self.assertTrue(_is_hard_pass(checked))

    def test_low_factual_correctness_is_rejected_even_when_overall_passes(self) -> None:
        result = JudgeResult(
            accepted=True,
            overall_score=0.8,
            boundary_pass=True,
            boundary_checked=True,
            dimension_scores={
                "factual_correctness": 0.75,
                "visual_grounding": 0.9,
                "reasoning_coherence": 0.9,
                "scientific_accuracy": 0.9,
                "answer_completeness": 0.9,
                "difficulty_appropriateness": 0.9,
            },
        )

        result = _enforce_thresholds(
            result,
            accept_threshold=0.7,
            dimension_floor=0.4,
            factual_correctness_floor=0.8,
        )

        self.assertFalse(result.accepted)
        self.assertTrue(any("critical floor" in reason for reason in result.reject_reasons))


class ComposeContractTests(unittest.TestCase):
    @staticmethod
    def _rrels() -> InduceRrelsResult:
        return InduceRrelsResult.from_dict({
            "rrel_bank": [{
                "id": "rrel_context",
                "claim": "A context dependent synthesis",
                "support": {},
                "reasoning": "reasoning",
                "kind": "synthesis",
                "score": 0.9,
                "knowledge": {"txt_required": "context"},
            }],
            "candidate_paths": [],
        })

    @staticmethod
    def _compose(must_include: list[str]) -> ComposePathResult:
        return ComposePathResult.from_dict({
            "reasoning_path": {
                "latent_target": "a valid target",
                "minimal_givens": ["Condition A denotes the treated cohort"],
                "reasoning": "S1",
                "reasoning_path": [{
                    "step_id": "S1",
                    "source_rrels": ["rrel_context"],
                    "step_claim": "The selected evidence supports the interpretation.",
                    "depends_on": [],
                    "step_kind": "synthesis",
                    "observability": "text_dependent",
                    "txt_dependency": "caption",
                }],
            },
            "compose_summary": {"selected_rrel_ids": ["rrel_context"]},
            "question_spec": {
                "source_mode": "image_plus_caption",
                "must_include": must_include,
                "must_not_use": [],
            },
        })

    def test_context_rrel_repairs_step_and_source_mode(self) -> None:
        result = normalize_and_validate_compose_result(
            self._compose(["Condition A denotes the treated cohort"]),
            self._rrels(),
        )

        self.assertEqual(result.reasoning_path.reasoning_path[0].txt_dependency, "context")
        self.assertEqual(result.reasoning_path.reasoning_path[0].observability, "text_dependent")
        self.assertEqual(result.question_spec["source_mode"], "image_plus_context")
        self.assertTrue(result.compose_summary["dependency_repairs"])

    def test_text_dependent_path_requires_question_premise(self) -> None:
        with self.assertRaises(ValueError):
            normalize_and_validate_compose_result(self._compose([]), self._rrels())

    def test_visual_conclusion_cannot_be_used_as_question_premise(self) -> None:
        with self.assertRaisesRegex(ValueError, "answer-revealing visual conclusion"):
            normalize_and_validate_compose_result(
                self._compose(["Panel MB.Q lacks notable mutational hotspots"]),
                self._rrels(),
            )

    def test_local_regeneration_recovers_missing_minimal_givens(self) -> None:
        invalid = self._compose(["Condition A denotes the treated cohort"]).to_dict()
        invalid["reasoning_path"]["minimal_givens"] = []
        valid = self._compose(["Condition A denotes the treated cohort"]).to_dict()
        sample = SampleInput.from_dict({"sample_id": "sample_test"})
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "operators.compose_path.invoke_prompt",
                side_effect=[{"path_output": invalid}, {"path_output": valid}],
            ) as invoke:
                result = run_compose_path(
                    sample,
                    rrel_result=self._rrels(),
                    router=object(),
                    operator_dir=Path(tmp_dir),
                    prompt_snapshot_enabled=False,
                    local_max_retries=1,
                )

            self.assertEqual(invoke.call_count, 2)
            self.assertEqual(result.compose_summary["local_regeneration_count"], 1)
            self.assertTrue((Path(tmp_dir) / "local_invalid_result_0.json").exists())
            self.assertTrue((Path(tmp_dir) / "local_validation_1.json").exists())


class RRELContractTests(unittest.TestCase):
    @staticmethod
    def _extract_result() -> ExtractEvidenceResult:
        return ExtractEvidenceResult.from_dict({
            "evidence_graph": {
                "vis": [{"id": "vis_1"}, {"id": "vis_2"}],
                "rels": [{"id": "rel_1"}],
            },
            "control_layer": [],
        })

    def test_current_depth_is_repaired_from_step_ids(self) -> None:
        result = InduceRrelsResult.from_dict({
            "candidate_paths": [{
                "path_id": "path_1",
                "step_rrel_ids": ["rrel_1", "rrel_2"],
                "current_depth": 1,
            }],
        })

        result = _repair_rrel_contract(result)

        self.assertEqual(result.candidate_paths[0].current_depth, 2)
        self.assertEqual(
            result.induction_summary["contract_repairs"][0]["field"],
            "candidate_paths.path_1.current_depth",
        )

    def test_direct_compare_can_serve_as_observational_root(self) -> None:
        result = InduceRrelsResult.from_dict({
            "rrel_bank": [
                {
                    "id": "rrel_1",
                    "kind": "compare",
                    "claim": "Panel a is dense whereas panel b is porous.",
                    "support": {"evidence_ids": ["vis_1", "vis_2"], "relation_ids": ["rel_1"]},
                    "knowledge": {"txt_required": "none"},
                },
                {
                    "id": "rrel_2",
                    "kind": "synthesis",
                    "claim": "The morphology is consistent with different accessible regions.",
                    "support": {"evidence_ids": ["vis_1", "vis_2"]},
                    "knowledge": {"txt_required": "none"},
                },
            ],
        })

        validated = normalize_and_validate_rrel_result(result, self._extract_result())

        self.assertEqual(validated.rrel_bank[0].kind, "compare")

    def test_rrel_bank_without_direct_observational_root_is_rejected(self) -> None:
        result = InduceRrelsResult.from_dict({
            "rrel_bank": [
                {
                    "id": "rrel_1",
                    "kind": "bridge",
                    "claim": "A textual interpretation.",
                    "support": {"evidence_ids": ["vis_1"]},
                    "knowledge": {"txt_required": "none"},
                },
                {
                    "id": "rrel_2",
                    "kind": "synthesis",
                    "claim": "A final interpretation.",
                    "support": {"evidence_ids": ["vis_2"]},
                    "knowledge": {"txt_required": "none"},
                },
            ],
        })

        with self.assertRaisesRegex(ValueError, "observational root"):
            normalize_and_validate_rrel_result(result, self._extract_result())


class GenerationContractTests(unittest.TestCase):
    def test_missing_required_premise_is_hard_error(self) -> None:
        validation = _validate_sft_qa(
            SFTQAResult(
                question="How do the visible shapes explain the observed comparison?",
                answer=(
                    "In panel a, the pocket is compact, while panel b shows an open pocket. "
                    "This visual contrast supports a different interaction pattern."
                ),
            ),
            {"must_include": ["The blue structure is KIVD"]},
        )

        self.assertTrue(any("required premise" in error for error in validation["hard_errors"]))

    def test_semantically_equivalent_definition_premise_passes(self) -> None:
        validation = _validate_sft_qa(
            SFTQAResult(
                question=(
                    "Given a founder mutation, defining it as the initial mutation present "
                    "in all subsequent subclones, which trajectory is supported?"
                ),
                answer=(
                    "In panel a, the labeled trajectory begins at the founder branch and "
                    "continues into the later subclones, supporting the requested comparison."
                ),
            ),
            {
                "must_include": [
                    "The definition of a founder mutation as the initial mutation present "
                    "in all subsequent subclones."
                ]
            },
        )

        self.assertFalse(validation["hard_errors"])

    def test_lack_word_form_variation_in_premise_passes(self) -> None:
        validation = _validate_sft_qa(
            SFTQAResult(
                question=(
                    "Given a baseline panel lacking notable mutational hotspots, "
                    "how do panels b and c differ from that reference?"
                ),
                answer=(
                    "Panel b shows a red cluster, while panel c shows green vertical "
                    "bands, providing visible contrasts to the baseline distribution."
                ),
            ),
            {"must_include": ["A baseline panel lacks notable mutational hotspots"]},
        )

        self.assertFalse(validation["hard_errors"])

    def test_text_dependency_level_is_derived_from_question_spec(self) -> None:
        result = _apply_deterministic_generation_metadata(
            SFTQAResult(
                question="How do panels a and b differ in the comparison?",
                answer=(
                    "Panel a shows a labeled first condition, while panel b shows the "
                    "second condition with a distinct plotted pattern."
                ),
                difficulty_signals={"text_dependency_level": "low"},
            ),
            {"source_mode": "image_plus_context"},
        )

        self.assertEqual(result.difficulty_signals["text_dependency_level"], "context")
        self.assertTrue(result.generation_summary["text_dependency_level_derived_from_question_spec"])

    def test_structural_model_strong_causal_language_is_hard_error(self) -> None:
        validation = _validate_sft_qa(
            SFTQAResult(
                question="How does the visible structural pocket relate to ligand placement?",
                answer=(
                    "In panel a, the modeled pocket forces the ligand into the displayed "
                    "orientation, while panel b shows an alternative placement."
                ),
            ),
            {"must_include": []},
        )

        self.assertTrue(any("strong causal wording" in error for error in validation["hard_errors"]))

    def test_non_molecular_morphology_wording_is_not_blocked_by_model_policy(self) -> None:
        validation = _validate_sft_qa(
            SFTQAResult(
                question="How do the structural differences dictate access across the fibres?",
                answer=(
                    "Panel a shows a dense carbon fibre surface, whereas panel b shows a "
                    "porous network that permits access throughout the fibre volume."
                ),
            ),
            {"must_include": []},
        )

        self.assertFalse(validation["hard_errors"])


class MetadataSecurityTests(unittest.TestCase):
    def test_meta_snapshot_redacts_resolved_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ScienceflowSFTRunStore(tmp_dir)
            run_dir = store.create_run_dir("sample_test", run_id="sample_test_run")
            store.save_meta(
                run_dir,
                sample={},
                config={
                    "models": {
                        "judge": {
                            "api_key": "resolved-secret",
                            "api_key_env": "JUDGE_API_KEY",
                        }
                    }
                },
                routing={},
            )
            with open(run_dir / "00_meta" / "config.json", "r", encoding="utf-8") as handle:
                stored = json.load(handle)

        self.assertEqual(stored["models"]["judge"]["api_key"], "***REDACTED***")
        self.assertEqual(stored["models"]["judge"]["api_key_env"], "JUDGE_API_KEY")

    def test_existing_snapshot_is_redacted_on_resume_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = ScienceflowSFTRunStore(tmp_dir)
            run_dir = store.create_run_dir("sample_test", run_id="sample_test_run")
            store.save_json(run_dir / "00_meta" / "config.json", {"token": "persisted-secret"})

            store.redact_existing_config_snapshot(run_dir)
            with open(run_dir / "00_meta" / "config.json", "r", encoding="utf-8") as handle:
                stored = json.load(handle)

        self.assertEqual(stored["token"], "***REDACTED***")

    def test_sample_metadata_omits_large_image_base64_by_default(self) -> None:
        snapshot = _sample_metadata_snapshot(
            {
                "sample_id": "sample_test",
                "image_base64": "data:image/png;base64," + "a" * 100,
                "raw_record": {"source": "unit"},
            },
            include_image_base64=False,
        )

        self.assertIsNone(snapshot["image_base64"])
        self.assertTrue(snapshot["raw_record"]["image_base64_omitted"])
        self.assertGreater(snapshot["raw_record"]["image_base64_chars"], 100)


class InvocationPerformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_completion_url_cache()

    def tearDown(self) -> None:
        _clear_completion_url_cache()

    def test_completion_url_cache_prioritizes_known_working_endpoint(self) -> None:
        self.assertEqual(
            _completion_urls("http://relay"),
            ["http://relay/chat/completions", "http://relay/v1/chat/completions"],
        )

        _remember_completion_url("http://relay", "http://relay/v1/chat/completions")

        self.assertEqual(
            _completion_urls("http://relay"),
            ["http://relay/v1/chat/completions", "http://relay/chat/completions"],
        )

    def test_snapshot_mode_accepts_failure_only_mode(self) -> None:
        self.assertEqual(_snapshot_mode("failure"), "failure")
        self.assertEqual(_snapshot_mode(True), "all")
        self.assertEqual(_snapshot_mode(False), "none")

    def test_failure_only_snapshot_skips_successful_calls(self) -> None:
        class Template:
            requires_image = False

            @staticmethod
            def render(payload):
                return "prompt"

        class Router:
            @staticmethod
            def resolve(_operator_name):
                return type("Runtime", (), {
                    "profile": type("Profile", (), {
                        "api_key": "test-key",
                        "api_key_env": "TEST_KEY",
                        "base_url": "http://relay/v1",
                        "model": "test-model",
                    })(),
                    "stream": False,
                    "temperature": 0.1,
                    "max_tokens": 128,
                    "timeout": 10,
                    "enable_thinking": False,
                    "reasoning_effort": "",
                })()

        response = {
            "choices": [{
                "message": {"content": "{\"ok\": true}"}
            }]
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            prompt_dir = Path(tmp_dir) / "prompt"
            with patch("dataflow.operators.common._post_json", return_value=response):
                parsed = invoke_prompt(
                    template=Template(),
                    payload={"x": 1},
                    image_base64=None,
                    operator_name="extract_evidence",
                    prompt_name="extract_evidence",
                    prompt_dir=prompt_dir,
                    router=Router(),
                    prompt_snapshot_enabled="failure",
                    max_retries=1,
                )

            self.assertEqual(parsed, {"ok": True})
            self.assertFalse(prompt_dir.exists())


class BatchWorkerTests(unittest.TestCase):
    def test_zero_workers_means_auto_high_throughput(self) -> None:
        workers = _resolve_worker_count(0, total=1000)

        self.assertGreaterEqual(workers, 32)
        self.assertLessEqual(workers, 1000)

    def test_positive_workers_are_preserved(self) -> None:
        self.assertEqual(_resolve_worker_count(7, total=1000), 7)


class ParquetSampleLoadingTests(unittest.TestCase):
    def test_load_sample_supports_automix_multimodal_parquet(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = Path(tmp_dir)
            table = pa.table({
                "sample_id": pa.array(["parquet_sample"]),
                "problem": pa.array(["<image>\nWhat differs between the panels?"]),
                "answer": pa.array(["Panel a is denser than panel b."]),
                "images": pa.array([[_TINY_PNG]], type=pa.list_(pa.binary())),
                "source": pa.array(["unit_test"]),
                "image_path": pa.array(["/does/not/exist.png"]),
                "question_type": pa.array(["compare"]),
            })
            pq.write_table(table, data_dir / "sft_vqa_00000.parquet")

            sample = load_sample(str(data_dir), "parquet_sample")

        self.assertEqual(sample.sample_id, "parquet_sample")
        self.assertEqual(sample.caption, "What differs between the panels?")
        self.assertEqual(sample.context, ["Panel a is denser than panel b."])
        self.assertTrue(sample.image_base64.startswith("data:image/png;base64,"))
        self.assertEqual(sample.raw_record["dataset_format"], "automix_multimodal_parquet")


if __name__ == "__main__":
    unittest.main()
