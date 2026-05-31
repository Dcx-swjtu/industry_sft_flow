#!/usr/bin/env python3
"""Export accepted scienceflow-sft runs to ShareGPT JSON and a QA report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_DIR.parent
DEFAULT_RUNS_DIR = PROJECT_DIR / "runs" / "image_caption_arch_cad"
DEFAULT_OUTPUT_JSON = DATASETS_DIR / "train_data" / "image_caption_scienceflow_sft_data" / "scienceflow_arch_cad_generated_v1.json"
DEFAULT_OUTPUT_MD = DATASETS_DIR / "train_data" / "image_caption_scienceflow_sft_data" / "scienceflow_arch_cad_generated_v1.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--include-rejected", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def iter_run_dirs(runs_dir: Path) -> list[Path]:
    if not runs_dir.exists():
        return []
    return sorted(path for path in runs_dir.iterdir() if path.is_dir())


def build_entry(run_dir: Path, include_rejected: bool) -> dict[str, Any] | None:
    final = load_json(run_dir / "04_generate_sft_qa" / "final_result.json")
    sample = load_json(run_dir / "00_meta" / "sample.json")
    if not final or not sample:
        return None
    if not include_rejected and not final.get("export_to_sft"):
        return None
    sft_qa = final.get("sft_qa") if isinstance(final.get("sft_qa"), dict) else {}
    question = str(sft_qa.get("question") or "").strip()
    answer = str(sft_qa.get("answer") or "").strip()
    image_path = str(sample.get("image_path") or "").strip()
    if not question or not answer or not image_path:
        return None
    judge = final.get("judge_result") if isinstance(final.get("judge_result"), dict) else {}
    raw_record = sample.get("raw_record") if isinstance(sample.get("raw_record"), dict) else {}
    return {
        "id": str(sample.get("sample_id") or run_dir.name),
        "messages": [
            {"role": "user", "content": f"<image>{question}"},
            {"role": "assistant", "content": answer},
        ],
        "images": [image_path],
        "source": "scienceflow_sft_image_caption",
        "meta": {
            "run_id": run_dir.name,
            "dataset": raw_record.get("dataset"),
            "doc_id": raw_record.get("doc_id"),
            "figure_id": raw_record.get("figure_id"),
            "visual_kind": raw_record.get("visual_kind"),
            "json_path": raw_record.get("json_path"),
            "caption": sample.get("caption"),
            "export_to_sft": final.get("export_to_sft"),
            "overall_score": judge.get("overall_score"),
            "dimension_scores": judge.get("dimension_scores"),
            "final_summary": final.get("final_summary"),
        },
    }


def markdown(entries: list[dict[str, Any]]) -> str:
    lines = ["# ScienceFlow Image Caption SFT QA Results", "", f"Accepted samples: {len(entries)}", ""]
    for index, entry in enumerate(entries, start=1):
        meta = entry.get("meta") or {}
        messages = entry.get("messages") or []
        question = messages[0].get("content", "").replace("<image>", "", 1)
        answer = messages[1].get("content", "") if len(messages) > 1 else ""
        score = meta.get("overall_score")
        model_title = "ScienceFlow Pipeline"
        lines.extend(
            [
                f"## sample_{index:03d} ({model_title}) Score: {score}",
                "",
                "**Question**",
                "",
                question,
                "",
                "**Answer**",
                "",
                answer,
                "",
                "**Original Image**",
                "",
                f"![Image]({(entry.get('images') or [''])[0]})",
                "",
                "---",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    entries = [
        entry
        for entry in (build_entry(run_dir, args.include_rejected) for run_dir in iter_run_dirs(args.runs_dir))
        if entry is not None
    ]
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown(entries), encoding="utf-8")
    print(json.dumps({
        "runs_dir": str(args.runs_dir),
        "exported": len(entries),
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
