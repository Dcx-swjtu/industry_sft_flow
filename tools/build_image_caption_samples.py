#!/usr/bin/env python3
"""Convert company image_caption records into ScienceFlow SampleInput JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = PROJECT_DIR.parent
DEFAULT_INPUT_DIRS = [
    DATASETS_DIR / "image_caption" / "02_architecture",
    DATASETS_DIR / "image_caption" / "05_cad",
]
DEFAULT_OUTPUT_DIR = DATASETS_DIR / "data" / "image_caption_arch_cad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", type=Path, help="image_caption subset directory")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-samples-per-dataset", type=int, default=0)
    parser.add_argument("--context-limit", type=int, default=6000)
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(clean_text(item) for item in value if clean_text(item))
    text = str(value).replace("\r", "\n").replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def truncate(text: str, limit: int) -> str:
    text = clean_text(text)
    if len(text) <= limit:
        return text
    head_len = int(limit * 0.7)
    tail_len = int(limit * 0.2)
    return f"{text[:head_len].rstrip()}\n\n[...truncated...]\n\n{text[-tail_len:].lstrip()}"


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def detect_visual_kind(path: Path, captions: list[str]) -> str:
    text = f"{path.name} {' '.join(captions)}".lower()
    if "cap.table" in text or re.search(r"\btable\s*\d*", text):
        return "table"
    if any(word in text for word in ("chart", "plot", "curve", "graph", "axis", "bar")):
        return "chart"
    if any(word in text for word in ("workflow", "pipeline", "framework")):
        return "workflow"
    if "cap.fig" in text or "fig" in text or "figure" in text:
        return "figure"
    return "other"


def figure_id(path: Path) -> str:
    match = re.search(r"Cap\.([^.]+)", path.name)
    return match.group(1) if match else path.stem


def sample_id_for(dataset: str, json_path: Path) -> str:
    digest = hashlib.sha1(str(json_path).encode("utf-8")).hexdigest()[:10]
    normalized_dataset = re.sub(r"[^A-Za-z0-9_]+", "_", dataset)
    return f"imgcap_{normalized_dataset}_{digest}"


def global_info_for(record_dir: Path) -> dict[str, Any]:
    matches = sorted(record_dir.glob("*_global_info.json"))
    if not matches:
        return {}
    return load_json(matches[0]) or {}


def build_sample(json_path: Path, dataset: str, context_limit: int) -> dict[str, Any] | None:
    data = load_json(json_path)
    if not data:
        return None
    captions = [clean_text(item) for item in data.get("captions", []) if clean_text(item)]
    if not captions:
        return None
    image_path = json_path.with_suffix(".image.png")
    if not image_path.exists():
        return None
    task = data.get("task") if isinstance(data.get("task"), dict) else {}
    contexts = [clean_text(item) for item in data.get("contexts", []) if clean_text(item)]
    merged_context = truncate("\n\n".join(contexts), context_limit)
    subfigures = data.get("subfigures_info")
    if not isinstance(subfigures, list):
        subfigures = []
    global_info = global_info_for(json_path.parent)
    fig_id = figure_id(json_path)
    visual_kind = detect_visual_kind(json_path, captions)
    pdf_path = clean_text(task.get("pdf_path") or global_info.get("原始文件路径"))
    doc_id = clean_text(task.get("token")) or json_path.parent.name.removesuffix("_manual")
    sample_id = sample_id_for(dataset, json_path)

    return {
        "sample_id": sample_id,
        "image_path": str(image_path.resolve()),
        "image_base64": None,
        "caption": clean_text(" ".join(captions)),
        "raw_caption": clean_text(" ".join(captions)),
        "context": [merged_context] if merged_context else [],
        "title": Path(pdf_path).stem if pdf_path else doc_id,
        "raw_subject": [
            f"dataset={dataset}",
            f"doc_id={doc_id}",
            f"figure_id={fig_id}",
            f"visual_kind={visual_kind}",
        ],
        "subfigure_infos": subfigures,
        "raw_record": {
            "dataset": dataset,
            "doc_id": doc_id,
            "figure_id": fig_id,
            "visual_kind": visual_kind,
            "json_path": str(json_path.resolve()),
            "image_path": str(image_path.resolve()),
            "group_path": str(json_path.with_suffix(".group.png").resolve()),
            "caption_image_path": str(json_path.with_suffix(".caption.png").resolve()),
            "pdf_path": pdf_path,
            "task": task,
            "global_info": global_info,
            "group_size": data.get("group_size"),
            "image_size": data.get("image_size"),
            "image_concat_type": data.get("image_concat_type"),
        },
    }


def iter_dataset_samples(input_dir: Path, context_limit: int) -> list[dict[str, Any]]:
    samples = []
    for json_path in sorted(input_dir.rglob("*.json")):
        if json_path.name.endswith("_global_info.json"):
            continue
        sample = build_sample(json_path, input_dir.name, context_limit)
        if sample:
            samples.append(sample)
    return samples


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dirs = args.input_dir or DEFAULT_INPUT_DIRS
    output_dir = args.output_dir
    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    all_samples: list[dict[str, Any]] = []
    for input_dir in input_dirs:
        samples = iter_dataset_samples(input_dir, args.context_limit)
        if args.max_samples_per_dataset > 0:
            samples = samples[: args.max_samples_per_dataset]
        all_samples.extend(samples)
        for sample in samples:
            write_json(samples_dir / f"{sample['sample_id']}.json", sample)

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for sample in all_samples:
            row = {
                "sample_id": sample["sample_id"],
                "dataset": sample["raw_record"]["dataset"],
                "doc_id": sample["raw_record"]["doc_id"],
                "figure_id": sample["raw_record"]["figure_id"],
                "visual_kind": sample["raw_record"]["visual_kind"],
                "image_path": sample["image_path"],
                "json_path": sample["raw_record"]["json_path"],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "sample_ids.txt").write_text(
        "\n".join(sample["sample_id"] for sample in all_samples) + "\n",
        encoding="utf-8",
    )
    summary = {
        "input_dirs": [str(path) for path in input_dirs],
        "output_dir": str(output_dir),
        "samples": len(all_samples),
        "sample_counts": dict(Counter(sample["raw_record"]["dataset"] for sample in all_samples)),
        "visual_kind_counts": dict(Counter(sample["raw_record"]["visual_kind"] for sample in all_samples)),
        "outputs": {
            "samples_dir": str(samples_dir),
            "manifest_jsonl": str(manifest_path),
            "sample_ids_txt": str(output_dir / "sample_ids.txt"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

