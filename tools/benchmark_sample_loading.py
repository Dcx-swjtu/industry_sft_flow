#!/usr/bin/env python3
"""Benchmark current sample loading against the pre-optimization parquet loader."""

from __future__ import annotations

import argparse
import base64
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


WANTED_COLUMNS = [
    "id",
    "problem",
    "answer",
    "images",
    "source",
    "sample_id",
    "run_id",
    "run_time",
    "qa_step",
    "qa_path",
    "image_path",
    "accepted",
    "score",
    "question_type",
    "grounding_confidence",
    "title",
    "doi",
    "subject",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "runs" / "perf_benchmarks")
    parser.add_argument("--methods", default="optimized,baseline", help="Comma list: optimized,baseline")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit for smoke tests")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r", "\n").replace("\x00", " ").strip()


def image_bytes_to_data_url(raw: bytes) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        mime = "image/gif"
    else:
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def first_image_data_url(images: Any) -> str:
    if images is None:
        return ""
    if isinstance(images, bytes):
        return image_bytes_to_data_url(images)
    if not isinstance(images, (list, tuple)):
        return ""
    for item in images:
        if isinstance(item, bytes):
            return image_bytes_to_data_url(item)
    return ""


def parquet_row_to_sample(row: dict[str, Any], *, parquet_path: Path, sample_id: str) -> dict[str, Any]:
    resolved_sample_id = clean_text(row.get("sample_id") or row.get("id") or sample_id)
    problem = clean_text(row.get("problem")).replace("<image>", "").strip()
    answer = clean_text(row.get("answer"))
    subject = clean_text(row.get("subject"))
    question_type = clean_text(row.get("question_type"))
    title = clean_text(row.get("title"))
    image_path = clean_text(row.get("image_path"))
    image_base64 = first_image_data_url(row.get("images"))
    raw_record = {
        key: value
        for key, value in row.items()
        if key != "images" and isinstance(value, (str, int, float, bool, type(None)))
    }
    raw_record.update({
        "parquet_path": str(parquet_path),
        "dataset_format": "automix_multimodal_parquet",
    })
    raw_subject = [value for value in [subject, question_type, clean_text(row.get("source"))] if value]
    return {
        "sample_id": resolved_sample_id,
        "image_path": image_path,
        "image_base64": image_base64 or None,
        "caption": problem,
        "raw_caption": problem,
        "context": [answer] if answer else [],
        "title": title,
        "raw_subject": raw_subject,
        "subfigure_infos": [],
        "raw_record": raw_record,
    }


def baseline_load_sample(data_dir: Path, sample_id: str) -> dict[str, Any]:
    """Pre-optimization parquet path: read each parquet table, then filter in memory."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    for parquet_path in sorted(data_dir.glob("*.parquet")):
        schema_names = set(pq.ParquetFile(parquet_path).schema_arrow.names)
        columns = [name for name in WANTED_COLUMNS if name in schema_names]
        table = pq.read_table(parquet_path, columns=columns)
        mask = None
        if "sample_id" in table.column_names:
            mask = pc.equal(table["sample_id"], sample_id)
        if "id" in table.column_names:
            id_mask = pc.equal(table["id"], sample_id)
            mask = id_mask if mask is None else pc.or_(mask, id_mask)
        if mask is None:
            continue
        matched = table.filter(mask)
        if matched.num_rows:
            return parquet_row_to_sample(
                matched.slice(0, 1).to_pylist()[0],
                parquet_path=parquet_path,
                sample_id=sample_id,
            )
    raise FileNotFoundError(f"sample not found under {data_dir}: {sample_id}")


def optimized_load_sample(data_dir: Path, sample_id: str) -> dict[str, Any]:
    from dataflow.infra.samples import load_sample

    return load_sample(str(data_dir), sample_id).to_dict()


def iter_sample_ids(data_dir: Path) -> list[str]:
    import pyarrow.parquet as pq

    sample_ids: list[str] = []
    for parquet_path in sorted(data_dir.glob("*.parquet")):
        table = pq.read_table(parquet_path, columns=["sample_id"])
        sample_ids.extend(str(value) for value in table.column("sample_id").to_pylist())
    return sample_ids


def load_completed(progress_path: Path, method: str) -> set[str]:
    completed: set[str] = set()
    if not progress_path.exists():
        return completed
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("method") == method and row.get("ok") is True:
            completed.add(str(row.get("sample_id") or ""))
    return completed


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_method(
    *,
    method: str,
    loader: Callable[[Path, str], dict[str, Any]],
    data_dir: Path,
    sample_ids: list[str],
    progress_path: Path,
    progress_every: int,
) -> dict[str, Any]:
    completed = load_completed(progress_path, method)
    times: list[float] = []
    started = time.perf_counter()
    total = len(sample_ids)
    remaining = [sample_id for sample_id in sample_ids if sample_id not in completed]
    print(f"[{method}] starting: total={total} completed={len(completed)} remaining={len(remaining)}", flush=True)

    for index, sample_id in enumerate(remaining, start=1):
        gc.collect()
        t0 = time.perf_counter()
        ok = False
        error = ""
        image_base64_chars = 0
        try:
            sample = loader(data_dir, sample_id)
            if sample.get("sample_id") != sample_id:
                raise AssertionError(f"loaded sample_id mismatch: {sample.get('sample_id')} != {sample_id}")
            image_base64_chars = len(str(sample.get("image_base64") or ""))
            ok = True
        except Exception as exc:  # pragma: no cover - benchmark diagnostics
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - t0
        if ok:
            times.append(elapsed)
        append_jsonl(progress_path, {
            "method": method,
            "sample_id": sample_id,
            "ok": ok,
            "elapsed_s": elapsed,
            "image_base64_chars": image_base64_chars,
            "error": error,
        })
        if not ok:
            raise RuntimeError(f"{method} failed for {sample_id}: {error}")
        done = len(completed) + index
        if index == 1 or index % progress_every == 0 or done == total:
            total_elapsed = time.perf_counter() - started
            rate = index / total_elapsed if total_elapsed else 0.0
            eta = (len(remaining) - index) / rate if rate else 0.0
            print(
                f"[{method}] {done}/{total} latest={elapsed:.3f}s "
                f"rate={rate:.3f}/s eta={eta/60:.1f}m",
                flush=True,
            )

    all_times = [
        json.loads(line)["elapsed_s"]
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        and (json.loads(line).get("method") == method)
        and (json.loads(line).get("ok") is True)
    ]
    return {
        "method": method,
        "samples": len(all_times),
        "total_s": sum(all_times),
        "mean_s": statistics.mean(all_times) if all_times else None,
        "median_s": statistics.median(all_times) if all_times else None,
        "min_s": min(all_times) if all_times else None,
        "max_s": max(all_times) if all_times else None,
        "samples_per_min": (len(all_times) / sum(all_times) * 60) if all_times and sum(all_times) else None,
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    run_id = args.run_id or time.strftime("sample_loading_%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_id
    progress_path = output_dir / "progress.jsonl"
    summary_path = output_dir / "summary.json"

    sample_ids = iter_sample_ids(data_dir)
    if args.limit:
        sample_ids = sample_ids[: args.limit]
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    loaders = {
        "optimized": optimized_load_sample,
        "baseline": baseline_load_sample,
    }

    summaries = []
    for method in methods:
        if method not in loaders:
            raise ValueError(f"unknown method: {method}")
        summary = run_method(
            method=method,
            loader=loaders[method],
            data_dir=data_dir,
            sample_ids=sample_ids,
            progress_path=progress_path,
            progress_every=args.progress_every,
        )
        summaries.append(summary)
        summary_path.write_text(json.dumps({
            "data_dir": str(data_dir),
            "run_id": run_id,
            "sample_count": len(sample_ids),
            "summaries": summaries,
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    by_method = {item["method"]: item for item in summaries}
    if "optimized" in by_method and "baseline" in by_method:
        opt = by_method["optimized"]["total_s"]
        base = by_method["baseline"]["total_s"]
        if opt:
            by_method["optimized"]["speedup_vs_baseline"] = base / opt
    summary_path.write_text(json.dumps({
        "data_dir": str(data_dir),
        "run_id": run_id,
        "sample_count": len(sample_ids),
        "summaries": summaries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
