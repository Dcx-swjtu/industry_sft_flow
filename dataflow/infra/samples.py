"""Sample loading for ScienceFlow-style JSON records."""

from __future__ import annotations

import base64
import json
import mimetypes
from functools import lru_cache
from pathlib import Path
from typing import Any


def _candidate_paths(data_dir: Path, sample_id: str) -> list[Path]:
    return [
        data_dir / f"{sample_id}.json",
        data_dir / "samples" / f"{sample_id}.json",
    ]


def _find_sample_path(data_dir: Path, sample_id: str) -> Path | None:
    for path in _candidate_paths(data_dir, sample_id):
        if path.exists():
            return path
    match = _find_sample_path_recursive(str(data_dir.resolve()), sample_id)
    if match:
        return Path(match)
    return None


@lru_cache(maxsize=8192)
def _find_sample_path_recursive(data_dir: str, sample_id: str) -> str:
    matches = sorted(Path(data_dir).rglob(f"{sample_id}.json"))
    return str(matches[0]) if matches else ""


def _image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    raw = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _image_bytes_to_data_url(raw: bytes) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif raw.startswith(b"GIF87a") or raw.startswith(b"GIF89a"):
        mime = "image/gif"
    else:
        mime = "image/png"
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r", "\n").replace("\x00", " ").strip()


def _first_image_data_url(images: Any) -> str:
    candidates = images
    if candidates is None:
        return ""
    if isinstance(candidates, bytes):
        return _image_bytes_to_data_url(candidates)
    if not isinstance(candidates, (list, tuple)):
        return ""
    for item in candidates:
        if isinstance(item, bytes):
            return _image_bytes_to_data_url(item)
    return ""


def _parquet_row_to_sample(row: dict[str, Any], *, parquet_path: Path, sample_id: str) -> dict[str, Any]:
    resolved_sample_id = _clean_text(row.get("sample_id") or row.get("id") or sample_id)
    problem = _clean_text(row.get("problem")).replace("<image>", "").strip()
    answer = _clean_text(row.get("answer"))
    subject = _clean_text(row.get("subject"))
    question_type = _clean_text(row.get("question_type"))
    title = _clean_text(row.get("title"))
    image_path = _clean_text(row.get("image_path"))
    image_base64 = _first_image_data_url(row.get("images"))

    raw_record = {
        key: value
        for key, value in row.items()
        if key != "images" and isinstance(value, (str, int, float, bool, type(None)))
    }
    raw_record.update({
        "parquet_path": str(parquet_path),
        "dataset_format": "automix_multimodal_parquet",
    })

    raw_subject = [value for value in [subject, question_type, _clean_text(row.get("source"))] if value]

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


@lru_cache(maxsize=64)
def _parquet_paths(data_dir: str) -> tuple[str, ...]:
    return tuple(str(path) for path in sorted(Path(data_dir).glob("*.parquet")))


@lru_cache(maxsize=512)
def _parquet_schema_names(parquet_path: str) -> tuple[str, ...]:
    import pyarrow.parquet as pq

    return tuple(pq.ParquetFile(parquet_path).schema_arrow.names)


def _read_filtered_parquet_row(
    *,
    parquet_path: Path,
    columns: list[str],
    schema_names: set[str],
    sample_id: str,
):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    filter_terms = []
    if "sample_id" in schema_names:
        filter_terms.append(("sample_id", "=", sample_id))
    if "id" in schema_names:
        filter_terms.append(("id", "=", sample_id))
    if not filter_terms:
        return None

    try:
        table = pq.read_table(
            parquet_path,
            columns=columns,
            filters=[[term] for term in filter_terms],
        )
    except (NotImplementedError, TypeError, ValueError):
        table = pq.read_table(parquet_path, columns=columns)
        mask = None
        if "sample_id" in table.column_names:
            mask = pc.equal(table["sample_id"], sample_id)
        if "id" in table.column_names:
            id_mask = pc.equal(table["id"], sample_id)
            mask = id_mask if mask is None else pc.or_(mask, id_mask)
        if mask is None:
            return None
        table = table.filter(mask)

    if table.num_rows:
        return table.slice(0, 1).to_pylist()[0]
    return None


def _load_parquet_sample(data_dir: Path, sample_id: str) -> dict[str, Any] | None:
    parquet_paths = _parquet_paths(str(data_dir.resolve()))
    if not parquet_paths:
        return None

    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            f"sample not found as JSON under {data_dir}: {sample_id}; "
            "parquet sample loading requires pyarrow"
        ) from exc

    wanted_columns = [
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

    for parquet_path_text in parquet_paths:
        parquet_path = Path(parquet_path_text)
        schema_names = set(_parquet_schema_names(parquet_path_text))
        columns = [name for name in wanted_columns if name in schema_names]
        row = _read_filtered_parquet_row(
            parquet_path=parquet_path,
            columns=columns,
            schema_names=schema_names,
            sample_id=sample_id,
        )
        if row is not None:
            return _parquet_row_to_sample(row, parquet_path=parquet_path, sample_id=sample_id)

    return None


def load_sample(data_dir: str, sample_id: str):
    from domain import SampleInput

    root = Path(data_dir).resolve()
    path = _find_sample_path(root, sample_id)
    if path is not None:
        with path.open("r", encoding="utf-8") as handle:
            data: dict[str, Any] = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"sample must be a JSON object: {path}")

        data.setdefault("sample_id", sample_id)
        data.setdefault("raw_record", {})
        data["raw_record"] = dict(data.get("raw_record") or {})
        data["raw_record"].setdefault("sample_json_path", str(path))

        image_path = Path(str(data.get("image_path") or ""))
        if image_path and not image_path.is_absolute():
            image_path = (path.parent / image_path).resolve()
            data["image_path"] = str(image_path)
        if not data.get("image_base64"):
            if not image_path.exists():
                raise FileNotFoundError(f"sample image not found: {image_path}")
            data["image_base64"] = _image_to_data_url(image_path)
    else:
        data = _load_parquet_sample(root, sample_id)
        if data is None:
            raise FileNotFoundError(f"sample not found under {root}: {sample_id}")

    return SampleInput.from_dict(data)
