"""Sample loading for ScienceFlow-style JSON records."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Any


def _candidate_paths(data_dir: Path, sample_id: str) -> list[Path]:
    return [
        data_dir / f"{sample_id}.json",
        data_dir / "samples" / f"{sample_id}.json",
    ]


def _find_sample_path(data_dir: Path, sample_id: str) -> Path:
    for path in _candidate_paths(data_dir, sample_id):
        if path.exists():
            return path
    matches = sorted(data_dir.rglob(f"{sample_id}.json"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"sample not found under {data_dir}: {sample_id}")


def _image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    raw = path.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def load_sample(data_dir: str, sample_id: str):
    from domain import SampleInput

    root = Path(data_dir).resolve()
    path = _find_sample_path(root, sample_id)
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

    return SampleInput.from_dict(data)

