#!/usr/bin/env python3
"""Redact persisted credentials from ScienceFlow-SFT run configuration snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SENSITIVE_KEYS = {"api_key", "token", "secret", "authorization", "password"}
REDACTED = "***REDACTED***"


def redact(value: Any) -> tuple[Any, bool]:
    if isinstance(value, dict):
        cleaned = {}
        changed = False
        for key, child in value.items():
            normalized = str(key).strip().lower()
            is_secret = normalized in SENSITIVE_KEYS or any(
                normalized.endswith(f"_{suffix}") for suffix in SENSITIVE_KEYS
            )
            if is_secret:
                cleaned[key] = REDACTED
                changed = changed or child != REDACTED
            else:
                cleaned[key], child_changed = redact(child)
                changed = changed or child_changed
        return cleaned, changed
    if isinstance(value, list):
        cleaned_items = []
        changed = False
        for child in value:
            cleaned_child, child_changed = redact(child)
            cleaned_items.append(cleaned_child)
            changed = changed or child_changed
        return cleaned_items, changed
    return value, False


def redact_file(path: Path, *, write: bool) -> bool:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cleaned, changed = redact(payload)
    if changed and write:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(cleaned, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_dir", nargs="?", default="runs", help="Directory containing run folders")
    parser.add_argument("--write", action="store_true", help="Apply redaction; default is dry-run")
    args = parser.parse_args()

    paths = sorted(Path(args.runs_dir).glob("*/00_meta/config.json"))
    changed = sum(1 for path in paths if redact_file(path, write=args.write))
    action = "redacted" if args.write else "would redact"
    print(f"{action} {changed} of {len(paths)} config snapshots")


if __name__ == "__main__":
    main()
