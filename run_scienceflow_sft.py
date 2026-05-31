#!/usr/bin/env python3
"""CLI entrypoint for scienceflow-sft — SFT training data pipeline."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def _ensure_repo_imports() -> None:
    here = Path(__file__).resolve().parent
    # Bundle-local imports: domain/operators/prompts/runner and vendored dataflow/.
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def _print_result(result: dict) -> None:
    print(f"Run ID: {result.get('run_id')}")
    sft_qa = result.get("sft_qa") or {}
    judge_result = result.get("judge_result") or {}

    print(f"Question: {sft_qa.get('question', '')[:240]}")
    print(f"Answer: {sft_qa.get('answer', '')[:240]}...")
    print(f"Accepted: {judge_result.get('accepted')}")
    print(f"Overall score: {judge_result.get('overall_score')}")
    print(f"Retry count: {result.get('retry_count')}")
    print(f"Rejected after retries: {result.get('final_summary', {}).get('rejected_after_retries')}")


def _run_batch(sample_ids: list[str], *, config_path: str, resume: bool, workers: int) -> None:
    from runner.scienceflow_sft_runner import run_scienceflow_sft

    total = len(sample_ids)
    results_summary: list[tuple[str, str, str]] = [None] * total  # type: ignore[assignment]

    def _run_one(idx: int, sid: str) -> tuple[int, str, dict | None, Exception | None]:
        try:
            result = run_scienceflow_sft(sid, config_path=config_path, resume=resume)
            return idx, sid, result, None
        except Exception as e:
            return idx, sid, None, e

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_one, i, sid): i for i, sid in enumerate(sample_ids)}
        for future in as_completed(futures):
            idx, sid, result, exc = future.result()
            if exc:
                print(f"[batch] {sid} FAILED: {exc}")
                results_summary[idx] = (sid, "FAILED", str(exc))
                continue

            judge = result.get("judge_result") or {}
            accepted = judge.get("accepted", False)
            score = judge.get("overall_score", 0)
            results_summary[idx] = (sid, "ACCEPTED" if accepted else "REJECTED", f"score={score}")
            print(f"[batch] {sid}: {'ACCEPTED' if accepted else 'REJECTED'} (score={score})")

    ok = sum(1 for _, s, _ in results_summary if s == "ACCEPTED")
    failed = sum(1 for _, s, _ in results_summary if s == "FAILED")
    rejected = total - ok - failed

    print(f"\n{'='*60}")
    print(f"[batch] Done: {total} total, {ok} accepted, {rejected} rejected, {failed} failed")
    print(f"{'='*60}")
    for sid, status, detail in results_summary:
        print(f"  {sid}: {status} ({detail})")


def main() -> None:
    _ensure_repo_imports()
    from runner.scienceflow_sft_runner import run_scienceflow_sft

    parser = argparse.ArgumentParser(description="Run the scienceflow-sft five-stage SFT training data pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    default_config = str(Path(__file__).resolve().parent / "configs" / "default.yaml")

    run_parser = subparsers.add_parser("run", help="Start a new scienceflow-sft run")
    run_parser.add_argument("sample_id", help="Sample ID, e.g. sample_020")
    run_parser.add_argument("--config", default=default_config, help="Path to config YAML")
    run_parser.add_argument("--run-id", default=None, help="Optional run ID")
    run_parser.add_argument("--no-resume", action="store_true", help="Disable resume")

    resume_parser = subparsers.add_parser("resume", help="Resume an existing scienceflow-sft run")
    resume_parser.add_argument("run_id", help="Existing run ID under runs/")
    resume_parser.add_argument("sample_id", help="Sample ID, e.g. sample_020")
    resume_parser.add_argument("--config", default=default_config, help="Path to config YAML")

    batch_parser = subparsers.add_parser("batch", help="Run multiple samples in parallel")
    batch_parser.add_argument("sample_ids", nargs="+", help="Sample IDs, e.g. sample_000 sample_001")
    batch_parser.add_argument("--config", default=default_config, help="Path to config YAML")
    batch_parser.add_argument("--no-resume", action="store_true", help="Disable resume")
    batch_parser.add_argument("-w", "--workers", type=int, default=4, help="Parallel workers (default: 4)")

    args = parser.parse_args()

    if args.command == "batch":
        _run_batch(args.sample_ids, config_path=args.config, resume=not args.no_resume, workers=args.workers)
    elif args.command == "run":
        result = run_scienceflow_sft(
            args.sample_id, config_path=args.config, run_id=args.run_id,
            resume=not args.no_resume,
        )
        _print_result(result)
    else:
        result = run_scienceflow_sft(
            args.sample_id, config_path=args.config, run_id=args.run_id,
            resume=True,
        )
        _print_result(result)


if __name__ == "__main__":
    main()
