#!/usr/bin/env python3
"""Integration evaluation for scicode-lint.

Full pipeline (no human-in-the-loop):
1. GENERATE (Sonnet) - Select patterns, generate code with bugs
2. VERIFY (Sonnet) - Confirm manifest is accurate
3. LINT (vLLM) - Run scicode-lint on generated code
4. JUDGE (Sonnet) - Categorize findings as TP/FP/FN

Modes:
- Ephemeral (default): Generate, evaluate, print report, discard
- Persistent (--save): Generate, evaluate, save to generated/<run-id>/
- Re-evaluate (--id exists): Load saved scenarios, re-run lint + judge

Usage:
    # Full pipeline - generate and evaluate
    python integration_eval.py --generate-count 10

    # Save with custom ID
    python integration_eval.py --generate-count 10 --save --id baseline_v1

    # Re-evaluate existing ID (no generation, just lint + judge)
    python integration_eval.py --id baseline_v1

    # List saved runs
    python integration_eval.py --list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dev_lib.run_output import RunOutput, write_worker

from .generator import ScenarioGenerator
from .models import JudgeResult, ScenarioResult, categorize_judge_results
from .storage import (
    GENERATED_DIR,
    list_saved_runs,
    load_scenarios_from_disk,
    save_scenarios,
)

# Directories
INTEGRATION_DIR = Path(__file__).parent
REPORTS_DIR = INTEGRATION_DIR / "reports"


async def reeval_existing_run(run_dir: Path, args: argparse.Namespace) -> int:
    """Re-evaluate an existing saved run (lint + judge only, no generation).

    Two-phase approach for GPU efficiency:
    - Phase 1: Run all linting concurrently (keeps GPU busy via vLLM batching)
    - Phase 2: Run all judging concurrently (Claude CLI, rate-limited)
    """
    include_unverified = getattr(args, "include_unverified", False)
    saved_scenarios = load_scenarios_from_disk(run_dir, include_unverified=include_unverified)
    if not saved_scenarios:
        print(f"Error: No scenarios found in {run_dir}")
        return 1

    print(f"Loaded {len(saved_scenarios)} scenarios from {run_dir}")

    # Create run output for streaming logs
    run_output = RunOutput.create(REPORTS_DIR, f"reeval_{args.id}", items_dirname="scenarios")
    print(f"Output directory: {run_output.run_dir}")
    print(f"Monitor progress: tail -f {run_output.log}")
    run_output.init_log()

    write_queue: asyncio.Queue[tuple[Path, str] | None] = asyncio.Queue()
    writer_task = asyncio.create_task(write_worker(write_queue))
    progress_file = run_output.log.open("a")

    try:
        # Create generator just for linter and judge
        generator = ScenarioGenerator(seed=args.seed)
        total = len(saved_scenarios)

        # Phase 1: Lint all scenarios concurrently (GPU stays busy)
        print(f"\n{'=' * 60}")
        print(f"PHASE 1: Linting all {total} scenarios (vLLM)")
        print(f"{'=' * 60}")

        async def lint_one(idx: int, name: str, code: str) -> tuple[int, str, list[dict[str, Any]]]:
            findings = await generator.run_linter(code)
            print(f"  [{idx + 1}/{total}] {name}: {len(findings)} findings")
            return idx, name, findings

        lint_tasks = [
            lint_one(i, name, code)
            for i, (name, code, _manifest, _verified) in enumerate(saved_scenarios)
        ]
        lint_results_raw = await asyncio.gather(*lint_tasks)
        # Sort by original index to maintain order
        lint_results = sorted(lint_results_raw, key=lambda x: x[0])
        all_findings = {name: findings for _idx, name, findings in lint_results}

        print(f"\nPhase 1 complete: linted {total} scenarios")

        # Phase 2: Judge all scenarios concurrently (Claude CLI rate-limited)
        results: list[ScenarioResult] = []

        if not args.skip_judge:
            print(f"\n{'=' * 60}")
            print(f"PHASE 2: Judging all {total} scenarios (Sonnet)")
            print(f"{'=' * 60}")

            async def judge_one(
                idx: int,
                name: str,
                code: str,
                manifest: list[dict[str, Any]],
                findings: list[dict[str, Any]],
            ) -> tuple[int, str, JudgeResult | None]:
                judge_result = await generator.judge_findings(code, manifest, findings)
                print(f"  [{idx + 1}/{total}] {name}: judged")
                return idx, name, judge_result

            judge_tasks = [
                judge_one(i, name, code, manifest, all_findings[name])
                for i, (name, code, manifest, _verified) in enumerate(saved_scenarios)
            ]
            judge_results_raw = await asyncio.gather(*judge_tasks)
            judge_results_map = {name: jr for _idx, name, jr in judge_results_raw}

            print(f"\nPhase 2 complete: judged {total} scenarios")
        else:
            judge_results_map = {}

        # Phase 3: Assemble results and write logs
        print(f"\n{'=' * 60}")
        print("Assembling results...")
        print(f"{'=' * 60}")

        for i, (name, code, manifest, verified) in enumerate(saved_scenarios):
            findings = all_findings[name]
            judge_result = judge_results_map.get(name)

            log_lines: list[str] = [f"Scenario: {name}\n{'=' * 60}\n"]
            log_lines.append(f"[LINT] {len(findings)} findings\n{json.dumps(findings, indent=2)}\n")

            tp_intended, tp_bonus, false_positives, false_negatives, judge_logs = (
                categorize_judge_results(judge_result, manifest, findings, args.skip_judge)
            )
            log_lines.extend(judge_logs)

            result = ScenarioResult(
                name=name,
                code=code,
                patterns=[],  # Not stored in saved runs
                manifest=manifest,
                verified=verified,
                bugs_intended=len(manifest),
                bugs_detected=len(tp_intended) + len(tp_bonus),
                tp_intended=len(tp_intended),
                tp_bonus=tp_bonus,
                false_positives=false_positives,
                false_negatives=false_negatives,
            )
            results.append(result)

            msg = (
                f"[{i + 1}/{total}] {name} "
                f"TP={result.tp_intended}/{result.bugs_intended} "
                f"Bonus={len(result.tp_bonus)} FP={len(result.false_positives)}"
            )
            print(
                f"  {name}: TP={result.tp_intended}/{result.bugs_intended}, "
                f"Bonus={len(result.tp_bonus)}, FP={len(result.false_positives)}"
            )

            log_lines.append(
                f"[RESULT] TP={result.tp_intended}/{result.bugs_intended}, "
                f"Bonus={len(result.tp_bonus)}, FP={len(result.false_positives)}, "
                f"FN={len(result.false_negatives)}\n"
            )

            # Stream to disk
            log_path = run_output.item_file(name)
            await write_queue.put((log_path, "\n".join(log_lines)))

            progress_file.write(msg + "\n")
            progress_file.flush()

        # Signal writer to stop and wait
        await write_queue.put(None)
        await writer_task
    finally:
        progress_file.close()

    # Print report
    print_report(results)

    # Save JSON report
    if args.output:
        save_json_report(results, args.output)
    else:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_output = REPORTS_DIR / f"integration_eval_{timestamp}.json"
        save_json_report(results, default_output)

    print(f"\nRun logs: {run_output.run_dir}")

    # Summary
    total_intended = sum(s.bugs_intended for s in results)
    total_tp = sum(s.tp_intended for s in results)
    success = total_tp == total_intended

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Scenarios: {len(results)}")
    print(f"Detection: {total_tp}/{total_intended} intended bugs")
    if success:
        print("Status: All intended bugs detected")
    else:
        print(f"Status: {total_intended - total_tp} bugs missed")

    return 0 if success else 1


def print_report(scenarios: list[ScenarioResult]) -> None:
    """Print evaluation report to console."""
    total_intended = sum(s.bugs_intended for s in scenarios)
    total_tp_intended = sum(s.tp_intended for s in scenarios)
    total_tp_bonus = sum(len(s.tp_bonus) for s in scenarios)
    total_fp = sum(len(s.false_positives) for s in scenarios)
    total_fn = sum(len(s.false_negatives) for s in scenarios)

    print("\n" + "=" * 60)
    print("INTEGRATION EVALUATION REPORT")
    print("=" * 60)
    print(f"Scenarios: {len(scenarios)}")
    print(f"Bugs intended: {total_intended}")
    print(f"TP-intended (expected bugs found): {total_tp_intended}")
    print(f"TP-bonus (verified extra bugs): {total_tp_bonus}")
    print(f"FP (rejected, not real bugs): {total_fp}")
    print(f"FN (missed bugs): {total_fn}")
    print()

    if total_intended > 0:
        recall = total_tp_intended / total_intended * 100
        print(f"Recall: {recall:.1f}%")

    total_findings = total_tp_intended + total_tp_bonus + total_fp
    if total_findings > 0:
        precision = (total_tp_intended + total_tp_bonus) / total_findings * 100
        print(f"Precision: {precision:.1f}%")

    print("=" * 60)

    # Per-scenario breakdown
    if len(scenarios) > 1:
        print("\nPer-scenario results:")
        for s in scenarios:
            status = "PASS" if s.tp_intended == s.bugs_intended else "FAIL"
            bonus_str = f" +{len(s.tp_bonus)} bonus" if s.tp_bonus else ""
            print(f"  {s.name[:40]}: {s.tp_intended}/{s.bugs_intended} TP{bonus_str} [{status}]")


def save_json_report(scenarios: list[ScenarioResult], output_path: Path) -> None:
    """Save JSON report."""
    total_intended = sum(s.bugs_intended for s in scenarios)
    total_tp_intended = sum(s.tp_intended for s in scenarios)
    total_tp_bonus = sum(len(s.tp_bonus) for s in scenarios)
    total_fn = sum(len(s.false_negatives) for s in scenarios)

    report: dict[str, Any] = {
        "summary": {
            "scenarios": len(scenarios),
            "total_bugs_intended": total_intended,
            "total_tp_intended": total_tp_intended,
            "total_tp_bonus": total_tp_bonus,
            "total_fn": total_fn,
            "recall": total_tp_intended / total_intended if total_intended > 0 else 0.0,
        },
        "scenarios": [
            {
                "name": s.name,
                "patterns": s.patterns,
                "bugs_intended": s.bugs_intended,
                "tp_intended": s.tp_intended,
                "tp_bonus": s.tp_bonus,
                "false_negatives": s.false_negatives,
                "verified": s.verified,
            }
            for s in scenarios
        ],
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved to: {output_path}")


# =============================================================================
# Main
# =============================================================================


def main() -> int:
    """Generate and evaluate integration test scenarios."""
    parser = argparse.ArgumentParser(
        description="Generate and evaluate integration test scenarios using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate and evaluate 10 scenarios (ephemeral - discards after)
  python integration_eval.py --generate-count 10

  # Skip verification/judge for faster (less accurate) evaluation
  python integration_eval.py --generate-count 10 --skip-verification --skip-judge

  # Save with auto-generated ID (timestamp)
  python integration_eval.py --generate-count 10 --save

  # Save with custom ID for regression testing
  python integration_eval.py --generate-count 10 --save --id baseline_v1

  # Re-evaluate existing saved run (no generation)
  python integration_eval.py --id baseline_v1

  # Re-evaluate including unverified scenarios
  python integration_eval.py --id baseline_v1 --include-unverified

  # Force regeneration of existing ID
  python integration_eval.py --generate-count 10 --save --id baseline_v1 --force

  # List all saved runs
  python integration_eval.py --list
""",
    )
    parser.add_argument(
        "--generate-count",
        type=int,
        dest="generate_count",
        help="Number of scenarios to generate (triggers generation mode)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip manifest verification",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip result judging (use deterministic comparison)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save scenarios to disk for regression tests (default: ephemeral mode)",
    )
    parser.add_argument(
        "--id",
        type=str,
        help="Run ID for saved scenarios (default: auto-generated timestamp)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if ID already exists",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List saved runs and exit",
    )
    parser.add_argument(
        "--include-unverified",
        action="store_true",
        dest="include_unverified",
        help="Include scenarios that failed verification (re-eval mode only)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON report path",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    # Handle --list
    if args.list:
        runs = list_saved_runs()
        if not runs:
            print("No saved runs found.")
            print(
                "Generate with: python integration_eval.py --generate-count 10 --save --id my_run"
            )
            return 0
        print("Saved runs:")
        for run_id, _, mtime in runs:
            print(f"  {run_id}  ({mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        return 0

    # Validate flag combinations
    if args.save and args.generate_count is None and args.id is None:
        print("Error: --save requires --generate-count")
        print("  Example: python integration_eval.py --generate-count 10 --save")
        return 1

    if args.force and (args.generate_count is None or args.id is None):
        print("Error: --force requires both --generate-count and --id")
        print(
            "  Example: python integration_eval.py --generate-count 10 --save --id baseline_v1 --force"
        )
        return 1

    if args.include_unverified and args.generate_count is not None:
        print(
            "Error: --include-unverified only applies to re-eval mode (--id without --generate-count)"
        )
        print("  Example: python integration_eval.py --id baseline_v1 --include-unverified")
        return 1

    # Determine mode: generate or re-eval
    run_id = args.id if args.id else datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = GENERATED_DIR / run_id
    id_exists = run_dir.exists() and (run_dir / "expected.yaml").exists()

    # Mode validation
    if args.generate_count is None and args.id is None:
        print("Error: Must specify --generate-count N (generate) or --id X (re-evaluate)")
        print("  Generate: python integration_eval.py --generate-count 10")
        print("  Re-eval:  python integration_eval.py --id baseline_v1")
        return 1

    if args.generate_count is None and args.id:
        # Re-eval mode: ID must exist
        if not id_exists:
            print(f"Error: Run ID '{args.id}' does not exist")
            print("Use --list to see available runs")
            return 1
        # Warn if --save is specified in re-eval mode
        if args.save:
            print("Warning: --save is ignored in re-eval mode (scenarios already saved)")
        print(f"Re-evaluating run '{args.id}'...")
        return asyncio.run(reeval_existing_run(run_dir, args))

    if args.generate_count and args.id and id_exists and not args.force:
        # ID exists but trying to generate - require --force
        print(f"Error: Run ID '{args.id}' already exists")
        print("Use --force to regenerate, or omit --generate-count to re-evaluate")
        return 1

    # Generate mode (args.generate_count is set at this point)
    assert args.generate_count is not None  # for type checker

    # Create generator (uses Claude CLI)
    mode = "persistent (--save)" if args.save else "ephemeral"
    print(f"Mode: {mode}")
    generator = ScenarioGenerator(seed=args.seed)

    # Create run output for streaming logs
    run_output = RunOutput.create(
        REPORTS_DIR, f"generate_{args.generate_count}", items_dirname="scenarios"
    )
    print(f"Output directory: {run_output.run_dir}")
    print(f"Monitor progress: tail -f {run_output.log}")
    run_output.init_log()

    # Generate scenarios
    print(f"\nGenerating {args.generate_count} scenarios (seed={args.seed})...")
    scenarios = asyncio.run(
        generator.generate_batch(
            count=args.generate_count,
            skip_verification=args.skip_verification,
            skip_judge=args.skip_judge,
            run_output=run_output,
        )
    )

    if not scenarios:
        print("\nNo scenarios generated successfully.")
        return 1

    # Print report
    print_report(scenarios)

    # Save if requested
    if args.save:
        save_scenarios(scenarios, run_dir, seed=args.seed)
        print(f"\nSaved {len(scenarios)} scenarios to {run_dir}")
        print(f"Run ID: {run_id}")
        print(f"To re-evaluate: python integration_eval.py --id {run_id}")

    # Save JSON report
    if args.output:
        save_json_report(scenarios, args.output)
    else:
        save_json_report(scenarios, run_output.run_dir / "report.json")

    print(f"\nRun logs: {run_output.run_dir}")

    # Summary
    total_intended = sum(s.bugs_intended for s in scenarios)
    total_tp = sum(s.tp_intended for s in scenarios)
    success = total_tp == total_intended

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Generated: {len(scenarios)}/{args.generate_count} scenarios")
    print(f"Detection: {total_tp}/{total_intended} intended bugs")
    if success:
        print("Status: All intended bugs detected")
    else:
        print(f"Status: {total_intended - total_tp} bugs missed")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
