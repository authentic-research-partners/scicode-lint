"""Scenario generator for integration evaluation."""

from __future__ import annotations

import asyncio
import json
import random
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from dev_lib.claude_cli import ClaudeCLI, ClaudeCLIError
from dev_lib.run_output import RunOutput, write_worker
from scicode_lint import SciCodeLinter
from scicode_lint.detectors.catalog import DetectionCatalog

from .models import (
    GeneratedScenario,
    JudgeResult,
    ManifestEntry,
    PatternInfo,
    PatternSelection,
    ScenarioResult,
    VerificationResult,
    categorize_judge_results,
)
from .prompts import (
    GENERATION_SYSTEM,
    GENERATION_USER,
    JUDGE_SYSTEM,
    JUDGE_USER,
    SELECTION_SYSTEM,
    SELECTION_USER,
    VERIFICATION_SYSTEM,
    VERIFICATION_USER,
)


class ScenarioGenerator:
    """Generates and evaluates integration test scenarios using Claude CLI (Sonnet)."""

    def __init__(
        self,
        patterns_dir: Path | None = None,
        seed: int = 42,
    ):
        self.catalog = DetectionCatalog(patterns_dir)
        self.rng = random.Random(seed)
        self.patterns = self._load_pattern_registry()
        self.linter = SciCodeLinter()
        self.cli = ClaudeCLI(model="sonnet", effort="medium")

    def _load_pattern_registry(self) -> list[PatternInfo]:
        """Load all patterns with ID and description."""
        return [
            PatternInfo(
                id=p.id,
                description=p.warning_message,
                category=p.category,
            )
            for p in self.catalog.patterns
        ]

    def _format_registry(self, patterns: list[PatternInfo] | None = None) -> str:
        """Format pattern registry for LLM prompt."""
        patterns = patterns or self.patterns
        # Shuffle to avoid always selecting same patterns
        shuffled = self.rng.sample(patterns, len(patterns))
        lines = []
        for p in shuffled:
            lines.append(f"- {p.id} [{p.category}]: {p.description}")
        return "\n".join(lines)

    def _format_pattern_descriptions(self, pattern_ids: list[str]) -> str:
        """Format selected patterns for generation prompt."""
        lines = []
        for pid in pattern_ids:
            for p in self.patterns:
                if p.id == pid:
                    lines.append(f"- {p.id}: {p.description}")
                    break
        return "\n".join(lines)

    async def _call_claude(
        self, system_prompt: str, user_prompt: str, effort: str = "medium"
    ) -> dict[str, Any]:
        """Call Claude CLI and parse JSON response.

        Args:
            system_prompt: System prompt text.
            user_prompt: User prompt text.
            effort: Thinking effort level.

        Returns:
            Parsed JSON dict from Claude response.

        Raises:
            ClaudeCLIError: On any Claude CLI error.
            ClaudeCLIParseError: If JSON parsing fails.
        """
        prompt = f"{system_prompt}\n\n{user_prompt}"
        return await self.cli.arun_json(prompt, effort=effort, timeout=180)

    async def select_patterns(self) -> PatternSelection:
        """Use Sonnet to select compatible patterns."""
        registry = self._format_registry()
        user_prompt = SELECTION_USER.format(registry=registry)

        result = await self._call_claude(SELECTION_SYSTEM, user_prompt)
        return PatternSelection.model_validate(result)

    async def generate_scenario(self, selection: PatternSelection) -> GeneratedScenario:
        """Use Sonnet to generate code with specified bugs."""
        pattern_descriptions = self._format_pattern_descriptions(selection.patterns)
        user_prompt = GENERATION_USER.format(
            scenario_type=selection.scenario_type,
            pattern_descriptions=pattern_descriptions,
        )

        result = await self._call_claude(GENERATION_SYSTEM, user_prompt)
        return GeneratedScenario.model_validate(result)

    async def verify_manifest(self, code: str, manifest: list[ManifestEntry]) -> VerificationResult:
        """Use Sonnet via Claude CLI to verify the manifest."""
        manifest_str = json.dumps([m.model_dump() for m in manifest], indent=2)
        prompt = (
            f"{VERIFICATION_SYSTEM}\n\n{VERIFICATION_USER.format(code=code, manifest=manifest_str)}"
        )

        try:
            content = await self.cli.arun_json(prompt, effort="high", timeout=120)
            return VerificationResult.model_validate(content)
        except ClaudeCLIError as e:
            print(f"Claude CLI error during verification: {e}")
            return VerificationResult(
                verified=[],
                quality="regenerate",
                corrected_manifest=None,
            )

    async def judge_findings(
        self, code: str, manifest: list[dict[str, Any]], findings: list[dict[str, Any]]
    ) -> JudgeResult:
        """Use Sonnet to judge linter findings against manifest.

        Args:
            code: The Python code that was analyzed
            manifest: Ground truth bugs (pattern_id, line, description)
            findings: Linter findings (pattern_id, line, message)

        Returns:
            JudgeResult with categorized findings and missed bugs
        """
        manifest_str = json.dumps(manifest, indent=2)
        findings_str = json.dumps(findings, indent=2)
        prompt = f"{JUDGE_SYSTEM}\n\n{JUDGE_USER.format(code=code, manifest=manifest_str, findings=findings_str)}"

        try:
            content = await self.cli.arun_json(prompt, effort="high", timeout=120)
            return JudgeResult.model_validate(content)
        except ClaudeCLIError as e:
            print(f"Claude CLI error during judging: {e}")
            return JudgeResult(findings=[], misses=[], summary=f"Claude error: {e}")

    async def run_linter(self, code: str) -> list[dict[str, Any]]:
        """Run linter on code and return findings as dicts.

        Args:
            code: The Python code to evaluate

        Returns:
            List of findings as dicts with pattern_id, line, message
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
        ) as f:
            f.write(code)
            temp_file = Path(f.name)

        try:
            result = await self.linter._check_file_async(temp_file)
            return [
                {
                    "pattern_id": f.id,
                    "line": f.location.lines[0] if f.location and f.location.lines else 0,
                    "message": f.explanation or "",
                }
                for f in result.findings
            ]
        finally:
            temp_file.unlink(missing_ok=True)

    async def generate_one(
        self,
        skip_verification: bool = False,
        skip_judge: bool = False,
        max_attempts: int = 3,
    ) -> tuple[ScenarioResult | None, str]:
        """Generate, verify, lint, and judge one scenario.

        Flow:
        1. Generate (Sonnet) - select patterns and generate code
        2. Verify manifest (Sonnet) - confirm bugs are present
        3. Run linter (vLLM) - detect bugs
        4. Judge results (Sonnet) - categorize findings

        Returns:
            Tuple of (ScenarioResult or None, raw_log) where raw_log contains
            serialized outputs from each step for disk persistence.
        """
        log_lines: list[str] = []

        for attempt in range(max_attempts):
            try:
                log_lines.append(f"=== Attempt {attempt + 1}/{max_attempts} ===\n")

                # Step 1: Generate (Sonnet)
                selection = await self.select_patterns()
                print(f"  Selected patterns: {selection.patterns}")
                print(f"  Scenario type: {selection.scenario_type}")
                log_lines.append(f"[SELECT]\n{selection.model_dump_json(indent=2)}\n")

                scenario = await self.generate_scenario(selection)
                print(f"  Generated {len(scenario.code)} chars with {len(scenario.manifest)} bugs")
                log_lines.append(
                    f"[GENERATE] {len(scenario.code)} chars, {len(scenario.manifest)} bugs\n"
                    f"Manifest: {json.dumps([m.model_dump() for m in scenario.manifest], indent=2)}\n"
                )

                # Step 2: Verify manifest (Sonnet)
                manifest = scenario.manifest
                verified = not skip_verification

                if not skip_verification:
                    print("  Verifying manifest (Sonnet)...")
                    verification = await self.verify_manifest(scenario.code, scenario.manifest)
                    log_lines.append(f"[VERIFY]\n{verification.model_dump_json(indent=2)}\n")

                    if verification.quality == "regenerate":
                        print(f"  Verification failed (attempt {attempt + 1}/{max_attempts})")
                        log_lines.append("[VERIFY] quality=regenerate, retrying...\n")
                        continue

                    if (
                        verification.quality == "needs_correction"
                        and verification.corrected_manifest
                    ):
                        manifest = verification.corrected_manifest
                        print("  Manifest corrected by verifier")

                # Step 3: Run linter (vLLM)
                print("  Running linter (vLLM)...")
                manifest_dicts = [m.model_dump() for m in manifest]
                findings = await self.run_linter(scenario.code)
                print(f"  Linter found {len(findings)} issues")
                log_lines.append(
                    f"[LINT] {len(findings)} findings\n{json.dumps(findings, indent=2)}\n"
                )

                # Step 4: Judge results (Sonnet)
                judge_result = None
                if not skip_judge:
                    print("  Judging results (Sonnet)...")
                    judge_result = await self.judge_findings(
                        scenario.code, manifest_dicts, findings
                    )

                tp_intended, tp_bonus, false_positives, false_negatives, judge_logs = (
                    categorize_judge_results(judge_result, manifest_dicts, findings, skip_judge)
                )
                log_lines.extend(judge_logs)

                # Create scenario name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                scenario_name = f"gen_{timestamp}_{selection.scenario_type.replace(' ', '_')[:30]}"
                scenario_name = "".join(
                    c if c.isalnum() or c == "_" else "_" for c in scenario_name
                )

                result = ScenarioResult(
                    name=scenario_name,
                    code=scenario.code,
                    patterns=selection.patterns,
                    manifest=manifest_dicts,
                    verified=verified,
                    bugs_intended=len(manifest_dicts),
                    bugs_detected=len(tp_intended) + len(tp_bonus),
                    tp_intended=len(tp_intended),
                    tp_bonus=tp_bonus,
                    false_positives=false_positives,
                    false_negatives=false_negatives,
                )

                log_lines.append(
                    f"[RESULT] TP={result.tp_intended}/{result.bugs_intended}, "
                    f"Bonus={len(result.tp_bonus)}, FP={len(result.false_positives)}, "
                    f"FN={len(result.false_negatives)}\n"
                )
                return result, "\n".join(log_lines)

            except Exception as e:
                print(f"  Error (attempt {attempt + 1}/{max_attempts}): {e}")
                log_lines.append(f"[ERROR] {e}\n")
                continue

        return None, "\n".join(log_lines)

    async def generate_batch(
        self,
        count: int,
        skip_verification: bool = False,
        skip_judge: bool = False,
        run_output: RunOutput | None = None,
    ) -> list[ScenarioResult]:
        """Generate multiple scenarios with optional incremental disk streaming.

        Args:
            count: Number of scenarios to generate.
            skip_verification: Skip manifest verification.
            skip_judge: Skip result judging.
            run_output: Optional output directory for streaming logs to disk.

        Returns:
            List of successful ScenarioResult objects.
        """
        write_queue: asyncio.Queue[tuple[Path, str] | None] = asyncio.Queue()
        writer_task = asyncio.create_task(write_worker(write_queue))
        progress_file = run_output.log.open("a") if run_output else None

        try:
            # Launch all scenarios concurrently (ClaudeCLI rate limiter handles throttling)
            tasks = []
            for i in range(count):
                task = asyncio.create_task(
                    self.generate_one(skip_verification=skip_verification, skip_judge=skip_judge)
                )
                tasks.append((i, task))

            scenarios = []
            completed = 0
            for coro in asyncio.as_completed([t for _, t in tasks]):
                result_tuple = await coro
                scenario, raw_log = result_tuple
                completed += 1

                if scenario:
                    scenarios.append(scenario)
                    msg = (
                        f"[{completed}/{count}] ✓ {scenario.name} "
                        f"TP={scenario.tp_intended}/{scenario.bugs_intended} "
                        f"Bonus={len(scenario.tp_bonus)} FN={len(scenario.false_negatives)}"
                    )
                    print(msg)

                    # Stream to disk
                    if run_output:
                        log_path = run_output.item_file(scenario.name)
                        await write_queue.put((log_path, raw_log))
                else:
                    msg = f"[{completed}/{count}] ✗ Failed to generate scenario"
                    print(msg)

                    # Write failure log too
                    if run_output and raw_log:
                        fail_name = f"failed_{completed}_{datetime.now().strftime('%H%M%S')}"
                        log_path = run_output.item_file(fail_name)
                        await write_queue.put((log_path, raw_log))

                if progress_file:
                    progress_file.write(msg + "\n")
                    progress_file.flush()

            # Signal writer to stop and wait
            await write_queue.put(None)
            await writer_task

            return scenarios
        finally:
            if progress_file:
                progress_file.close()
