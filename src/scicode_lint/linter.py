"""Main linter orchestration."""

import asyncio
import json
import time
from pathlib import Path
from typing import Literal

from loguru import logger

from scicode_lint.ast_utils import resolve_name_with_fallback
from scicode_lint.config import LinterConfig, get_default_config, get_strip_comments
from scicode_lint.detectors.catalog import DetectionCatalog, DetectionPattern
from scicode_lint.detectors.prompts import generate_detection_prompt, get_system_prompt
from scicode_lint.exceptions import LLMConnectionError, SciCodeLintError
from scicode_lint.llm.client import create_client
from scicode_lint.llm.exceptions import ContextLengthError
from scicode_lint.llm.models import DetectionResult
from scicode_lint.llm.tokens import check_context_length
from scicode_lint.output.formatter import (
    Finding,
    LintError,
    LintResult,
    Location,
    PatternCheckResult,
    PatternFailure,
)
from scicode_lint.preprocessing.comments import strip_comments


class NotebookParseError(SciCodeLintError):
    """Raised when a Jupyter notebook cannot be parsed."""


def calculate_scaled_timeout(line_count: int, base_timeout: float) -> float:
    """Calculate timeout scaled by code line count.

    Larger files need more processing time. Scale timeout based on line count:
    - Files <= 200 lines: base_timeout
    - Files > 200 lines: scale up, +50s per 200 lines above threshold
    - Maximum: 5x base_timeout

    Args:
        line_count: Number of lines in code.
        base_timeout: Base timeout for small files.

    Returns:
        Scaled timeout in seconds.
    """
    if line_count <= 200:
        return base_timeout

    # Scale: +50s per 200 lines above threshold
    extra_blocks = (line_count - 200) // 200
    scaled = base_timeout + (extra_blocks * 50)

    # Cap at 5x base timeout
    max_timeout = base_timeout * 5
    return min(scaled, max_timeout)


def extract_code_from_notebook(notebook_path: Path) -> str:
    """Extract Python code from Jupyter notebook.

    Args:
        notebook_path: Path to .ipynb file.

    Returns:
        Concatenated Python code from all code cells.

    Raises:
        NotebookParseError: If notebook cannot be parsed.
    """
    try:
        with open(notebook_path, encoding="utf-8") as f:
            notebook = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise NotebookParseError(f"Could not parse notebook {notebook_path}: {e}") from e

    code_cells = []
    cells = notebook.get("cells", [])

    for cell in cells:
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                code_cells.append("".join(source))
            elif isinstance(source, str):
                code_cells.append(source)

    return "\n\n".join(code_cells)


class SciCodeLinter:
    """Main linter class for checking scientific Python code.

    Designed for both human users and GenAI coding agents.
    Detects common patterns of bugs in scientific code including
    data leakage, PyTorch training issues, numerical errors, and more.

    Example:
        >>> from scicode_lint import SciCodeLinter
        >>> from pathlib import Path
        >>> linter = SciCodeLinter()
        >>> result = linter.check_file(Path("myfile.py"))
        >>> for finding in result.findings:
        ...     print(f"{finding.id}: {finding.explanation}")
    """

    def __init__(self, config: LinterConfig | None = None):
        """
        Initialize linter.

        Args:
            config: Linter configuration (uses defaults if None)
        """
        self.config = config or get_default_config()
        self.catalog = DetectionCatalog(self.config.patterns_dir)
        self.llm = create_client(self.config.llm_config)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    def get_pattern(self, pattern_id: str) -> DetectionPattern | None:
        """
        Get pattern details by ID.

        Useful for GenAI agents to understand what a pattern checks for.

        Args:
            pattern_id: Pattern ID (e.g., "ml-001", "pt-001")

        Returns:
            DetectionPattern object with id, category, severity,
            detection_question, and warning_message fields.
            Returns None if pattern not found.

        Example:
            >>> linter = SciCodeLinter()
            >>> pattern = linter.get_pattern("ml-001")
            >>> print(pattern.warning_message)
        """
        return self.catalog.get_pattern(pattern_id)

    def list_patterns(self) -> list[DetectionPattern]:
        """
        List all available detection patterns.

        Returns:
            List of DetectionPattern objects

        Example:
            >>> linter = SciCodeLinter()
            >>> for pattern in linter.list_patterns():
            ...     print(f"{pattern.id}: {pattern.warning_message[:50]}")
        """
        return self.catalog.patterns

    def check_file(self, file_path: Path) -> LintResult:
        """
        Check a single file.

        Args:
            file_path: Path to Python file to check

        Returns:
            Lint results for the file
        """
        return asyncio.run(self._check_file_async(file_path))

    async def _check_file_async(
        self, file_path: Path, pattern_timeout: float | None = None
    ) -> LintResult:
        """
        Check a single file asynchronously with parallel pattern checks.

        Args:
            file_path: Path to Python file to check
            pattern_timeout: Optional timeout in seconds per pattern, applied AFTER
                semaphore acquired (so queue wait doesn't count toward timeout).

        Returns:
            Lint results for the file
        """
        file_start = time.time()
        logger.info(f"Checking file: {file_path}")

        # Read file (extract Python from notebooks)
        if file_path.suffix == ".ipynb":
            code = extract_code_from_notebook(file_path)
            logger.debug(f"Extracted {len(code)} bytes from notebook")
        else:
            code = file_path.read_text()
            logger.debug(f"File size: {len(code)} bytes")

        # Strip comments before LLM analysis (if enabled)
        if get_strip_comments():
            original_len = len(code)
            code = strip_comments(code)
            stripped_len = original_len - len(code)
            if stripped_len > 0:
                logger.debug(f"Stripped {stripped_len} bytes of comments")

        # Scale timeout based on line count (larger files need more time)
        if pattern_timeout is not None:
            line_count = len(code.splitlines())
            scaled_timeout = calculate_scaled_timeout(line_count, pattern_timeout)
            if scaled_timeout != pattern_timeout:
                logger.debug(
                    f"Scaled timeout: {pattern_timeout}s -> {scaled_timeout}s ({line_count} lines)"
                )
            pattern_timeout = scaled_timeout

        # Collect patterns to check
        patterns_to_check = []
        for pattern in self.catalog.patterns:
            # Skip if severity not enabled
            if pattern.severity not in self.config.enabled_severities:
                logger.debug(f"Skipping pattern {pattern.id} (severity not enabled)")
                continue

            # Skip if pattern ID filter specified and this pattern not included
            if self.config.enabled_patterns and pattern.id not in self.config.enabled_patterns:
                logger.debug(f"Skipping pattern {pattern.id} (not in pattern filter)")
                continue

            # Skip if category filter specified and this category not included
            if (
                self.config.enabled_categories
                and pattern.category not in self.config.enabled_categories
            ):
                logger.debug(
                    f"Skipping pattern {pattern.id} (category {pattern.category} not in filter)"
                )
                continue

            patterns_to_check.append(pattern)

        logger.info(f"Checking {len(patterns_to_check)} patterns concurrently")

        # Pre-generate all prompts (synchronous work) BEFORE creating async tasks
        # This prevents sequential prompt generation from blocking concurrent execution
        logger.debug("Generating prompts...")
        system_prompt = get_system_prompt()
        prompts = [
            (pattern, generate_detection_prompt(code, pattern)) for pattern in patterns_to_check
        ]
        logger.debug(f"Generated {len(prompts)} prompts")

        # Check context length before sending to LLM
        # Use first prompt as representative (all have similar size)
        if prompts:
            max_tokens = self.llm.get_max_model_len()
            _, user_prompt = prompts[0]
            try:
                # vLLM reserves max_completion_tokens for output
                output_buffer = self.config.llm_config.max_completion_tokens
                _, estimated = check_context_length(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    file_path=str(file_path),
                    output_buffer=output_buffer,
                )
                logger.debug(
                    f"Context check passed: {estimated:,} tokens "
                    f"(limit: {max_tokens:,}, {estimated * 100 // max_tokens}% used)"
                )
            except ContextLengthError as e:
                logger.error(str(e))
                # Return result with error for structured reporting
                error = LintError(
                    file=file_path,
                    error_type="ContextLengthError",
                    message=str(e),
                    details=e.to_dict(),
                )
                return LintResult(file=file_path, findings=[], error=error)

        # Execute all patterns concurrently - vLLM handles batching internally
        batch_start = time.time()
        logger.debug(f"Checking {len(prompts)} patterns concurrently (vLLM will batch)")

        tasks = [
            self._check_pattern_async_with_prompt(
                pattern, system_prompt, user_prompt, file_path, pattern_timeout
            )
            for pattern, user_prompt in prompts
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        batch_elapsed = time.time() - batch_start
        logger.debug(f"Batch completed in {batch_elapsed:.2f}s")

        # Process results
        findings: list[Finding] = []
        checked_patterns: list[PatternCheckResult] = []
        failed_patterns: list[PatternFailure] = []
        patterns_checked = 0
        patterns_failed = 0

        for pattern, result in zip(patterns_to_check, results):
            if isinstance(result, Exception):
                # LLMConnectionError is systemic (server unreachable) — not a
                # per-pattern failure. Bubble it up so the CLI can surface a
                # single clear error and exit with code 2 instead of pretending
                # the file was linted "successfully with N pattern failures".
                if isinstance(result, LLMConnectionError):
                    raise result
                patterns_failed += 1
                if isinstance(result, asyncio.TimeoutError):
                    error_type = "timeout"
                    error_msg = f"Timed out after {pattern_timeout}s"
                    logger.warning(f"Pattern {pattern.id} timed out after {pattern_timeout}s")
                elif (
                    "context length" in str(result).lower() or "input tokens" in str(result).lower()
                ):
                    error_type = "context_length"
                    error_msg = str(result)[:200]
                    logger.warning(f"Pattern {pattern.id} context error: {result}")
                else:
                    error_type = "api_error"
                    error_msg = str(result)[:200]
                    logger.warning(f"Error running pattern {pattern.id}: {result}")
                failed_patterns.append(
                    PatternFailure(
                        pattern_id=pattern.id,
                        error_type=error_type,
                        error_message=error_msg,
                    )
                )
                continue

            # Type narrowing: result is tuple[DetectionResult, float] here
            assert isinstance(result, tuple)
            patterns_checked += 1
            detection, pattern_elapsed = result

            logger.info(
                f"Pattern {pattern.id} completed in {pattern_elapsed:.2f}s "
                f"(detected={detection.detected}, confidence={detection.confidence:.2f})"
            )
            if detection.reasoning:
                logger.info(f"  Reasoning: {detection.reasoning}")

            # Record all pattern check results (for eval/debugging)
            checked_patterns.append(
                PatternCheckResult(
                    pattern_id=pattern.id,
                    detected=detection.detected,
                    confidence=detection.confidence,
                    reasoning=detection.reasoning,
                    thinking=detection.thinking,
                )
            )

            # Add findings if detected (yes or context-dependent)
            if (
                detection.detected in ["yes", "context-dependent"]
                and detection.confidence >= self.config.min_confidence
            ):
                findings.extend(self._create_findings(pattern, detection, file_path, code))

        file_elapsed = time.time() - file_start
        logger.success(
            f"File {file_path.name} completed in {file_elapsed:.2f}s "
            f"(batch: {batch_elapsed:.2f}s, {patterns_checked} patterns, "
            f"{patterns_failed} failed, {len(findings)} findings)"
        )

        return LintResult(
            file=file_path,
            findings=findings,
            checked_patterns=checked_patterns,
            patterns_failed=patterns_failed,
            failed_patterns=failed_patterns,
        )

    async def _check_pattern_async_with_prompt(
        self,
        pattern: DetectionPattern,
        system_prompt: str,
        user_prompt: str,
        file_path: Path,
        timeout: float | None = None,
    ) -> tuple[DetectionResult, float]:
        """
        Check a single pattern asynchronously with pre-generated prompts.

        Args:
            pattern: Detection pattern
            system_prompt: Pre-generated system prompt
            user_prompt: Pre-generated user prompt
            file_path: File being analyzed
            timeout: Optional timeout in seconds for LLM processing time only.
                     Queue wait time (semaphore) does not count toward timeout,
                     ensuring fair processing time for all patterns regardless
                     of queue position.

        Returns:
            Tuple of (detection result, elapsed time)

        Raises:
            asyncio.TimeoutError: If LLM processing exceeds timeout
            Exception: If pattern check fails for other reasons
        """
        async with self._semaphore:
            pattern_start = time.time()
            logger.debug(f"Starting async check for pattern {pattern.id}")

            # Timeout applies to LLM call only (after semaphore acquired)
            llm_coro = self.llm.async_complete_structured(
                system_prompt, user_prompt, DetectionResult
            )
            if timeout is not None:
                detection = await asyncio.wait_for(llm_coro, timeout=timeout)
            else:
                detection = await llm_coro

            pattern_elapsed = time.time() - pattern_start
            return detection, pattern_elapsed

    def _create_findings(
        self,
        pattern: DetectionPattern,
        detection: DetectionResult,
        file_path: Path,
        code: str,
    ) -> list[Finding]:
        """
        Create Finding objects from detection result.

        Uses AST resolution to convert name-based locations from LLM to actual
        line numbers and code snippets. The result includes:
        - lines: Full function/method range (for context)
        - focus_line: Specific line to look at (verified from near_line)
        - snippet: Code snippet for the function/method
        - name: Verified function/class/method name

        Args:
            pattern: Detection pattern
            detection: Validated detection result from LLM
            file_path: File being analyzed
            code: Source code of the file (for extracting snippets)

        Returns:
            List of findings
        """
        # Resolve name-based location to actual lines using AST
        lines: list[int] = []
        focus_line: int | None = None
        snippet = ""
        name: str | None = None
        location_type: str | None = None

        if detection.location:
            name = detection.location.name
            location_type = detection.location.location_type
            near_line = detection.location.near_line

            # Resolve using AST
            resolved = resolve_name_with_fallback(
                code=code,
                name=name,
                location_type=detection.location.location_type,
                near_line=near_line,
            )

            if resolved:
                lines = resolved.lines
                snippet = resolved.snippet

                # Verify and set focus_line from near_line hint
                if near_line:
                    if resolved.start_line <= near_line <= resolved.end_line:
                        # near_line is within the resolved function - verified
                        focus_line = near_line
                        logger.debug(
                            f"Resolved '{name}' lines {resolved.start_line}-"
                            f"{resolved.end_line}, focus: {focus_line}"
                        )
                    else:
                        # near_line outside resolved range - use start
                        focus_line = resolved.start_line
                        logger.debug(f"near_line {near_line} outside '{name}', using start")
                else:
                    # No near_line hint - use start of function
                    focus_line = resolved.start_line
                    logger.debug(
                        f"Resolved '{name}' lines {resolved.start_line}-{resolved.end_line}"
                    )
            else:
                # Fallback: use near_line if resolution fails
                if near_line:
                    code_lines = code.splitlines()
                    start = max(1, near_line - 2)
                    end = min(len(code_lines), near_line + 2)
                    lines = list(range(start, end + 1))
                    focus_line = near_line
                    snippet = "\n".join(code_lines[start - 1 : end])
                    logger.warning(f"AST resolution failed for '{name}', using near_line fallback")
                else:
                    logger.warning(f"AST resolution failed for '{name}' with no fallback")

        # Create location with resolved and verified data
        location = Location(
            lines=lines,
            focus_line=focus_line,
            snippet=snippet,
            name=name,
            location_type=location_type,
        )

        # detection.detected is filtered to be "yes" or "context-dependent" by caller
        assert detection.detected in ["yes", "context-dependent"]
        detection_type: Literal["yes", "context-dependent"] = detection.detected  # type: ignore[assignment]

        finding = Finding(
            id=pattern.id,
            category=pattern.category,
            severity=pattern.severity,
            location=location,
            issue=f"{pattern.id}: Issue detected",
            explanation=pattern.warning_message,
            suggestion="Review the code and fix according to the explanation.",
            confidence=detection.confidence,
            reasoning=detection.reasoning,
            detection_type=detection_type,
            thinking=detection.thinking,
        )

        return [finding]
