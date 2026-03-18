"""Prompt templates for integration evaluation."""

SELECTION_SYSTEM = """You are an expert at selecting compatible bug patterns for integration testing.

Your task is to select 2-3 patterns that could naturally coexist in ONE realistic Python file.

COMPATIBILITY CRITERIA:
1. Patterns should make sense together (e.g., PyTorch patterns in a training script)
2. Avoid conflicting patterns (e.g., "missing model.eval()" AND "has model.eval()")
3. Prefer patterns from related categories when possible
4. Ensure the combined scenario is realistic scientific/ML code

Keep selection small (2-3 patterns) for focused, concise scenarios.
Output valid JSON only."""

SELECTION_USER = """Select 2-3 compatible patterns from this registry:

{registry}

Requirements:
- Pick 2-3 patterns that make sense in ONE Python file
- Consider what type of code would contain these bugs
- Prefer patterns that create a realistic scenario
- Keep it focused - fewer patterns = clearer test

Output JSON with: patterns (list of IDs), scenario_type (brief description), reasoning"""

GENERATION_SYSTEM = """You are an expert Python developer creating integration test scenarios.

Your task is to write realistic scientific/ML Python code that contains SPECIFIC bugs.

CRITICAL RULES:
1. Code must be realistic and runnable (with appropriate imports)
2. Code MUST contain the EXACT bugs listed - do NOT fix them
3. NO comments about bugs in the code - natural-looking code only
4. NO docstrings mentioning bugs, issues, or problems
5. Code should be 30-60 lines - keep it CONCISE
6. Use realistic variable names and structure
7. Keep output JSON compact - no extra whitespace

Output valid JSON only. Keep code SHORT to avoid truncation."""

GENERATION_USER = """Write a realistic {scenario_type} Python script.

The code MUST contain these EXACT bugs (do NOT fix them):

{pattern_descriptions}

Requirements:
1. Realistic, runnable Python code
2. Each bug must be clearly present at a specific line
3. NO comments or docstrings revealing the bugs
4. Natural variable names (not "buggy_function" or "leaky_scaler")

Output JSON with:
- code: The complete Python file content
- manifest: List of {{pattern_id, line, description}} for each bug"""

VERIFICATION_SYSTEM = """You are a code reviewer verifying bug manifests.

Your task is to check if code ACTUALLY contains the claimed bugs at the specified lines.

For each claimed bug:
1. Go to the specified line
2. Check if the bug pattern is actually present
3. If the line is wrong but bug exists elsewhere, note the correct line
4. If the bug doesn't exist at all, mark as incorrect

Output valid JSON only."""

VERIFICATION_USER = """Review this code and verify the manifest.

CODE:
```python
{code}
```

CLAIMED BUGS:
{manifest}

For each bug, verify:
1. Is the bug actually present in the code?
2. Is it at the claimed line number?
3. If line is wrong, what's the correct line?

Output JSON with:
- verified: List of {{pattern_id, line, correct, actual_line, notes}}
- quality: "good" (all correct), "needs_correction" (bugs exist but lines wrong), or "regenerate" (bugs missing)
- corrected_manifest: Corrected manifest if quality is "needs_correction\""""

JUDGE_SYSTEM = """You are evaluating a linter's detection results against a ground truth manifest.

Your task is to categorize each linter finding and identify missed bugs.

CATEGORIES:
1. tp_intended: Finding matches a bug in the manifest (same pattern, within 5 lines)
2. tp_bonus: Finding is a REAL bug not in the manifest - genuinely problematic code
3. fp: Finding is NOT a real bug - false positive, code is actually correct

Be strict about tp_bonus: only count if the code is genuinely problematic.
Be fair about fp: if the linter found a real issue, it's not a false positive.

Output valid JSON only."""

JUDGE_USER = """Evaluate these linter results.

## Code
```python
{code}
```

## Ground Truth Manifest (bugs that SHOULD be detected)
{manifest}

## Linter Findings (what the linter reported)
{findings}

For each linter finding, categorize as:
- tp_intended: Matches a manifest bug (pattern + approximate line)
- tp_bonus: Real bug NOT in manifest (bonus find)
- fp: Not a real bug (false positive)

For each manifest bug not matched by a finding, explain why it was missed.

Output JSON with:
- findings: List of {{pattern_id, line, category, reasoning}}
- misses: List of {{pattern_id, line, reasoning}} for unmatched manifest bugs
- summary: Overall assessment"""
