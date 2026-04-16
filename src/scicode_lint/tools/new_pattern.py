#!/usr/bin/env python3
"""
Create a new pattern from template.

Scaffolds a new pattern directory with pattern.toml template and test directories.
"""

import argparse
import datetime
from pathlib import Path


class PatternCreator:
    """Create new pattern from template."""

    def __init__(self, patterns_dir: Path):
        """
        Initialize pattern creator.

        Args:
            patterns_dir: Root patterns directory
        """
        self.patterns_dir = Path(patterns_dir)

    def create_pattern(
        self,
        pattern_id: str,
        pattern_name: str,
        category: str,
        severity: str = "high",
    ) -> None:
        """
        Create a new pattern directory and template.

        Args:
            pattern_id: Pattern ID (e.g., "ml-999")
            pattern_name: Pattern name (e.g., "my-pattern")
            category: Category name
            severity: Severity level (critical, high, medium)
        """
        # Create directory
        pattern_dir = self.patterns_dir / category / f"{pattern_id}-{pattern_name}"

        if pattern_dir.exists():
            raise FileExistsError(f"Pattern directory already exists: {pattern_dir}")

        pattern_dir.mkdir(parents=True, exist_ok=True)
        (pattern_dir / "test_positive").mkdir(exist_ok=True)
        (pattern_dir / "test_negative").mkdir(exist_ok=True)
        (pattern_dir / "test_context_dependent").mkdir(exist_ok=True)

        # Generate pattern.toml
        today = datetime.date.today().isoformat()

        toml_content = f'''# Pattern: {pattern_id}-{pattern_name}
# Created: {today}

[meta]
id = "{pattern_id}"
name = "{pattern_name}"
category = "{category}"
severity = "{severity}"
version = "1.0.0"
created = "{today}"
updated = "{today}"
author = "scicode-lint"

description = """
Brief description of what this pattern detects.
TODO: Update this description.
"""

explanation = """
Detailed explanation of:
- What the issue is
- Why it matters (impact on research/results)
- How to fix it
- Common manifestations

TODO: Write detailed explanation.
"""

research_impact = """
TODO: Describe impact on research quality and scientific conclusions.
"""

tags = []
related_patterns = []
references = []

[detection]
question = """
Clear, specific question the LLM will answer about the code.

Good examples:
- Is there a scaler that is fit on the full dataset before train/test split?
- Does this training loop call optimizer.step() before loss.backward()?

Bad examples:
- Is there data leakage? (too vague)
- Does this code have bugs? (not specific enough)

TODO: Write detection question.
"""

warning_message = """
User-facing message explaining:
1. What the issue is
2. Why it matters
3. How to fix it

TODO: Write warning message.
"""

suggestion = """
Concrete steps to fix the issue.
TODO: Add fix suggestions.
"""

min_confidence = 0.85
false_positive_risks = []  # Optional: Known sources of false positives

[tests]
# Add test cases here
# See existing patterns for examples

[[tests.positive]]
file = "test_positive/example_positive.py"
description = "TODO: Describe what this test case shows"
expected_issue = "TODO: Expected issue description"
min_confidence = 0.85

[tests.positive.expected_location]
type = "function"  # or "class", "method", "module"
name = "example_function"
snippet = "# Code snippet that triggers the issue"

[[tests.negative]]
file = "test_negative/example_negative.py"
description = "TODO: Describe correct code that should NOT trigger detection"

[ai_science]
domains = []  # e.g., ["machine-learning", "deep-learning"]
audience = []  # e.g., ["ml-researchers", "data-scientists"]
paper_sections = []  # e.g., ["methods", "experimental-setup"]
impact_severity = ""  # critical, high, medium
educational_notes = """
TODO: Notes for teaching/learning about this pattern.
"""
'''  # nosec B608

        # Write pattern.toml
        toml_file = pattern_dir / "pattern.toml"
        toml_file.write_text(toml_content)

        # Create example test files
        self._create_example_test_files(pattern_dir)

        print(f"✓ Created pattern: {pattern_dir}")
        print("  - pattern.toml")
        print("  - test_positive/")
        print("  - test_negative/")
        print("  - test_context_dependent/")
        print("\nNext steps:")
        print(f"1. Edit {toml_file}")
        print("2. Add test files to test_positive/ and test_negative/")
        print(f"3. Validate: python -m scicode_lint.tools.validate_pattern {pattern_dir}")
        print("4. Rebuild registry: python -m scicode_lint.tools.rebuild_registry")

    def _create_example_test_files(self, pattern_dir: Path) -> None:
        """Create example test files with comments."""

        # Positive test example
        positive_example = pattern_dir / "test_positive" / "example_positive.py"
        positive_example.write_text(
            '''"""
Example positive test case.

This file should contain code that MUST trigger the pattern detection.
This is a True Positive case - the bug is definitely present.
"""

# TODO: Add code that demonstrates the bug/issue
# Example:
# def buggy_function():
#     # Code with the issue
#     pass
'''
        )

        # Negative test example
        negative_example = pattern_dir / "test_negative" / "example_negative.py"
        negative_example.write_text(
            '''"""
Example negative test case.

This file should contain CORRECT code that must NOT trigger detection.
This is a True Negative case - no bug is present.
"""

# TODO: Add correct code that should NOT trigger the pattern
# Example:
# def correct_function():
#     # Correct implementation
#     pass
'''
        )

        # Context-dependent example
        context_example = pattern_dir / "test_context_dependent" / "example_context.py"
        context_example.write_text(
            '''"""
Example context-dependent test case.

This file contains code where detection is acceptable either way.
These are edge cases where both detecting and not detecting can be valid
depending on coding style, use case, or interpretation.
"""

# TODO: Add edge case code where either outcome is acceptable
# Example:
# def ambiguous_case():
#     # Code where it's debatable whether this is an issue
#     pass
'''
        )


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Create a new pattern from template",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new pattern
  python -m scicode_lint.tools.new_pattern \\
    --id ml-999 \\
    --name my-pattern \\
    --category ai-training \\
    --severity critical

  # Create with default severity (high)
  python -m scicode_lint.tools.new_pattern \\
    --id num-050 \\
    --name integer-overflow \\
    --category scientific-numerical

Available categories:
  - ai-training
  - ai-inference
  - scientific-numerical
  - scientific-reproducibility
  - scientific-performance
""",
    )

    parser.add_argument("--id", required=True, help="Pattern ID (e.g., ml-999, pt-050)")
    parser.add_argument(
        "--name", required=True, help="Pattern name in kebab-case (e.g., my-pattern)"
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=[
            "ai-training",
            "ai-inference",
            "scientific-numerical",
            "scientific-reproducibility",
            "scientific-performance",
        ],
        help="Pattern category",
    )
    parser.add_argument(
        "--severity",
        default="high",
        choices=["critical", "high", "medium"],
        help="Severity level (default: high)",
    )
    parser.add_argument(
        "--patterns-dir",
        type=Path,
        default=Path("patterns"),
        help="Patterns directory (default: patterns)",
    )

    args = parser.parse_args()

    creator = PatternCreator(args.patterns_dir)

    try:
        creator.create_pattern(
            pattern_id=args.id,
            pattern_name=args.name,
            category=args.category,
            severity=args.severity,
        )
        return 0
    except FileExistsError as e:
        print(f"ERROR: {e}")
        return 1
    except Exception as e:
        print(f"ERROR: Failed to create pattern: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
