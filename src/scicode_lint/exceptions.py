"""User-facing exception hierarchy for scicode-lint.

All documented failure modes inherit from `SciCodeLintError`. Catching this
base class guarantees a clean error path for any expected linter error
(vLLM unreachable, context overflow, notebook parse failure, etc.).

Specialized subclasses live next to the code that raises them:
- `LLMConnectionError` — here (raised by `llm.client.detect_vllm`)
- `ContextLengthError` — `scicode_lint.llm.exceptions`
- `MissingLocationError` — `scicode_lint.llm.client`
- `NotebookParseError` — `scicode_lint.linter`

All of the above inherit from `SciCodeLintError`.
"""

from __future__ import annotations


class SciCodeLintError(Exception):
    """Base class for all user-facing scicode-lint errors."""


class LLMConnectionError(SciCodeLintError):
    """vLLM server unreachable.

    Raised when no vLLM server responds on any expected port. The error
    message includes the command to start the server.
    """


class ConfigError(SciCodeLintError):
    """Required configuration missing or invalid.

    Raised when config.toml is missing a key that has no safe default.
    The error message names the exact missing key.
    """


__all__ = [
    "SciCodeLintError",
    "LLMConnectionError",
    "ConfigError",
]
