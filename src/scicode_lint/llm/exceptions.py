"""Custom exceptions for LLM operations."""

from typing import Any

from scicode_lint.exceptions import SciCodeLintError


class ContextLengthError(SciCodeLintError):
    """Raised when input exceeds model's context length.

    Attributes:
        file_path: Path to the file being checked
        estimated_tokens: Estimated input size in tokens
        max_tokens: Maximum context length supported
        message: Error message with helpful suggestions
    """

    def __init__(
        self,
        file_path: str,
        estimated_tokens: int,
        max_tokens: int,
    ):
        self.file_path = file_path
        self.estimated_tokens = estimated_tokens
        self.max_tokens = max_tokens

        # Create helpful error message
        overflow = estimated_tokens - max_tokens
        self.message = (
            f"File too large for context window\n"
            f"  File: {file_path}\n"
            f"  Estimated tokens: {estimated_tokens:,}\n"
            f"  Context limit: {max_tokens:,}\n"
            f"  Overflow: {overflow:,} tokens\n\n"
            f"Suggestions:\n"
            f"  • Split into smaller files (< {max_tokens:,} tokens)\n"
            f"  • Focus linting on specific functions/classes\n"
            f"  • Increase max_model_len when starting vLLM server\n"
            f"  • Use a model with larger context window"
        )
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to structured dictionary for GenAI agent consumption.

        Returns:
            Dictionary with error details and suggestions

        Example:
            >>> try:
            ...     check_large_file()
            ... except ContextLengthError as e:
            ...     error_data = e.to_dict()
            ...     print(error_data["suggestions"])
        """
        overflow = self.estimated_tokens - self.max_tokens
        return {
            "error": "ContextLengthError",
            "file_path": self.file_path,
            "estimated_tokens": self.estimated_tokens,
            "max_tokens": self.max_tokens,
            "overflow": overflow,
            "suggestions": [
                f"Split into smaller files (< {self.max_tokens:,} tokens)",
                "Focus linting on specific functions/classes",
                "Increase max_model_len when starting vLLM server",
                "Use a model with larger context window",
            ],
        }
