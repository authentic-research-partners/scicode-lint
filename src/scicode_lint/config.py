"""Configuration for scicode-lint with Pydantic validation."""

import functools
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from scicode_lint.exceptions import ConfigError


def _load_bundled_config() -> dict[str, Any]:
    """Load defaults from bundled config.toml (single source of truth).

    Raises:
        RuntimeError: If config.toml file not found
    """

    # Bundled config.toml is the source of truth for defaults
    package_config = Path(__file__).parent / "config.toml"
    if not package_config.exists():
        raise RuntimeError(
            f"Bundled config.toml not found at {package_config}. "
            "This indicates a broken installation. Reinstall scicode-lint."
        )

    with open(package_config, "rb") as f:
        return tomllib.load(f)


# Load defaults at module init - config.toml is single source of truth
_BUNDLED = _load_bundled_config()
_LLM_DEFAULTS = _BUNDLED.get("llm", {})
_PREPROCESSING_DEFAULTS = _BUNDLED.get("preprocessing", {})


class Severity(StrEnum):
    """Issue severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"


class LLMConfig(BaseSettings):
    """
    LLM client configuration with validation and env var support.

    Defaults are loaded from bundled config.toml (single source of truth).

    Override via environment variables:
    - SCICODE_LINT_TEMPERATURE: Sampling temperature
    - SCICODE_LINT_TOP_P: Top-p sampling
    - SCICODE_LINT_TOP_K: Top-k sampling
    - SCICODE_LINT_TIMEOUT: Timeout in seconds
    - SCICODE_LINT_MAX_INPUT_TOKENS: Maximum input tokens
    - SCICODE_LINT_MAX_COMPLETION_TOKENS: Maximum output tokens
    - OPENAI_BASE_URL: vLLM server URL

    See config.toml for recommended values and documentation.
    """

    base_url: str = Field(
        default_factory=lambda: _LLM_DEFAULTS.get("base_url", ""),
        description="API base URL (empty = auto-detect)",
    )
    model: str = Field(
        default_factory=lambda: _LLM_DEFAULTS["model"],
        description="Model path for starting vLLM (HuggingFace model ID)",
    )
    model_served_name: str = Field(
        default_factory=lambda: _LLM_DEFAULTS["model_served_name"],
        description="Served model name for API calls (must match vLLM --served-model-name)",
    )
    timeout: int = Field(
        default_factory=lambda: _LLM_DEFAULTS["timeout"],
        ge=10,
        le=600,
        description="Request timeout in seconds",
    )
    temperature: float = Field(
        default_factory=lambda: _LLM_DEFAULTS["temperature"],
        ge=0.0,
        le=1.0,
        description="Sampling temperature (see config.toml for Qwen3 recommendations)",
    )
    top_p: float = Field(
        default_factory=lambda: _LLM_DEFAULTS["top_p"],
        ge=0.0,
        le=1.0,
        description="Top-p (nucleus) sampling",
    )
    top_k: int = Field(
        default_factory=lambda: _LLM_DEFAULTS["top_k"],
        ge=1,
        le=100,
        description="Top-k sampling",
    )
    max_input_tokens: int = Field(
        default_factory=lambda: _LLM_DEFAULTS.get("max_input_tokens", 16000),
        ge=1000,
        le=32000,
        description="Maximum input tokens (code + prompts)",
    )
    max_completion_tokens: int = Field(
        default_factory=lambda: _LLM_DEFAULTS.get("max_completion_tokens", 4096),
        ge=256,
        le=32768,
        description="Maximum output tokens (thinking + response)",
    )
    thinking_budget: int = Field(
        default_factory=lambda: _LLM_DEFAULTS.get("thinking_budget", 3584),
        ge=0,
        le=32768,
        description="Thinking token budget — hard cap (0 = no cap)",
    )
    thinking_effort: float | None = Field(
        default_factory=lambda: _LLM_DEFAULTS.get("thinking_effort"),
        description="Thinking effort — soft guide (0.0-1.0, None = omit)",
    )

    @property
    def max_model_len(self) -> int:
        """Total context length (max_input_tokens + max_completion_tokens)."""
        return self.max_input_tokens + self.max_completion_tokens

    model_config = SettingsConfigDict(
        env_prefix="SCICODE_LINT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    def __init__(self, **data: Any) -> None:
        """Initialize and apply standard OpenAI env vars if present."""
        import os

        # Standard OpenAI env vars take precedence
        if not data.get("base_url") and os.getenv("OPENAI_BASE_URL"):
            data["base_url"] = os.getenv("OPENAI_BASE_URL")
        super().__init__(**data)


class LinterConfig:
    """Linter configuration (not using Pydantic to keep simple)."""

    def __init__(
        self,
        patterns_dir: Path | None = None,
        llm_config: LLMConfig | None = None,
        min_confidence: float = 0.7,
        enabled_severities: set[Severity] | None = None,
        enabled_patterns: set[str] | None = None,
        enabled_categories: set[str] | None = None,
        max_concurrent: int = 60,
    ):
        self.patterns_dir = patterns_dir or get_default_patterns_dir()
        self.llm_config = llm_config or load_llm_config()
        self.min_confidence = min_confidence
        self.enabled_severities = enabled_severities or {
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
        }
        self.enabled_patterns = enabled_patterns
        self.enabled_categories = enabled_categories
        self.max_concurrent = max_concurrent


def get_default_patterns_dir() -> Path:
    """
    Get path to patterns directory.

    Searches in order:
    1. Package-bundled patterns (works after pip install)
    2. ~/.config/scicode-lint/patterns (user patterns override)

    Returns:
        Path to patterns directory
    """
    # Priority 1: Package-bundled patterns
    pkg_patterns = Path(__file__).parent / "patterns"
    if pkg_patterns.exists():
        return pkg_patterns

    # Priority 2: User patterns directory (for custom patterns)
    user_patterns = Path.home() / ".config" / "scicode-lint" / "patterns"
    if user_patterns.exists():
        return user_patterns

    raise RuntimeError(
        f"No patterns directory found. Expected packaged patterns at {pkg_patterns} "
        f"or user patterns at {user_patterns}. Reinstall scicode-lint if the bundled "
        "patterns directory is missing."
    )


@functools.lru_cache(maxsize=1)
def load_config_from_toml() -> dict[str, Any]:
    """
    Load configuration from config.toml (cached after first call).

    Searches in order:
    1. $SCICODE_LINT_CONFIG (env var pointing to config file)
    2. ./config.toml (current directory)
    3. ~/.config/scicode-lint/config.toml (user config)
    4. Package default (built-in)

    Returns:
        Parsed config dict (empty if file not found)
    """
    import os

    # Priority 1: Environment variable
    config_path_env = os.getenv("SCICODE_LINT_CONFIG")
    if config_path_env:
        config_path = Path(config_path_env)
        if config_path.exists():
            with open(config_path, "rb") as f:
                return tomllib.load(f)
        else:
            raise FileNotFoundError(f"SCICODE_LINT_CONFIG={config_path_env} not found")

    # Priority 2: Current directory
    config_path = Path("config.toml")
    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)

    # Priority 3: User config directory
    user_config = Path.home() / ".config" / "scicode-lint" / "config.toml"
    if user_config.exists():
        with open(user_config, "rb") as f:
            return tomllib.load(f)

    # Priority 4: Package default
    package_config = Path(__file__).parent / "config.toml"
    if package_config.exists():
        with open(package_config, "rb") as f:
            return tomllib.load(f)

    return {}


def load_llm_config() -> LLMConfig:
    """
    Load LLM configuration from TOML file with environment variable overrides.

    Priority (highest to lowest):
    1. Environment variables (SCICODE_LINT_*)
    2. .env file
    3. config.toml file
    4. Defaults

    Returns:
        Validated LLMConfig instance
    """
    # Load TOML config
    toml_config = load_config_from_toml()
    llm_toml = toml_config.get("llm", {})

    # Pydantic Settings will automatically:
    # - Load from TOML values (passed as kwargs)
    # - Override with env vars (SCICODE_LINT_*)
    # - Validate types and constraints
    return LLMConfig(**llm_toml)


def get_default_config() -> LinterConfig:
    """Get default linter configuration with TOML/env overrides."""
    toml_config = load_config_from_toml()
    if "performance" not in toml_config or "lint_concurrency" not in toml_config["performance"]:
        raise ConfigError(
            "config.toml is missing [performance].lint_concurrency — "
            "required for linter concurrency"
        )
    return LinterConfig(
        patterns_dir=get_default_patterns_dir(),
        llm_config=load_llm_config(),
        max_concurrent=toml_config["performance"]["lint_concurrency"],
    )


def get_ml_import_keywords() -> list[str]:
    """Get ML import keywords for preprocessing filter.

    Returns:
        List of keywords that indicate ML-related code.
    """
    toml_config = load_config_from_toml()
    preprocessing = toml_config.get("preprocessing", {})
    keywords = preprocessing.get(
        "ml_import_keywords",
        _PREPROCESSING_DEFAULTS.get("ml_import_keywords", []),
    )
    return list(keywords) if keywords else []


def get_filter_concurrency() -> int:
    """Get max concurrent LLM requests for file filtering (--filter-concurrency).

    Returns:
        Maximum concurrent requests for file classification.
    """
    toml_config = load_config_from_toml()
    preprocessing = toml_config.get("preprocessing", {})
    value = preprocessing.get(
        "filter_concurrency",
        _PREPROCESSING_DEFAULTS.get("filter_concurrency", 50),
    )
    return int(value) if value else 50


def get_strip_comments() -> bool:
    """Get whether to strip comments before LLM analysis.

    Returns:
        True if comments should be stripped (default), False otherwise.
    """
    toml_config = load_config_from_toml()
    preprocessing = toml_config.get("preprocessing", {})
    value = preprocessing.get(
        "strip_comments",
        _PREPROCESSING_DEFAULTS.get("strip_comments", True),
    )
    return bool(value)
