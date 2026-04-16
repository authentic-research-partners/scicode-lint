"""File classification for self-contained ML workflow detection.

This module classifies Python files to determine if they contain complete
ML workflows (self-contained) or are just fragments requiring other files.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

from scicode_lint.llm.client import LLMClient

# Bounded short-string type used inside list fields of response schemas.
# See `llm/models.py` module docstring and `llm/CONSTRAINED_DECODING.md` for the
# per-field length policy.
ShortTag = Annotated[str, StringConstraints(max_length=80)]

# System prompt for classification
# Following code-first pattern: simple instructions that don't need to come before code
CLASSIFY_SYSTEM_PROMPT = """You classify Python files for ML/AI analysis.

A file is SELF-CONTAINED if it has a complete ML workflow:
- Loads or generates data (not just receives it as parameter)
- Defines or imports a model
- Trains or fits the model
- Has evaluation, prediction, or saves results

A file is FRAGMENT if it:
- Only defines functions/classes without running them
- Expects data/model to be passed in from elsewhere
- Is a utility module (helpers, constants, configs)
- Is part of a larger pipeline but not the entry point

Output JSON only."""

# User prompt template - code comes first for vLLM prefix caching
CLASSIFY_USER_PROMPT = """<CODE_TO_ANALYZE>
{code}
</CODE_TO_ANALYZE>

Classify this file and return JSON with these exact fields:

1. "classification": one of "self_contained", "fragment", or "uncertain"
   - "self_contained": Has complete ML workflow (data -> model -> train -> output)
   - "fragment": Partial code, needs other files to run
   - "uncertain": Can't determine (e.g., dynamic imports)

2. "confidence": float between 0.0 and 1.0

3. "entry_point_indicators": list of signs this runs directly (e.g., "if __name__", "argparse").
   At most 10 items, most important first. Empty list if none.

4. "missing_components": list of what's expected from elsewhere (e.g., "data loading", "model training").
   At most 10 items, most important first. Empty list if none.

5. "reasoning": brief explanation of your classification decision (under 100 words)
"""


class FileClassification(BaseModel):
    """Classification result for a Python file.

    Attributes:
        classification: Whether the file is self-contained, a fragment, or uncertain.
        confidence: Model's confidence in the classification (0.0-1.0).
        entry_point_indicators: Signs this file runs directly (if __name__, argparse, etc).
        missing_components: What's expected from elsewhere (data loading, model def, etc).
        reasoning: Brief explanation of the classification decision.
    """

    classification: Literal["self_contained", "fragment", "uncertain"] = Field(
        description="Whether the file has a complete ML workflow"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the classification (0.0-1.0)",
    )
    entry_point_indicators: list[ShortTag] = Field(
        default_factory=list,
        max_length=10,
        description="Signs this runs directly: if __name__, argparse, etc.",
    )
    missing_components: list[ShortTag] = Field(
        default_factory=list,
        max_length=10,
        description="What's expected from elsewhere: data loading, model def, etc.",
    )
    reasoning: str = Field(
        max_length=400,
        description="Brief explanation of the classification decision",
    )


async def classify_file(code: str, llm_client: LLMClient) -> FileClassification:
    """Classify a Python file as self-contained or fragment.

    Args:
        code: Python source code to classify.
        llm_client: LLM client for making classification requests.

    Returns:
        FileClassification with the classification result.

    Example:
        >>> from scicode_lint.config import load_llm_config
        >>> from scicode_lint.llm.client import create_client
        >>> llm_config = load_llm_config()
        >>> client = create_client(llm_config)
        >>> code = '''
        ... import pandas as pd
        ... from sklearn.ensemble import RandomForestClassifier
        ... data = pd.read_csv("data.csv")
        ... model = RandomForestClassifier()
        ... model.fit(data.drop("target", axis=1), data["target"])
        ... '''
        >>> result = await classify_file(code, client)
        >>> result.classification
        'self_contained'
    """
    user_prompt = CLASSIFY_USER_PROMPT.format(code=code)

    result = await llm_client.async_complete_structured(
        system_prompt=CLASSIFY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema=FileClassification,
        thinking_budget=200,
        thinking_effort=0.3,
    )

    return result
