"""Doc cache functionality for reference URL fetching and caching.

Handles CHECK 14 (Reference URL Validation and Caching) and related utilities:
- Cache management (filename generation, validation, cleanup)
- URL reachability checks and content fetching
- HTML navigation stripping and vLLM-based content extraction
"""

import asyncio
import hashlib
import sys
import time
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# Add project root to sys.path so pattern_verification can be imported
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pattern_verification.deterministic.models import ValidationIssue, ValidationResult

# Cache settings for reference docs
DOC_CACHE_DIR = Path(__file__).parent / "doc_cache"
DOC_CACHE_RAW_DIR = DOC_CACHE_DIR / "raw"
DOC_CACHE_CLEAN_DIR = DOC_CACHE_DIR / "clean"
DOC_CACHE_MAX_AGE_DAYS = 7

# vLLM context: 16K tokens input ~ 64KB text. With ~100 chars/line, use 500 lines/chunk
VLLM_CHUNK_SIZE = 500
VLLM_CHUNK_OVERLAP = 100
MAX_REFERENCE_URLS = 5
MAX_DOC_LINES = 1000  # Warn if cached doc exceeds this (find more specific page)


# Pydantic model for vLLM doc extraction response
class DocCutResponse(BaseModel):
    """Response indicating which line ranges to cut."""

    cut: list[list[int]]  # List of [start, end] pairs


def get_cache_filename(url: str, pattern_id: str = "") -> str:
    """Get cache filename for a URL, optionally prefixed with pattern ID."""
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    try:
        from urllib.parse import urlparse

        domain = urlparse(url).netloc.replace(".", "_")
        if pattern_id:
            return f"{pattern_id}_{domain}_{url_hash}.md"
        return f"{domain}_{url_hash}.md"
    except Exception:
        if pattern_id:
            return f"{pattern_id}_{url_hash}.md"
        return f"{url_hash}.md"


def get_cache_path(url: str, pattern_id: str = "") -> Path:
    """Get clean cache file path for a URL (markdown format)."""
    return DOC_CACHE_CLEAN_DIR / get_cache_filename(url, pattern_id)


def get_raw_cache_path(url: str, pattern_id: str = "") -> Path:
    """Get raw cache file path for a URL (markdown format)."""
    return DOC_CACHE_RAW_DIR / get_cache_filename(url, pattern_id)


def is_cache_valid(cache_path: Path) -> bool:
    """Check if cache file exists and is not expired."""
    if not cache_path.exists():
        return False
    age_days = (time.time() - cache_path.stat().st_mtime) / (60 * 60 * 24)
    return age_days < DOC_CACHE_MAX_AGE_DAYS


def extract_doc_content_with_vllm(markdown_content: str) -> tuple[str, bool]:
    """Use local vLLM to extract documentation content with async chunked processing.

    Asks vLLM for line ranges to CUT (navigation/boilerplate), not content to keep.
    Uses existing VLLMClient infrastructure with guided_json for structured output.

    Returns:
        Tuple of (filtered_content, success). If vLLM fails, returns (original_content, False).
    """
    from scicode_lint.config import load_llm_config
    from scicode_lint.llm.client import LLMClient, create_client

    lines = markdown_content.split("\n")

    # If content is small, no vLLM needed
    if len(lines) < 50:
        return markdown_content, True

    # Check content length against max_input_tokens (~4 chars per token estimate)
    try:
        llm_config = load_llm_config()
        max_chars = llm_config.max_input_tokens * 4  # ~4 chars per token
        if len(markdown_content) > max_chars:
            # Content too large for vLLM context
            return markdown_content, False
    except Exception:
        return markdown_content, False

    # Build list of chunks with overlap
    chunks: list[tuple[int, int, str]] = []
    for start in range(0, len(lines), VLLM_CHUNK_SIZE - VLLM_CHUNK_OVERLAP):
        end = min(start + VLLM_CHUNK_SIZE, len(lines))
        chunk_lines = lines[start:end]
        # Number lines (line numbers are 1-indexed in the full file)
        numbered = "\n".join(f"{start + i + 1}: {line}" for i, line in enumerate(chunk_lines))
        chunks.append((start, end, numbered))

    async def process_chunk(client: LLMClient, numbered: str) -> tuple[set[int], bool]:
        """Process a single chunk and return (line numbers to CUT, success)."""
        system_prompt = (
            "You are a documentation parser. Identify navigation and boilerplate content to remove."
        )
        user_prompt = f"""CUT: nav menus, sidebars, footers, "skip to content", sign-in, social links, cookies.
KEEP: API docs, code, params, descriptions, warnings, notes.
Return {{"cut": [[start,end], ...]}} for line ranges to remove. Empty if nothing to cut.

{numbered}"""

        try:
            result = await client.async_complete_structured(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=DocCutResponse,
            )
            cut_lines: set[int] = set()
            for range_pair in result.cut:
                if len(range_pair) == 2:
                    cut_lines.update(range(range_pair[0], range_pair[1] + 1))
            return cut_lines, True
        except Exception:
            return set(), False

    async def process_all_chunks(client: LLMClient) -> tuple[set[int], bool]:
        """Process all chunks concurrently. Returns (cut_lines, all_succeeded)."""
        tasks = [process_chunk(client, numbered) for _, _, numbered in chunks]
        results = await asyncio.gather(*tasks)
        cut_lines: set[int] = set()
        all_succeeded = True
        for chunk_cut_lines, success in results:
            cut_lines.update(chunk_cut_lines)
            if not success:
                all_succeeded = False
        return cut_lines, all_succeeded

    # Run async processing using existing VLLMClient (with auto-detection)
    try:
        client = create_client(llm_config)
        cut_lines, vllm_success = asyncio.run(process_all_chunks(client))
    except Exception:
        return markdown_content, False

    if not vllm_success:
        return markdown_content, False

    # Keep lines NOT in cut_lines
    filtered = "\n".join(line for i, line in enumerate(lines, 1) if i not in cut_lines)
    return filtered, True


def strip_html_nav_elements(html_content: str) -> str:
    """Strip navigation elements from HTML before markdown conversion.

    Removes: nav, footer, header (if it looks like site header), aside, and common nav classes.
    Uses Python's built-in html.parser for reliability.
    """
    from html.parser import HTMLParser

    # Tags to remove entirely (including content)
    nav_tags = {"nav", "footer", "aside", "header"}
    # Tags to check for nav-like classes (substring match)
    nav_classes = {
        "navbar",
        "sidebar",
        "site-header",
        "site-footer",
        "breadcrumb",
        "toc",
        "menu",
        "topnav",
        "header-nav",
        "footer-nav",
        "skip-link",
        "mobile-menu",
    }

    class NavStripper(HTMLParser):
        def __init__(self) -> None:
            super().__init__()
            self.result: list[str] = []
            self.skip_depth = 0  # When > 0, skip content
            self.skip_tag_stack: list[str] = []

        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            attrs_dict = dict(attrs)
            class_attr = attrs_dict.get("class", "") or ""
            id_attr = attrs_dict.get("id", "") or ""

            # Check if this is a nav element to skip
            should_skip = tag in nav_tags
            if not should_skip:
                # Check for nav-like classes/ids
                for nav_class in nav_classes:
                    if nav_class in class_attr.lower() or nav_class in id_attr.lower():
                        should_skip = True
                        break

            if should_skip:
                self.skip_depth += 1
                self.skip_tag_stack.append(tag)
            elif self.skip_depth == 0:
                # Rebuild the tag
                attr_str = " ".join(f'{k}="{v}"' if v else k for k, v in attrs)
                if attr_str:
                    self.result.append(f"<{tag} {attr_str}>")
                else:
                    self.result.append(f"<{tag}>")

        def handle_endtag(self, tag: str) -> None:
            if self.skip_depth > 0 and self.skip_tag_stack and self.skip_tag_stack[-1] == tag:
                self.skip_depth -= 1
                self.skip_tag_stack.pop()
            elif self.skip_depth == 0:
                self.result.append(f"</{tag}>")

        def handle_data(self, data: str) -> None:
            if self.skip_depth == 0:
                self.result.append(data)

        def handle_comment(self, data: str) -> None:
            if self.skip_depth == 0:
                self.result.append(f"<!--{data}-->")

        def handle_decl(self, decl: str) -> None:
            self.result.append(f"<!{decl}>")

        def get_result(self) -> str:
            return "".join(self.result)

    parser = NavStripper()
    try:
        parser.feed(html_content)
        return parser.get_result()
    except Exception:
        # If parsing fails, return original content
        return html_content


def fetch_and_cache_url(url: str, result: ValidationResult, pattern_id: str = "") -> bool:
    """Fetch URL content and cache it as markdown in raw/ and clean/ subdirectories."""
    import html2text
    import httpx

    clean_path = get_cache_path(url, pattern_id)
    raw_path = get_raw_cache_path(url, pattern_id)

    # Check if cache is still valid (prefer clean, fall back to raw)
    if is_cache_valid(clean_path) or is_cache_valid(raw_path):
        return True

    # Ensure cache directories exist
    DOC_CACHE_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DOC_CACHE_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    try:
        with httpx.Client(
            timeout=10.0,
            follow_redirects=True,
            headers={"User-Agent": "scicode-lint/1.0 (pattern-verification)"},
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            html_content = response.text

        # Setup markdown converter
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_emphasis = False
        h.body_width = 0  # Don't wrap lines

        # Convert raw HTML to markdown (for raw cache)
        markdown_content = h.handle(html_content)

        # Check for empty/minimal content (likely redirect or fetch failure)
        content_chars = len(markdown_content.strip())
        if content_chars < 100:
            result.issues.append(
                ValidationIssue(
                    "warning",
                    "reference_url",
                    f"Fetch returned minimal content ({content_chars} chars) for {url} "
                    "- check for redirects",
                )
            )
            return False

        header = f"# Source: {url}\n# Fetched: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"

        # Save raw markdown
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(header)
            f.write(markdown_content)

        # Step 1: Strip HTML nav elements (deterministic, fast)
        stripped_html = strip_html_nav_elements(html_content)
        clean_markdown = h.handle(stripped_html)

        # Step 2: Pass through vLLM for additional cleaning
        # HTML stripping handles structural nav, vLLM catches remaining boilerplate
        original_len = len(markdown_content)
        html_stripped_len = len(clean_markdown)
        html_reduction = (1 - html_stripped_len / original_len) * 100 if original_len > 0 else 0

        # Pass HTML-stripped content to vLLM for further cleaning
        filtered_content, vllm_success = extract_doc_content_with_vllm(clean_markdown)

        if not vllm_success:
            # vLLM not available or content too large - warn and don't save to clean/
            result.issues.append(
                ValidationIssue(
                    "warning",
                    "reference_url",
                    f"vLLM unavailable for {url} (HTML stripped: {html_reduction:.0f}%)",
                )
            )
            return True  # Raw file still saved

        filtered_len = len(filtered_content)
        total_reduction = (1 - filtered_len / original_len) * 100 if original_len > 0 else 0

        # Save to clean/ if we achieved significant total reduction
        if filtered_len < original_len * 0.9:
            with open(clean_path, "w", encoding="utf-8") as f:
                f.write(header)
                f.write(filtered_content)

            # Check if doc is too large (suggests unfocused reference)
            line_count = filtered_content.count("\n") + 1
            if line_count > MAX_DOC_LINES:
                result.issues.append(
                    ValidationIssue(
                        "warning",
                        "reference_url",
                        f"Doc too large ({line_count} lines > {MAX_DOC_LINES}): "
                        f"{url} - find more focused page",
                    )
                )
        else:
            result.issues.append(
                ValidationIssue(
                    "warning",
                    "reference_url",
                    f"Doc cleaning ineffective for {url} (total: {total_reduction:.0f}%)",
                )
            )

        return True

    except httpx.HTTPStatusError as e:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                f"HTTP {e.response.status_code} fetching {url}",
            )
        )
        return False
    except httpx.RequestError as e:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                f"Failed to fetch {url}: {e}",
            )
        )
        return False
    except Exception as e:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                f"Error fetching {url}: {e}",
            )
        )
        return False


def check_url_reachable(url: str, result: ValidationResult) -> bool:
    """Check if URL is reachable with HEAD request. Returns True if OK."""
    import httpx

    try:
        with httpx.Client(
            timeout=5.0,
            follow_redirects=True,
            headers={"User-Agent": "scicode-lint/1.0 (pattern-verification)"},
        ) as client:
            response = client.head(url)
            response.raise_for_status()
            return True

    except httpx.HTTPStatusError as e:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                f"HTTP {e.response.status_code} for {url}",
            )
        )
        return False
    except httpx.RequestError as e:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                f"Cannot reach {url}: {e}",
            )
        )
        return False
    except Exception as e:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                f"Error checking {url}: {e}",
            )
        )
        return False


def check_reference_urls(
    toml_data: dict[str, Any], result: ValidationResult, fetch: bool = False
) -> None:
    """Check reference URLs (HEAD request by default, full fetch with --fetch-refs)."""
    meta = toml_data.get("meta", {})
    references = meta.get("references", [])
    if not references:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                "No reference URLs - add links to official documentation",
            )
        )
        return

    if len(references) > MAX_REFERENCE_URLS:
        result.issues.append(
            ValidationIssue(
                "warning",
                "reference_url",
                f"Too many references ({len(references)}), max is {MAX_REFERENCE_URLS}",
            )
        )

    for url in references:
        if not isinstance(url, str):
            result.issues.append(
                ValidationIssue("error", "reference_url", f"Invalid reference: {url}")
            )
            continue

        if not url.startswith(("http://", "https://")):
            result.issues.append(
                ValidationIssue("error", "reference_url", f"Invalid URL format: {url}")
            )
            continue

        if fetch:
            # Full fetch and cache (prefix with pattern ID for easy lookup)
            fetch_and_cache_url(url, result, result.pattern_id)
        else:
            # Lightweight HEAD check (default)
            check_url_reachable(url, result)


def collect_all_pattern_references(patterns_dir: Path) -> list[tuple[str, str]]:
    """Collect all (pattern_id, url) pairs from all patterns."""
    refs: list[tuple[str, str]] = []
    for category in patterns_dir.iterdir():
        if not category.is_dir() or category.name.startswith((".", "_")):
            continue
        for pattern_dir in category.iterdir():
            if not pattern_dir.is_dir():
                continue
            toml_path = pattern_dir / "pattern.toml"
            if not toml_path.exists():
                continue
            pattern_id = pattern_dir.name
            try:
                with open(toml_path, "rb") as f:
                    data = tomllib.load(f)
                for url in data.get("meta", {}).get("references", []):
                    if isinstance(url, str):
                        refs.append((pattern_id, url))
            except Exception:
                continue
    return refs


def clean_orphaned_cache(patterns_dir: Path) -> list[str]:
    """Remove cached docs that are no longer referenced by any pattern.

    Returns list of removed files.
    """
    if not DOC_CACHE_DIR.exists():
        return []

    # Get all referenced (pattern_id, url) pairs and their cache filenames
    refs = collect_all_pattern_references(patterns_dir)
    referenced_filenames = {get_cache_filename(url, pattern_id) for pattern_id, url in refs}

    # Find and remove orphaned cache files from both raw/ and clean/ subdirs
    # Also clean up legacy files in root doc_cache/
    removed: list[str] = []
    dirs_to_check = [DOC_CACHE_DIR, DOC_CACHE_RAW_DIR, DOC_CACHE_CLEAN_DIR]

    for cache_dir in dirs_to_check:
        if not cache_dir.exists():
            continue
        for file_pattern in ("*.md", "*.txt"):
            for cache_file in cache_dir.glob(file_pattern):
                if cache_file.name not in referenced_filenames:
                    cache_file.unlink()
                    removed.append(f"{cache_dir.name}/{cache_file.name}")

    return removed
