"""Clone GitHub repositories from filtered papers.

Async git clone with concurrency control, retries, and resumable progress.

Usage:
    python clone_repos.py [--max-concurrent 3] [--timeout 120]
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from real_world_demo.config import (
    CLONED_DIR,
    DATA_DIR,
    DEFAULT_CLONE_TIMEOUT,
    DEFAULT_MAX_CONCURRENT_CLONES,
)


class CloneResult:
    """Result of a single clone operation."""

    def __init__(
        self,
        repo_url: str,
        success: bool,
        error: str | None = None,
        repo_path: Path | None = None,
    ):
        self.repo_url = repo_url
        self.success = success
        self.error = error
        self.repo_path = repo_path

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "repo_url": self.repo_url,
            "success": self.success,
            "error": self.error,
            "repo_path": str(self.repo_path) if self.repo_path else None,
        }


def repo_url_to_path(repo_url: str, base_dir: Path) -> Path:
    """Convert repo URL to local path.

    Args:
        repo_url: GitHub repo URL.
        base_dir: Base directory for cloned repos.

    Returns:
        Local path for the cloned repo.
    """
    # Extract owner/repo from URL
    # https://github.com/owner/repo -> owner__repo
    parts = repo_url.rstrip("/").split("/")
    if len(parts) >= 2:
        owner = parts[-2]
        repo = parts[-1]
        return base_dir / f"{owner}__{repo}"
    return base_dir / parts[-1]


async def _symlink_if_redirected(requested_url: str, cloned_path: Path, output_dir: Path) -> None:
    """Create symlink if git remote URL differs from requested URL (GitHub redirect).

    For example, if we requested https://github.com/deepmind/graphcast but GitHub
    redirected to https://github.com/google-deepmind/graphcast, create a symlink
    google-deepmind__graphcast -> deepmind__graphcast so both paths work.

    Args:
        requested_url: The original URL we asked to clone.
        cloned_path: The local path where the repo was cloned.
        output_dir: Parent directory containing all cloned repos.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(cloned_path),
            "remote",
            "get-url",
            "origin",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await process.communicate()
        actual_url = stdout.decode().strip().rstrip("/").removesuffix(".git")
        requested_clean = requested_url.rstrip("/").removesuffix(".git")

        if actual_url and actual_url != requested_clean:
            # GitHub redirected - create symlink from the actual URL's path
            actual_path = repo_url_to_path(actual_url, output_dir)
            if not actual_path.exists():
                actual_path.symlink_to(cloned_path.name)
                logger.info(f"Created redirect symlink: {actual_path.name} -> {cloned_path.name}")
    except Exception as e:
        logger.debug(f"Could not check redirect for {requested_url}: {e}")


async def clone_repo(
    repo_url: str,
    output_dir: Path,
    timeout: int,
    semaphore: asyncio.Semaphore,
) -> CloneResult:
    """Clone a single repository.

    Args:
        repo_url: GitHub repo URL to clone.
        output_dir: Directory to clone into.
        timeout: Clone timeout in seconds.
        semaphore: Semaphore for concurrency control.

    Returns:
        CloneResult with success status and any error message.
    """
    repo_path = repo_url_to_path(repo_url, output_dir)

    # Skip if already cloned
    if repo_path.exists() and (repo_path / ".git").exists():
        logger.debug(f"Already cloned: {repo_url}")
        return CloneResult(repo_url, success=True, repo_path=repo_path)

    async with semaphore:
        try:
            logger.info(f"Cloning {repo_url}...")

            # Create shallow clone
            process = await asyncio.create_subprocess_exec(
                "git",
                "clone",
                "--depth",
                "1",
                "--single-branch",
                repo_url,
                str(repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except TimeoutError:
                process.kill()
                await process.wait()
                # Clean up partial clone
                if repo_path.exists():
                    import shutil

                    shutil.rmtree(repo_path)
                return CloneResult(repo_url, success=False, error="timeout")

            if process.returncode != 0:
                error_msg = stderr.decode().strip()

                # Categorize errors
                if "not found" in error_msg.lower() or "404" in error_msg:
                    error_type = "not_found"
                elif "authentication" in error_msg.lower() or "403" in error_msg:
                    error_type = "private"
                elif "rate limit" in error_msg.lower():
                    error_type = "rate_limited"
                else:
                    error_type = f"git_error: {error_msg[:100]}"

                # Clean up partial clone
                if repo_path.exists():
                    import shutil

                    shutil.rmtree(repo_path)

                return CloneResult(repo_url, success=False, error=error_type)

            logger.info(f"Cloned: {repo_url}")

            # Create symlink if GitHub redirected (e.g., deepmind -> google-deepmind)
            await _symlink_if_redirected(repo_url, repo_path, output_dir)

            return CloneResult(repo_url, success=True, repo_path=repo_path)

        except Exception as e:
            logger.error(f"Error cloning {repo_url}: {e}")
            return CloneResult(repo_url, success=False, error=str(e))


async def clone_all_repos(
    repos: list[dict[str, Any]],
    output_dir: Path,
    max_concurrent: int,
    timeout: int,
) -> list[CloneResult]:
    """Clone all repositories with concurrency control.

    Args:
        repos: List of repo records with repo_url field.
        output_dir: Directory to clone repos into.
        max_concurrent: Maximum concurrent clone operations.
        timeout: Clone timeout in seconds per repo.

    Returns:
        List of CloneResult for each repo.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [clone_repo(repo["repo_url"], output_dir, timeout, semaphore) for repo in repos]

    results = await asyncio.gather(*tasks)
    return list(results)


def load_repos(repos_file: Path) -> list[dict[str, Any]]:
    """Load repos from JSON file.

    Supports two formats:
    1. repos.json format: [{"repo_url": "...", ...}, ...]
    2. papers format (ai_science_papers.json): [{"repo_urls": ["...", ...], ...}, ...]

    The papers format is expanded to individual repo records.

    Args:
        repos_file: Path to repos.json or ai_science_papers.json file.

    Returns:
        List of repo records with repo_url field.

    Raises:
        FileNotFoundError: If repos file doesn't exist.
    """
    if not repos_file.exists():
        raise FileNotFoundError(
            f"Repos file not found: {repos_file}. "
            "Run filter_papers.py and filter_abstracts.py first."
        )

    with open(repos_file) as f:
        records: list[dict[str, Any]] = json.load(f)

    # Expand papers format (repo_urls list) to individual repo records
    repos = []
    seen_urls: set[str] = set()

    for record in records:
        if "repo_url" in record:
            # Already in repo format
            url = record["repo_url"]
            if url not in seen_urls:
                seen_urls.add(url)
                repos.append(record)
        elif "repo_urls" in record:
            # Paper format with list of repo URLs
            for url in record.get("repo_urls", []):
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    repos.append(
                        {
                            "repo_url": url,
                            "paper_url": record.get("paper_url", ""),
                            "paper_title": record.get("title", ""),
                            "domain": record.get("matched_domain", ""),
                            "tasks": record.get("tasks", []),
                            "arxiv_id": record.get("arxiv_id", ""),
                        }
                    )

    return repos


def save_results(results: list[CloneResult], output_file: Path) -> None:
    """Save clone results to JSON file.

    Args:
        results: List of CloneResult objects.
        output_file: Path to output JSON file.
    """
    with open(output_file, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    logger.info(f"Saved clone results to {output_file}")


def print_summary(results: list[CloneResult]) -> None:
    """Print summary statistics of clone results.

    Args:
        results: List of CloneResult objects.
    """
    total = len(results)
    successful = sum(1 for r in results if r.success)
    failed = total - successful

    logger.info("=" * 50)
    logger.info("Clone Summary:")
    logger.info(f"  Total repos: {total:,}")
    logger.info(f"  Successful: {successful:,} ({100 * successful / total:.1f}%)")
    logger.info(f"  Failed: {failed:,} ({100 * failed / total:.1f}%)")

    if failed > 0:
        error_counts: dict[str, int] = {}
        for r in results:
            if not r.success and r.error:
                error_type = r.error.split(":")[0] if ":" in r.error else r.error
                error_counts[error_type] = error_counts.get(error_type, 0) + 1

        logger.info("  Failure breakdown:")
        for error, count in sorted(error_counts.items(), key=lambda x: -x[1]):
            logger.info(f"    {error}: {count:,}")


def count_already_cloned(repos: list[dict[str, Any]], output_dir: Path) -> int:
    """Count how many repos are already cloned.

    Args:
        repos: List of repo records.
        output_dir: Directory containing cloned repos.

    Returns:
        Number of already cloned repos.
    """
    count = 0
    for repo in repos:
        repo_path = repo_url_to_path(repo["repo_url"], output_dir)
        if repo_path.exists() and (repo_path / ".git").exists():
            count += 1
    return count


def main() -> None:
    """Main entry point for repo cloning."""
    parser = argparse.ArgumentParser(description="Clone GitHub repos from filtered papers")
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT_CLONES,
        help=f"Maximum concurrent clone operations (default: {DEFAULT_MAX_CONCURRENT_CLONES})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_CLONE_TIMEOUT,
        help=f"Clone timeout per repo in seconds (default: {DEFAULT_CLONE_TIMEOUT})",
    )
    parser.add_argument(
        "--repos-file",
        type=Path,
        default=DATA_DIR / "sampled_papers.json",
        help="Path to papers JSON file (ai_science_papers.json or repos.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CLONED_DIR,
        help="Directory to clone repos into",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of repos to clone",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-clone by deleting existing clone directory",
    )
    args = parser.parse_args()

    # Handle --force: delete existing clones
    if args.force and args.output_dir.exists():
        import shutil

        logger.info(f"Force mode: deleting existing clones in {args.output_dir}")
        shutil.rmtree(args.output_dir)

    # Load repos
    repos = load_repos(args.repos_file)
    logger.info(f"Loaded {len(repos):,} repos from {args.repos_file}")

    if args.limit:
        repos = repos[: args.limit]
        logger.info(f"Limited to {len(repos):,} repos")

    # Check how many are already cloned (resumable)
    already_cloned = count_already_cloned(repos, args.output_dir)
    if already_cloned > 0:
        logger.info(f"Already cloned: {already_cloned:,} repos (will be skipped)")
        logger.info(f"Remaining to clone: {len(repos) - already_cloned:,} repos")

    # Clone repos (automatically skips already cloned)
    results = asyncio.run(
        clone_all_repos(
            repos,
            args.output_dir,
            args.max_concurrent,
            args.timeout,
        )
    )

    # Save results
    save_results(results, DATA_DIR / "clone_results.json")

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
