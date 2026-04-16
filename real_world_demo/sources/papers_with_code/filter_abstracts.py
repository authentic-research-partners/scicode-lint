"""Filter papers by abstract using vLLM to identify AI+science papers.

Uses LLM to semantically determine if a paper applies AI/ML to real scientific
problems vs. being pure ML methodology research.

Usage:
    python filter_abstracts.py [--max-concurrent 20]
"""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, Field

from real_world_demo.config import (
    DATA_DIR,
    DEFAULT_ABSTRACT_FILTER_CONCURRENT,
)
from scicode_lint.config import load_llm_config
from scicode_lint.llm.client import create_client


class AbstractFilterResult(BaseModel):
    """Result of abstract filtering with LLM-assigned categorization."""

    is_ai_science: bool = Field(
        description="True if paper applies AI/ML to solve a real scientific problem"
    )
    confidence: float = Field(
        description="Confidence in the classification (0.0 to 1.0)", ge=0.0, le=1.0
    )
    science_domain: Literal[
        "biology",
        "chemistry",
        "medicine",
        "physics",
        "materials",
        "neuroscience",
        "earth_science",
        "astronomy",
        "economics",
        "social_science",
        "engineering",
        "none",
    ] = Field(description="Scientific domain, or 'none' if not applied to science")
    application_type: Literal[
        "prediction",
        "analysis",
        "discovery",
        "simulation",
        "diagnosis",
        "methodology",
    ] = Field(description="What the AI/ML is used for; 'methodology' if pure ML research")
    explanation: str = Field(
        max_length=400,
        description="Brief explanation of the decision (1-2 sentences)",
    )


# Abstract filter prompt - identifies AI applied to science with categorization
ABSTRACT_FILTER_PROMPT = """Classify this paper: Does it apply AI/ML to solve a real scientific research problem?

INCLUDE (is_ai_science: true) if the PRIMARY goal is:
- Scientific discovery or analysis in a domain (biology, chemistry, physics, medicine, earth science, etc.)
- Predicting scientific outcomes (protein structure, drug efficacy, material properties, climate, disease, etc.)
- Processing or analyzing scientific data (genomics, medical imaging, simulations, sensor data, etc.)
- Solving domain-specific scientific problems using computational methods

EXCLUDE (is_ai_science: false) if the PRIMARY goal is:
- Advancing ML/AI methodology itself (new architectures, training techniques, optimization)
- Using scientific data only as a benchmark to evaluate an ML method
- General-purpose AI (language models, image recognition, speech) without specific scientific application
- Pure theoretical ML/statistics research

ABSTRACT:
{abstract}

Respond as JSON with these fields:
- is_ai_science: true/false
- confidence: 0.0-1.0 (how certain you are)
- science_domain: biology, chemistry, medicine, physics, materials, neuroscience, earth_science, astronomy, economics, social_science, engineering, or "none"
- application_type: prediction, analysis, discovery, simulation, diagnosis, or "methodology"
- explanation: brief reason"""


SYSTEM_PROMPT = """You are a scientific paper classifier. Analyze paper abstracts and determine if they apply AI/ML to solve real scientific problems. Be precise and brief."""


async def check_paper_with_llm(
    paper: dict[str, Any],
    llm_client: Any,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Check if a paper is AI+science using LLM.

    Args:
        paper: Paper record with abstract.
        llm_client: VLLMClient instance.
        semaphore: Semaphore for concurrency control.

    Returns:
        Dict with paper_url, is_ai_science, and explanation.
    """
    paper_url = paper.get("paper_url", "unknown")
    abstract = paper.get("abstract", "")

    async with semaphore:
        # Skip papers without abstracts
        if not abstract or len(abstract.strip()) < 50:
            logger.debug(f"No abstract: {paper_url}")
            return {
                "paper_url": paper_url,
                "is_ai_science": False,
                "explanation": "no_abstract",
            }

        try:
            # Truncate very long abstracts (keep first ~2000 chars)
            if len(abstract) > 2000:
                abstract = abstract[:2000] + "..."

            prompt = ABSTRACT_FILTER_PROMPT.format(abstract=abstract)

            result = await llm_client.async_complete_structured(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                schema=AbstractFilterResult,
                thinking_budget=200,
                thinking_effort=0.3,
            )

            title = paper.get("title", "")[:50]
            logger.debug(
                f"{'✓' if result.is_ai_science else '✗'} [{result.science_domain}] {title}..."
            )
            return {
                "paper_url": paper_url,
                "is_ai_science": result.is_ai_science,
                "confidence": result.confidence,
                "science_domain": result.science_domain,
                "application_type": result.application_type,
                "explanation": result.explanation,
            }

        except TimeoutError:
            logger.warning(f"Timeout checking {paper_url}")
            return {
                "paper_url": paper_url,
                "is_ai_science": True,  # Include on timeout (conservative)
                "error": "timeout",
            }
        except Exception as e:
            logger.warning(f"Error checking {paper_url}: {e}")
            return {
                "paper_url": paper_url,
                "is_ai_science": True,  # Include on error (conservative)
                "error": str(e),
            }


async def filter_papers_by_abstract(
    papers: list[dict[str, Any]],
    llm_client: Any,
    max_concurrent: int = 20,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter papers by abstract using LLM.

    Args:
        papers: List of paper records with abstracts.
        llm_client: VLLMClient instance.
        max_concurrent: Maximum concurrent LLM requests.

    Returns:
        Tuple of (ai_science_papers, excluded_papers).
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    tasks = [check_paper_with_llm(paper, llm_client, semaphore) for paper in papers]

    logger.info(f"Filtering {len(tasks)} papers by abstract with LLM...")
    results = await asyncio.gather(*tasks)

    # Partition into AI+science and excluded papers
    ai_science_papers = []
    excluded_papers = []

    for paper, result in zip(papers, results, strict=True):
        # Store all filter results in the paper record
        paper["llm_classification"] = {
            "is_ai_science": result.get("is_ai_science", False),
            "confidence": result.get("confidence", 0.0),
            "science_domain": result.get("science_domain", "none"),
            "application_type": result.get("application_type", "methodology"),
            "explanation": result.get("explanation", result.get("error", "")),
        }
        # Also store domain at top level for easy access (replaces keyword-based domain)
        if result.get("is_ai_science", False):
            paper["matched_domain"] = result.get("science_domain", "unknown")
            ai_science_papers.append(paper)
        else:
            excluded_papers.append(paper)

    return ai_science_papers, excluded_papers


def sample_by_domain(
    papers: list[dict[str, Any]],
    target_count: int,
    min_confidence: float = 0.7,
) -> list[dict[str, Any]]:
    """Sample papers balanced by LLM-assigned domain, prioritizing high confidence.

    Args:
        papers: Papers with llm_classification.
        target_count: Target number of papers to sample.
        min_confidence: Minimum confidence threshold.

    Returns:
        Sampled papers balanced across domains.
    """
    import random

    # Filter by confidence
    confident_papers = [
        p for p in papers if p.get("llm_classification", {}).get("confidence", 0) >= min_confidence
    ]
    logger.info(f"  Papers with confidence >= {min_confidence}: {len(confident_papers)}")

    # Group by LLM-assigned domain
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for paper in confident_papers:
        domain = paper.get("matched_domain", "unknown")
        if domain not in by_domain:
            by_domain[domain] = []
        by_domain[domain].append(paper)

    # Sort each domain by confidence (highest first)
    for domain in by_domain:
        by_domain[domain].sort(
            key=lambda p: p.get("llm_classification", {}).get("confidence", 0), reverse=True
        )

    # Calculate per-domain allocation
    n_domains = len(by_domain)
    if n_domains == 0:
        return []

    per_domain = target_count // n_domains
    remainder = target_count % n_domains

    sampled: list[dict[str, Any]] = []
    domains_sorted = sorted(by_domain.keys())
    random.shuffle(domains_sorted)  # Randomize which domains get remainder

    for i, domain in enumerate(domains_sorted):
        domain_papers = by_domain[domain]
        # Give extra to first 'remainder' domains
        count = per_domain + (1 if i < remainder else 0)
        sampled.extend(domain_papers[:count])
        logger.info(f"    {domain}: {min(len(domain_papers), count)} papers")

    return sampled


def load_filtered_papers(input_file: Path) -> list[dict[str, Any]]:
    """Load filtered papers from JSON.

    Args:
        input_file: Path to filtered_papers.json.

    Returns:
        List of paper records.
    """
    if not input_file.exists():
        raise FileNotFoundError(
            f"Filtered papers not found: {input_file}. Run filter_papers.py first."
        )

    with open(input_file) as f:
        result: list[dict[str, Any]] = json.load(f)
        return result


def save_results(
    ai_science_papers: list[dict[str, Any]],
    excluded_papers: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save abstract filter results.

    Args:
        ai_science_papers: Papers identified as AI+science.
        excluded_papers: Papers excluded (pure ML).
        output_dir: Output directory.
    """
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save AI+science papers (these will proceed to cloning)
    ai_science_file = output_dir / "ai_science_papers.json"
    with open(ai_science_file, "w") as f:
        json.dump(ai_science_papers, f, indent=2)
    logger.info(f"Saved {len(ai_science_papers)} AI+science papers to {ai_science_file}")

    # Save excluded papers (for review/debugging)
    excluded_file = output_dir / "abstract_excluded.json"
    with open(excluded_file, "w") as f:
        json.dump(excluded_papers, f, indent=2)
    logger.info(f"Saved {len(excluded_papers)} excluded papers to {excluded_file}")


def print_summary(
    ai_science_papers: list[dict[str, Any]], excluded_papers: list[dict[str, Any]]
) -> None:
    """Print abstract filter summary.

    Args:
        ai_science_papers: Papers kept.
        excluded_papers: Papers excluded.
    """
    total = len(ai_science_papers) + len(excluded_papers)
    kept = len(ai_science_papers)
    excluded = len(excluded_papers)

    logger.info("=" * 50)
    logger.info("Abstract Filter Summary:")
    logger.info(f"  Total papers checked: {total}")
    logger.info(f"  AI+Science papers: {kept} ({100 * kept / total:.1f}%)")
    logger.info(f"  Excluded (pure ML): {excluded} ({100 * excluded / total:.1f}%)")

    # Domain distribution
    if ai_science_papers:
        domain_counts: dict[str, int] = {}
        for paper in ai_science_papers:
            d = paper.get("matched_domain", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1
        logger.info("  By domain:")
        for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
            logger.info(f"    {domain}: {count}")


def main() -> None:
    """Main entry point for abstract filtering."""
    # Load config from scicode_lint
    llm_config = load_llm_config()

    parser = argparse.ArgumentParser(
        description="Filter papers by abstract using LLM to identify AI+science"
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=DATA_DIR / "filtered_papers.json",
        help="Input file with filtered papers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_ABSTRACT_FILTER_CONCURRENT,
        help=f"Maximum concurrent LLM requests (default: {DEFAULT_ABSTRACT_FILTER_CONCURRENT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-filtering even if output exists",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="Sample N papers balanced by domain (0 = no sampling, keep all)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for sampling (default: 0.7)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible sampling",
    )
    parser.add_argument(
        "--exclude-from-prefilter-run",
        type=int,
        nargs="+",
        metavar="RUN_ID",
        help="Exclude papers already used in these prefilter runs (prevents data contamination)",
    )
    parser.add_argument(
        "--exclude-from-paper-set",
        type=Path,
        nargs="+",
        metavar="JSON_FILE",
        help="Exclude papers listed in these paper set JSON files",
    )
    parser.add_argument(
        "--no-auto-exclude",
        action="store_true",
        help="Disable automatic exclusion of all paper sets in real_world_demo/paper_sets/",
    )
    args = parser.parse_args()

    # Check if output already exists
    output_file = args.output_dir / "ai_science_papers.json"
    if output_file.exists() and not args.force:
        logger.info(f"Output already exists: {output_file}")
        logger.info("Use --force to re-filter")
        with open(output_file) as f:
            existing = json.load(f)
        logger.info(f"Existing AI+science papers: {len(existing)}")
        return

    # Load filtered papers
    papers = load_filtered_papers(args.input_file)
    logger.info(f"Loaded {len(papers)} filtered papers")

    if not papers:
        logger.warning("No papers to filter!")
        return

    # Collect paper URLs to exclude from all sources
    excluded_urls: set[str] = set()

    # Auto-exclude all paper sets in real_world_demo/paper_sets/ (unless --no-auto-exclude)
    if not args.no_auto_exclude:
        paper_sets_dir = Path(__file__).resolve().parent.parent.parent / "paper_sets"
        if paper_sets_dir.exists():
            for json_file in sorted(paper_sets_dir.glob("*.json")):
                with open(json_file) as f:
                    paper_set = json.load(f)
                urls = {p.get("paper_url") for p in paper_set if p.get("paper_url")}
                excluded_urls.update(urls)
                logger.info(f"Auto-excluding {len(urls)} papers from {json_file.name}")

    # Exclude from explicit paper set files
    if args.exclude_from_paper_set:
        for json_file in args.exclude_from_paper_set:
            if not json_file.exists():
                logger.warning(f"Paper set file not found: {json_file}")
                continue
            with open(json_file) as f:
                paper_set = json.load(f)
            urls = {p.get("paper_url") for p in paper_set if p.get("paper_url")}
            excluded_urls.update(urls)
            logger.info(f"Excluding {len(urls)} papers from {json_file.name}")

    # Exclude from prefilter runs (DB lookup)
    if args.exclude_from_prefilter_run:
        from real_world_demo.database import get_paper_urls_from_prefilter_runs, init_db

        conn = init_db()
        run_urls = get_paper_urls_from_prefilter_runs(conn, args.exclude_from_prefilter_run)
        conn.close()
        excluded_urls.update(run_urls)
        logger.info(
            f"Excluding {len(run_urls)} papers from prefilter runs "
            f"{args.exclude_from_prefilter_run}"
        )

    # Apply exclusion
    if excluded_urls:
        before = len(papers)
        papers = [p for p in papers if p.get("paper_url") not in excluded_urls]
        logger.info(
            f"Excluded {before - len(papers)} papers total "
            f"({len(excluded_urls)} unique URLs), {len(papers)} remaining"
        )

    # Create LLM client using scicode_lint infrastructure
    llm_client = create_client(llm_config)
    logger.info(f"Using LLM: {llm_config.base_url} (model: {llm_config.model_served_name})")

    # Run abstract filter
    ai_science_papers, excluded_papers = asyncio.run(
        filter_papers_by_abstract(
            papers,
            llm_client,
            max_concurrent=args.max_concurrent,
        )
    )

    # Save all filtered results
    save_results(ai_science_papers, excluded_papers, args.output_dir)

    # Handle seed warnings
    if args.seed is not None and args.sample == 0:
        logger.warning("--seed has no effect without --sample")
    elif args.sample > 0 and args.seed is None:
        logger.warning("--sample without --seed: results will not be reproducible")

    # Sample if requested
    final_papers = ai_science_papers
    if args.sample > 0:
        # Set random seed for reproducibility
        if args.seed is not None:
            import random

            random.seed(args.seed)
            logger.info(f"Random seed set to {args.seed}")
        logger.info(f"Sampling {args.sample} papers balanced by domain...")
        final_papers = sample_by_domain(
            ai_science_papers,
            target_count=args.sample,
            min_confidence=args.min_confidence,
        )
        # Save sampled papers for cloning
        sampled_file = args.output_dir / "sampled_papers.json"
        with open(sampled_file, "w") as f:
            json.dump(final_papers, f, indent=2)
        logger.info(f"Saved {len(final_papers)} sampled papers to {sampled_file}")

    # Print summary
    print_summary(final_papers, excluded_papers)


if __name__ == "__main__":
    main()
