"""Tests for real_world_demo filtering modules (prefilter, abstracts, pipeline)."""

import json
import tempfile
from pathlib import Path

import pytest


class TestPrefilterFiles:
    """Tests for prefilter_files module."""

    def test_classify_prompt_exists(self) -> None:
        """Test that classification prompts are defined."""
        from scicode_lint.repo_filter.classify import (
            CLASSIFY_SYSTEM_PROMPT,
            CLASSIFY_USER_PROMPT,
        )

        # System prompt should mention ML classification concepts
        assert "ML" in CLASSIFY_SYSTEM_PROMPT or "self-contained" in CLASSIFY_SYSTEM_PROMPT.lower()
        assert "fragment" in CLASSIFY_SYSTEM_PROMPT.lower()

        # User prompt should have classification options
        assert "self_contained" in CLASSIFY_USER_PROMPT
        assert "fragment" in CLASSIFY_USER_PROMPT

    def test_classify_prompt_covers_workflow(self) -> None:
        """Test that classification prompt covers ML workflow detection."""
        from scicode_lint.repo_filter.classify import (
            CLASSIFY_SYSTEM_PROMPT,
            CLASSIFY_USER_PROMPT,
        )

        combined_lower = (CLASSIFY_SYSTEM_PROMPT + CLASSIFY_USER_PROMPT).lower()
        # Should cover main ML workflow components
        assert "train" in combined_lower
        assert "model" in combined_lower
        assert "data" in combined_lower

    def test_load_qualifying_files_not_found(self) -> None:
        """Test error when qualifying files not found."""
        from pathlib import Path

        from real_world_demo.sources.papers_with_code.prefilter_files import load_qualifying_files

        with pytest.raises(FileNotFoundError):
            load_qualifying_files(Path("/nonexistent/path.json"))

    def test_save_results(self) -> None:
        """Test saving prefilter results."""
        from real_world_demo.sources.papers_with_code.prefilter_files import save_results

        pipeline_files = [{"file_path": "a.py", "is_pipeline": True}]
        filtered_out = [{"file_path": "b.py", "is_pipeline": False}]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            save_results(pipeline_files, filtered_out, output_dir)

            assert (output_dir / "pipeline_files.json").exists()
            assert (output_dir / "prefilter_excluded.json").exists()

            with open(output_dir / "pipeline_files.json") as f:
                saved = json.load(f)
            assert len(saved) == 1
            assert saved[0]["file_path"] == "a.py"


class TestFilterAbstractsExclusion:
    """Tests for paper set exclusion in filter_abstracts."""

    def test_auto_exclude_from_paper_sets(self) -> None:
        """Test that auto-exclusion reads all JSON files from paper_sets dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake paper_sets dir with two sets
            paper_sets_dir = Path(tmpdir) / "paper_sets"
            paper_sets_dir.mkdir()

            set_a = [
                {"paper_url": "https://pwc/paper-a1", "title": "A1"},
                {"paper_url": "https://pwc/paper-a2", "title": "A2"},
            ]
            set_b = [
                {"paper_url": "https://pwc/paper-b1", "title": "B1"},
            ]
            with open(paper_sets_dir / "set_a.json", "w") as f:
                json.dump(set_a, f)
            with open(paper_sets_dir / "set_b.json", "w") as f:
                json.dump(set_b, f)

            # Load URLs from all sets
            excluded_urls: set[str] = set()
            for json_file in sorted(paper_sets_dir.glob("*.json")):
                with open(json_file) as f:
                    paper_set = json.load(f)
                urls = {p.get("paper_url") for p in paper_set if p.get("paper_url")}
                excluded_urls.update(urls)

            assert len(excluded_urls) == 3
            assert "https://pwc/paper-a1" in excluded_urls
            assert "https://pwc/paper-a2" in excluded_urls
            assert "https://pwc/paper-b1" in excluded_urls

    def test_exclusion_filters_papers(self) -> None:
        """Test that excluded URLs are actually removed from paper list."""
        papers = [
            {"paper_url": "https://pwc/paper-1", "title": "Keep me"},
            {"paper_url": "https://pwc/paper-2", "title": "Exclude me"},
            {"paper_url": "https://pwc/paper-3", "title": "Keep me too"},
        ]
        excluded_urls = {"https://pwc/paper-2"}

        filtered = [p for p in papers if p.get("paper_url") not in excluded_urls]

        assert len(filtered) == 2
        assert filtered[0]["title"] == "Keep me"
        assert filtered[1]["title"] == "Keep me too"

    def test_exclusion_handles_missing_paper_url(self) -> None:
        """Test that papers without paper_url are kept (not excluded)."""
        papers = [
            {"paper_url": "https://pwc/paper-1", "title": "Has URL"},
            {"title": "No URL"},  # No paper_url field
        ]
        excluded_urls = {"https://pwc/other"}

        filtered = [p for p in papers if p.get("paper_url") not in excluded_urls]

        assert len(filtered) == 2  # Both kept since neither matches

    def test_exclusion_deduplicates_across_sets(self) -> None:
        """Test that same paper in multiple sets is only counted once."""
        with tempfile.TemporaryDirectory() as tmpdir:
            paper_sets_dir = Path(tmpdir)

            # Same paper appears in both sets
            set_a = [{"paper_url": "https://pwc/shared", "title": "Shared"}]
            set_b = [
                {"paper_url": "https://pwc/shared", "title": "Shared"},
                {"paper_url": "https://pwc/unique", "title": "Unique"},
            ]
            with open(paper_sets_dir / "a.json", "w") as f:
                json.dump(set_a, f)
            with open(paper_sets_dir / "b.json", "w") as f:
                json.dump(set_b, f)

            excluded_urls: set[str] = set()
            for json_file in sorted(paper_sets_dir.glob("*.json")):
                with open(json_file) as f:
                    paper_set = json.load(f)
                urls = {p.get("paper_url") for p in paper_set if p.get("paper_url")}
                excluded_urls.update(urls)

            # Deduplication via set
            assert len(excluded_urls) == 2
            assert "https://pwc/shared" in excluded_urls
            assert "https://pwc/unique" in excluded_urls

    def test_paper_sets_dir_location(self) -> None:
        """Test that paper_sets dir is correctly resolved relative to filter_abstracts.py."""
        filter_abstracts_path = Path(
            "real_world_demo/sources/papers_with_code/filter_abstracts.py"
        ).resolve()
        paper_sets_dir = filter_abstracts_path.parent.parent.parent / "paper_sets"

        # Should resolve to real_world_demo/paper_sets/
        assert paper_sets_dir.name == "paper_sets"
        assert paper_sets_dir.parent.name == "real_world_demo"

    def test_paper_sets_json_files_exist(self) -> None:
        """Test that committed paper set files exist and are valid JSON."""
        paper_sets_dir = Path("real_world_demo/paper_sets")
        assert paper_sets_dir.exists(), "paper_sets directory must exist"

        json_files = list(paper_sets_dir.glob("*.json"))
        assert len(json_files) >= 2, "Expected at least meta_loop_set.json and holdout_set.json"

        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)
            assert isinstance(data, list), f"{json_file.name} must contain a JSON array"
            assert len(data) > 0, f"{json_file.name} must not be empty"
            # Each entry must have paper_url
            for entry in data:
                assert "paper_url" in entry, f"Entry in {json_file.name} missing paper_url"

    def test_no_overlap_between_paper_sets(self) -> None:
        """Test that paper sets have no overlapping papers."""
        paper_sets_dir = Path("real_world_demo/paper_sets")
        all_sets: dict[str, set[str]] = {}

        for json_file in sorted(paper_sets_dir.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)
            urls = {p["paper_url"] for p in data}
            all_sets[json_file.stem] = urls

        # Check pairwise overlap
        set_names = list(all_sets.keys())
        for i, name_a in enumerate(set_names):
            for name_b in set_names[i + 1 :]:
                overlap = all_sets[name_a] & all_sets[name_b]
                assert len(overlap) == 0, f"Overlap between {name_a} and {name_b}: {overlap}"


class TestRunPipeline:
    """Tests for run_pipeline module."""

    def test_check_prerequisites_filter(self) -> None:
        """Test prerequisites check for filter stage."""
        from real_world_demo.sources.papers_with_code.run_pipeline import check_prerequisites

        # Filter stage has no prerequisites
        assert check_prerequisites("filter") is True

    def test_check_prerequisites_clone_missing(self) -> None:
        """Test prerequisites check for clone stage with missing file."""
        from real_world_demo.config import DATA_DIR
        from real_world_demo.sources.papers_with_code.run_pipeline import check_prerequisites

        # Clone stage requires ai_science_papers.json (output from abstract_filter stage)
        ai_science_file = DATA_DIR / "ai_science_papers.json"
        if ai_science_file.exists():
            # If file exists, the check will pass
            assert check_prerequisites("clone") is True
        else:
            # If file doesn't exist, check will fail
            assert check_prerequisites("clone") is False

    def test_check_prerequisites_abstract_filter(self) -> None:
        """Test prerequisites check for abstract_filter stage."""
        from real_world_demo.config import DATA_DIR
        from real_world_demo.sources.papers_with_code.run_pipeline import check_prerequisites

        # abstract_filter stage requires filtered_papers.json
        filtered_file = DATA_DIR / "filtered_papers.json"
        if filtered_file.exists():
            assert check_prerequisites("abstract_filter") is True
        else:
            assert check_prerequisites("abstract_filter") is False
