"""Tests for real_world_demo pipeline modules (config, filter, clone, files, manifest, analysis)."""

import json
import tempfile
from pathlib import Path


class TestConfig:
    """Tests for config module."""

    def test_scientific_domains_defined(self) -> None:
        """Verify scientific domains are defined."""
        from real_world_demo.config import SCIENTIFIC_DOMAINS

        assert "biology" in SCIENTIFIC_DOMAINS
        assert "chemistry" in SCIENTIFIC_DOMAINS
        assert "medical" in SCIENTIFIC_DOMAINS
        assert "physics" in SCIENTIFIC_DOMAINS

    def test_ml_imports_defined(self) -> None:
        """Verify ML imports list is defined."""
        from real_world_demo.config import ML_IMPORTS

        assert "sklearn" in ML_IMPORTS
        assert "torch" in ML_IMPORTS
        assert "tensorflow" in ML_IMPORTS

    def test_directories_created(self) -> None:
        """Verify data directories exist."""
        from real_world_demo.config import COLLECTED_DIR, DATA_DIR

        assert DATA_DIR.exists()
        assert COLLECTED_DIR.exists()


class TestFilterPapers:
    """Tests for filter_papers module."""

    def test_matches_domain_biology(self) -> None:
        """Test domain matching for biology tasks."""
        from real_world_demo.sources.papers_with_code.filter_papers import matches_domain

        tasks = ["Protein Structure Prediction", "Gene Expression"]
        assert matches_domain(tasks) == "biology"

    def test_matches_domain_medical(self) -> None:
        """Test domain matching for medical tasks."""
        from real_world_demo.sources.papers_with_code.filter_papers import matches_domain

        tasks = ["Medical Image Segmentation", "Disease Classification"]
        assert matches_domain(tasks) == "medical"

    def test_matches_domain_none(self) -> None:
        """Test no match for non-scientific tasks."""
        from real_world_demo.sources.papers_with_code.filter_papers import matches_domain

        tasks = ["Image Classification", "Object Detection"]
        assert matches_domain(tasks) is None

    def test_matches_domain_empty(self) -> None:
        """Test empty tasks list."""
        from real_world_demo.sources.papers_with_code.filter_papers import matches_domain

        assert matches_domain([]) is None
        assert matches_domain(None) is None

    def test_matches_domain_specific_domains(self) -> None:
        """Test filtering to specific domains."""
        from real_world_demo.sources.papers_with_code.filter_papers import matches_domain

        tasks = ["Protein Folding", "Drug Discovery"]
        # Should match biology when filtering for biology only
        assert matches_domain(tasks, domains=["biology"]) == "biology"
        # Should not match when filtering for physics only
        assert matches_domain(tasks, domains=["physics"]) is None

    def test_should_exclude_benchmark(self) -> None:
        """Test exclusion of benchmark tasks."""
        from real_world_demo.sources.papers_with_code.filter_papers import should_exclude

        tasks = ["Image Classification Benchmark", "Model Evaluation"]
        assert should_exclude(tasks) is True

    def test_should_exclude_false(self) -> None:
        """Test non-exclusion of scientific tasks."""
        from real_world_demo.sources.papers_with_code.filter_papers import should_exclude

        tasks = ["Protein Structure Prediction"]
        assert should_exclude(tasks) is False

    def test_normalize_github_url(self) -> None:
        """Test GitHub URL normalization."""
        from real_world_demo.sources.papers_with_code.filter_papers import normalize_github_url

        # Remove trailing slash
        assert (
            normalize_github_url("https://github.com/owner/repo/")
            == "https://github.com/owner/repo"
        )

        # Remove .git suffix
        assert (
            normalize_github_url("https://github.com/owner/repo.git")
            == "https://github.com/owner/repo"
        )

        # Convert git:// to https://
        assert (
            normalize_github_url("git://github.com/owner/repo") == "https://github.com/owner/repo"
        )

        # Convert SSH to HTTPS
        assert normalize_github_url("git@github.com:owner/repo") == "https://github.com/owner/repo"

    def test_filter_papers_basic(self) -> None:
        """Test basic paper filtering."""
        from real_world_demo.sources.papers_with_code.filter_papers import filter_papers

        papers = [
            {
                "paper_url": "https://paperswithcode.com/paper/1",
                "tasks": ["Protein Folding"],
                "title": "Paper 1",
            },
            {
                "paper_url": "https://paperswithcode.com/paper/2",
                "tasks": ["Image Classification"],
                "title": "Paper 2",
            },
        ]
        links = [
            {
                "paper_url": "https://paperswithcode.com/paper/1",
                "repo_url": "https://github.com/owner/repo1",
            },
        ]

        filtered = filter_papers(papers, links)

        assert len(filtered) == 1
        assert filtered[0]["title"] == "Paper 1"
        # repo_urls are now embedded in paper records
        assert "repo_urls" in filtered[0]
        assert "https://github.com/owner/repo1" in filtered[0]["repo_urls"]

    def test_filter_papers_balanced_sampling(self) -> None:
        """Test balanced sampling across domains."""
        from real_world_demo.sources.papers_with_code.filter_papers import filter_papers

        # Create papers across multiple domains
        papers = []
        links = []
        # 10 biology papers
        for i in range(10):
            papers.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/bio{i}",
                    "tasks": ["Protein Folding"],
                    "title": f"Bio Paper {i}",
                }
            )
            links.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/bio{i}",
                    "repo_url": f"https://github.com/owner/bio{i}",
                }
            )
        # 10 medical papers
        for i in range(10):
            papers.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/med{i}",
                    "tasks": ["Medical Imaging"],
                    "title": f"Medical Paper {i}",
                }
            )
            links.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/med{i}",
                    "repo_url": f"https://github.com/owner/med{i}",
                }
            )

        # Request 10 papers with balanced sampling
        filtered = filter_papers(papers, links, limit=10, balanced=True)

        # Count domains
        domain_counts: dict[str, int] = {}
        for paper in filtered:
            d = paper.get("matched_domain", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1

        # Should have roughly equal distribution (5 each)
        assert len(filtered) == 10
        assert domain_counts.get("biology", 0) == 5
        assert domain_counts.get("medical", 0) == 5

    def test_filter_papers_unbalanced(self) -> None:
        """Test unbalanced sampling (first-come-first-served)."""
        from real_world_demo.sources.papers_with_code.filter_papers import filter_papers

        # Create papers - biology first, then medical
        papers = []
        links = []
        for i in range(10):
            papers.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/bio{i}",
                    "tasks": ["Protein Folding"],
                    "title": f"Bio Paper {i}",
                }
            )
            links.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/bio{i}",
                    "repo_url": f"https://github.com/owner/bio{i}",
                }
            )
        for i in range(10):
            papers.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/med{i}",
                    "tasks": ["Medical Imaging"],
                    "title": f"Medical Paper {i}",
                }
            )
            links.append(
                {
                    "paper_url": f"https://paperswithcode.com/paper/med{i}",
                    "repo_url": f"https://github.com/owner/med{i}",
                }
            )

        # Request 10 papers WITHOUT balanced sampling
        filtered = filter_papers(papers, links, limit=10, balanced=False)

        assert len(filtered) == 10


class TestCloneRepos:
    """Tests for clone_repos module."""

    def test_repo_url_to_path(self) -> None:
        """Test converting repo URL to local path."""
        from real_world_demo.sources.papers_with_code.clone_repos import repo_url_to_path

        base_dir = Path("/tmp/repos")
        url = "https://github.com/owner/repo"
        path = repo_url_to_path(url, base_dir)

        assert path == base_dir / "owner__repo"

    def test_clone_result_to_dict(self) -> None:
        """Test CloneResult serialization."""
        from real_world_demo.sources.papers_with_code.clone_repos import CloneResult

        result = CloneResult(
            repo_url="https://github.com/owner/repo",
            success=True,
            repo_path=Path("/tmp/repos/owner__repo"),
        )
        d = result.to_dict()

        assert d["repo_url"] == "https://github.com/owner/repo"
        assert d["success"] is True
        assert d["repo_path"] == "/tmp/repos/owner__repo"
        assert d["error"] is None

    def test_clone_result_to_dict_with_error(self) -> None:
        """Test CloneResult serialization with error."""
        from real_world_demo.sources.papers_with_code.clone_repos import CloneResult

        result = CloneResult(
            repo_url="https://github.com/owner/repo",
            success=False,
            error="not_found",
        )
        d = result.to_dict()

        assert d["success"] is False
        assert d["error"] == "not_found"
        assert d["repo_path"] is None


class TestFilterFiles:
    """Tests for filter_files module."""

    def test_extract_imports(self) -> None:
        """Test import extraction from Python code."""
        from real_world_demo.sources.papers_with_code.filter_files import extract_imports

        code = """
import numpy as np
import pandas
from sklearn.model_selection import train_test_split
from torch import nn
"""
        imports = extract_imports(code)

        assert "numpy" in imports
        assert "pandas" in imports
        assert "sklearn" in imports
        assert "torch" in imports

    def test_has_ml_imports_true(self) -> None:
        """Test ML import detection - positive case."""
        from real_world_demo.sources.papers_with_code.filter_files import has_ml_imports

        imports = {"numpy", "sklearn", "pandas"}
        has_ml, matched = has_ml_imports(imports)

        assert has_ml is True
        assert "sklearn" in matched

    def test_has_ml_imports_false(self) -> None:
        """Test ML import detection - negative case."""
        from real_world_demo.sources.papers_with_code.filter_files import has_ml_imports

        imports = {"os", "sys", "json"}
        has_ml, matched = has_ml_imports(imports)

        assert has_ml is False
        assert len(matched) == 0

    def test_has_scientific_imports(self) -> None:
        """Test scientific import detection."""
        from real_world_demo.sources.papers_with_code.filter_files import has_scientific_imports

        imports = {"numpy", "scipy", "matplotlib"}
        has_sci, matched = has_scientific_imports(imports)

        assert has_sci is True
        assert "numpy" in matched
        assert "scipy" in matched

    def test_matches_exclude_pattern_setup(self) -> None:
        """Test file exclusion for setup.py."""
        from real_world_demo.sources.papers_with_code.filter_files import matches_exclude_pattern

        repo_root = Path("/tmp/repo")
        file_path = repo_root / "setup.py"

        assert matches_exclude_pattern(file_path, repo_root) is True

    def test_matches_exclude_pattern_test_file(self) -> None:
        """Test file exclusion for test files."""
        from real_world_demo.sources.papers_with_code.filter_files import matches_exclude_pattern

        repo_root = Path("/tmp/repo")
        file_path = repo_root / "test_model.py"

        assert matches_exclude_pattern(file_path, repo_root) is True

    def test_matches_exclude_pattern_normal_file(self) -> None:
        """Test non-exclusion for normal files."""
        from real_world_demo.sources.papers_with_code.filter_files import matches_exclude_pattern

        repo_root = Path("/tmp/repo")
        file_path = repo_root / "model.py"

        assert matches_exclude_pattern(file_path, repo_root) is False


class TestGenerateManifest:
    """Tests for generate_manifest module."""

    def test_generate_unique_path(self) -> None:
        """Test unique path generation within repo directories."""
        from real_world_demo.sources.papers_with_code.generate_manifest import generate_unique_path

        seen_per_repo: dict[str, set[str]] = {}
        file_path = Path("/tmp/repo/model.py")
        repo_name = "owner__repo"

        repo_dir, name1 = generate_unique_path(file_path, repo_name, seen_per_repo)
        assert repo_dir == "owner__repo"
        assert name1 == "model.py"
        assert name1 in seen_per_repo[repo_name]

        # Second call with same file should get a unique name
        repo_dir, name2 = generate_unique_path(file_path, repo_name, seen_per_repo)
        assert repo_dir == "owner__repo"
        assert name2 == "model_1.py"
        assert name2 in seen_per_repo[repo_name]

    def test_load_repos_metadata(self) -> None:
        """Test loading repo metadata from papers file with embedded repo_urls."""
        from real_world_demo.sources.papers_with_code.generate_manifest import load_repos_metadata

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Papers format with embedded repo_urls
            papers = [
                {
                    "paper_url": "https://paperswithcode.com/paper1",
                    "title": "Paper 1",
                    "matched_domain": "biology",
                    "repo_urls": ["https://github.com/owner/repo1"],
                },
                {
                    "paper_url": "https://paperswithcode.com/paper2",
                    "title": "Paper 2",
                    "matched_domain": "medical",
                    "repo_urls": ["https://github.com/owner/repo2"],
                },
            ]
            json.dump(papers, f)
            f.flush()
            temp_path = Path(f.name)

        try:
            metadata = load_repos_metadata(temp_path)

            assert "owner__repo1" in metadata
            assert metadata["owner__repo1"]["domain"] == "biology"
            assert "owner__repo2" in metadata
        finally:
            temp_path.unlink(missing_ok=True)


class TestRunAnalysis:
    """Tests for run_analysis module."""

    def test_aggregate_findings_empty(self) -> None:
        """Test aggregation with no findings."""
        from real_world_demo.run_analysis import aggregate_findings

        results = [
            {"file_path": "file1.py", "success": True, "findings": [], "domain": "biology"},
            {"file_path": "file2.py", "success": True, "findings": [], "domain": "medical"},
        ]

        stats = aggregate_findings(results)

        assert stats["total_files"] == 2
        assert stats["analyzed_successfully"] == 2
        assert stats["files_with_findings"] == 0
        assert stats["total_findings"] == 0

    def test_aggregate_findings_with_findings(self) -> None:
        """Test aggregation with findings."""
        from real_world_demo.run_analysis import aggregate_findings

        results = [
            {
                "file_path": "file1.py",
                "success": True,
                "findings": [
                    {"pattern_id": "dl-001", "category": "data-leakage"},
                    {"pattern_id": "dl-002", "category": "data-leakage"},
                ],
                "domain": "biology",
                "repo_name": "repo1",
                "paper_url": "https://paper1",
            },
            {
                "file_path": "file2.py",
                "success": True,
                "findings": [{"pattern_id": "rs-001", "category": "reproducibility"}],
                "domain": "medical",
                "repo_name": "repo2",
                "paper_url": "https://paper2",
            },
        ]

        stats = aggregate_findings(results)

        assert stats["total_files"] == 2
        assert stats["files_with_findings"] == 2
        assert stats["total_findings"] == 3
        assert stats["findings_by_pattern"]["dl-001"] == 1
        assert stats["findings_by_pattern"]["dl-002"] == 1
        assert stats["findings_by_category"]["data-leakage"] == 2
        assert stats["findings_by_category"]["reproducibility"] == 1
        assert stats["repos_with_findings"] == 2
        assert stats["papers_with_findings"] == 2

    def test_aggregate_findings_failed_analysis(self) -> None:
        """Test aggregation handles failed analyses."""
        from real_world_demo.run_analysis import aggregate_findings

        results = [
            {"file_path": "file1.py", "success": False, "error": "timeout", "findings": []},
            {"file_path": "file2.py", "success": True, "findings": [], "domain": "biology"},
        ]

        stats = aggregate_findings(results)

        assert stats["total_files"] == 2
        assert stats["analyzed_successfully"] == 1

    def test_load_manifest(self) -> None:
        """Test loading manifest CSV."""
        import csv

        from real_world_demo.run_analysis import load_manifest

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["file_path", "domain", "repo_name"])
            writer.writeheader()
            writer.writerow(
                {
                    "file_path": "files/test.py",
                    "domain": "biology",
                    "repo_name": "repo1",
                }
            )
            f.flush()
            temp_path = Path(f.name)

        try:
            manifest = load_manifest(temp_path)

            assert len(manifest) == 1
            assert manifest[0]["file_path"] == "files/test.py"
            assert manifest[0]["domain"] == "biology"
        finally:
            temp_path.unlink(missing_ok=True)
