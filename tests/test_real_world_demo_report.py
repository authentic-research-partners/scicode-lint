"""Tests for real_world_demo report generation and models."""

import tempfile
from pathlib import Path


class TestGenerateReport:
    """Tests for generate_report module."""

    def test_generate_markdown_report_empty_db(self) -> None:
        """Test report generation with empty database."""
        from real_world_demo.database import init_db
        from real_world_demo.generate_report import generate_markdown_report

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            report = generate_markdown_report(conn)

            assert "No Analysis Data" in report
            conn.close()

    def test_generate_markdown_report_with_data(self) -> None:
        """Test report generation with actual data."""
        from real_world_demo.database import (
            get_or_create_repo,
            init_db,
            insert_file,
            insert_file_analysis,
            insert_findings,
            start_analysis_run,
        )
        from real_world_demo.generate_report import generate_markdown_report

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            # Set up data
            repo_id = get_or_create_repo(
                conn,
                {
                    "repo_url": "https://github.com/test/repo",
                    "repo_name": "test__repo",
                    "domain": "biology",
                },
            )
            file_id = insert_file(
                conn,
                repo_id,
                {
                    "file_path": "files/test__repo/model.py",
                    "original_path": "src/model.py",
                },
            )
            run_id = start_analysis_run(conn, total_files=1)
            analysis_id = insert_file_analysis(conn, run_id, file_id, "success")
            insert_findings(
                conn,
                analysis_id,
                [
                    {
                        "id": "dl-001",
                        "category": "data-leakage",
                        "severity": "high",
                        "confidence": 0.85,
                        "issue": "Test issue",
                        "explanation": "Test explanation",
                        "location": {"lines": [10], "snippet": "test code"},
                    }
                ],
            )

            report = generate_markdown_report(conn, run_id)

            assert "Real-World Scientific ML Code Analysis Report" in report
            assert "data-leakage" in report
            assert "dl-001" in report
            assert "biology" in report
            conn.close()

    def test_get_example_findings(self) -> None:
        """Test getting example findings grouped by category."""
        from real_world_demo.database import (
            get_or_create_repo,
            init_db,
            insert_file,
            insert_file_analysis,
            insert_findings,
            start_analysis_run,
        )
        from real_world_demo.generate_report import get_example_findings

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            # Set up data
            repo_id = get_or_create_repo(
                conn,
                {
                    "repo_url": "https://github.com/test/repo",
                    "repo_name": "test__repo",
                    "domain": "biology",
                },
            )
            file_id = insert_file(conn, repo_id, {"file_path": "model.py"})
            run_id = start_analysis_run(conn, total_files=1)
            analysis_id = insert_file_analysis(conn, run_id, file_id, "success")
            insert_findings(
                conn,
                analysis_id,
                [
                    {
                        "id": "dl-001",
                        "category": "data-leakage",
                        "severity": "high",
                        "confidence": 0.9,
                        "issue": "Issue 1",
                        "location": {},
                    },
                    {
                        "id": "rep-001",
                        "category": "reproducibility",
                        "severity": "medium",
                        "confidence": 0.8,
                        "issue": "Issue 2",
                        "location": {},
                    },
                ],
            )

            examples = get_example_findings(conn, run_id)

            assert "data-leakage" in examples
            assert "reproducibility" in examples
            assert len(examples["data-leakage"]) == 1
            assert examples["data-leakage"][0]["pattern_id"] == "dl-001"
            conn.close()


class TestModels:
    """Tests for Pydantic models."""

    def test_finding_model(self) -> None:
        """Test Finding model validation."""
        from real_world_demo.models import Finding, Severity

        finding = Finding(
            pattern_id="dl-001",
            category="data-leakage",
            severity=Severity.HIGH,
            confidence=0.85,
            issue="Data leakage detected",
        )

        assert finding.pattern_id == "dl-001"
        assert finding.severity == Severity.HIGH
        assert finding.confidence == 0.85

    def test_analysis_run_model(self) -> None:
        """Test AnalysisRun model."""
        from real_world_demo.models import AnalysisRun, RunStatus

        run = AnalysisRun(
            run_id=1,
            total_files=100,
            analyzed_files=95,
            files_with_findings=20,
            total_findings=35,
        )

        assert run.status == RunStatus.RUNNING
        assert run.total_files == 100

    def test_domain_stats_model(self) -> None:
        """Test DomainStats model."""
        from real_world_demo.models import DomainStats

        stats = DomainStats(
            domain="biology",
            total_files=50,
            analyzed_files=48,
            files_with_findings=10,
            total_findings=15,
            finding_rate=20.8,
        )

        assert stats.domain == "biology"
