"""Tests for real_world_demo database module."""

import tempfile
from pathlib import Path


class TestDatabase:
    """Tests for database module."""

    def test_init_db_creates_tables(self) -> None:
        """Test database initialization creates all tables."""
        from real_world_demo.database import init_db

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            # Check tables exist
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = {row[0] for row in cursor.fetchall()}

            assert "papers" in tables
            assert "repos" in tables
            assert "files" in tables
            assert "analysis_runs" in tables
            assert "file_analyses" in tables
            assert "findings" in tables

            conn.close()

    def test_insert_paper(self) -> None:
        """Test inserting a paper record."""
        from real_world_demo.database import init_db, insert_paper

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            paper_data = {
                "paper_url": "https://paperswithcode.com/paper/test",
                "paper_title": "Test Paper",
                "arxiv_id": "2301.00001",
                "tasks": ["Protein Folding"],
                "domain": "biology",
            }
            paper_id = insert_paper(conn, paper_data)

            assert paper_id > 0

            # Inserting same paper should return existing ID
            paper_id2 = insert_paper(conn, paper_data)
            assert paper_id2 == paper_id

            conn.close()

    def test_get_or_create_repo(self) -> None:
        """Test getting or creating a repo record."""
        from real_world_demo.database import get_or_create_repo, init_db

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            repo_data = {
                "repo_url": "https://github.com/owner/repo",
                "repo_name": "owner__repo",
                "domain": "biology",
                "paper_url": "https://paperswithcode.com/paper/test",
            }
            repo_id = get_or_create_repo(conn, repo_data)

            assert repo_id > 0

            # Getting same repo should return existing ID
            repo_id2 = get_or_create_repo(conn, repo_data)
            assert repo_id2 == repo_id

            conn.close()

    def test_insert_file(self) -> None:
        """Test inserting a file record."""
        from real_world_demo.database import get_or_create_repo, init_db, insert_file

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            # Create repo first
            repo_id = get_or_create_repo(
                conn,
                {
                    "repo_url": "https://github.com/owner/repo",
                    "repo_name": "owner__repo",
                    "domain": "biology",
                },
            )

            file_data = {
                "file_path": "files/owner__repo/model.py",
                "original_path": "src/model.py",
                "ml_imports": ["torch", "sklearn"],
                "scientific_imports": ["numpy"],
                "file_size": 1000,
                "line_count": 50,
            }
            file_id = insert_file(conn, repo_id, file_data)

            assert file_id > 0

            conn.close()

    def test_analysis_run_lifecycle(self) -> None:
        """Test starting and completing an analysis run."""
        from real_world_demo.database import (
            complete_analysis_run,
            init_db,
            list_runs,
            start_analysis_run,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            # Start run
            run_id = start_analysis_run(
                conn,
                total_files=10,
                config={"max_files": 10},
                model_name="test-model",
                notes="Test run",
            )

            assert run_id > 0

            # Complete run
            complete_analysis_run(
                conn,
                run_id=run_id,
                analyzed=8,
                with_findings=3,
                total_findings=5,
                status="completed",
            )

            # Verify via list_runs
            runs = list_runs(conn, limit=1)
            assert len(runs) == 1
            assert runs[0].run_id == run_id
            assert runs[0].total_files == 10
            assert runs[0].analyzed_files == 8
            assert runs[0].files_with_findings == 3
            assert runs[0].total_findings == 5

            conn.close()

    def test_insert_findings(self) -> None:
        """Test inserting findings for a file analysis."""
        from real_world_demo.database import (
            get_or_create_repo,
            init_db,
            insert_file,
            insert_file_analysis,
            insert_findings,
            start_analysis_run,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            # Set up hierarchy
            repo_id = get_or_create_repo(
                conn,
                {"repo_url": "https://github.com/o/r", "repo_name": "o__r", "domain": "biology"},
            )
            file_id = insert_file(conn, repo_id, {"file_path": "files/o__r/model.py"})
            run_id = start_analysis_run(conn, total_files=1)
            analysis_id = insert_file_analysis(conn, run_id, file_id, "success", duration=1.5)

            # Insert findings
            findings = [
                {
                    "id": "dl-001",
                    "category": "data-leakage",
                    "severity": "high",
                    "confidence": 0.85,
                    "issue": "Test issue",
                    "location": {"lines": [10, 11], "snippet": "code"},
                },
                {
                    "id": "rep-002",
                    "category": "reproducibility",
                    "severity": "medium",
                    "confidence": 0.7,
                    "issue": "Another issue",
                    "location": {},
                },
            ]
            count = insert_findings(conn, analysis_id, findings)

            assert count == 2

            # Verify findings in DB
            cursor = conn.execute(
                "SELECT COUNT(*) FROM findings WHERE file_analysis_id = ?",
                (analysis_id,),
            )
            assert cursor.fetchone()[0] == 2

            conn.close()

    def test_get_run_stats(self) -> None:
        """Test getting statistics for an analysis run."""
        from real_world_demo.database import (
            get_or_create_repo,
            get_run_stats,
            init_db,
            insert_file,
            insert_file_analysis,
            insert_findings,
            start_analysis_run,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            conn = init_db(db_path)

            # Create data
            repo_id = get_or_create_repo(
                conn,
                {"repo_url": "https://github.com/o/r", "repo_name": "o__r", "domain": "biology"},
            )
            file_id = insert_file(conn, repo_id, {"file_path": "files/o__r/model.py"})
            run_id = start_analysis_run(conn, total_files=1)
            analysis_id = insert_file_analysis(conn, run_id, file_id, "success")
            findings = [
                {
                    "id": "dl-001",
                    "category": "data-leakage",
                    "severity": "high",
                    "confidence": 0.9,
                    "issue": "Test",
                }
            ]
            insert_findings(conn, analysis_id, findings)

            # Get stats
            stats = get_run_stats(conn, run_id)

            assert stats.run_id == run_id
            assert stats.total_repos == 1
            assert len(stats.by_category) >= 1

            conn.close()
