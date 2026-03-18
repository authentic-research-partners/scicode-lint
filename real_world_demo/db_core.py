"""Core database operations: connection, schema, migrations."""

import sqlite3
import subprocess
from pathlib import Path

from loguru import logger

from .config import DATA_DIR

__all__ = [
    "get_db_path",
    "init_db",
    "get_current_git_commit",
]


def get_db_path() -> Path:
    """Get path to SQLite database."""
    return DATA_DIR / "analysis.db"


def _get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def get_current_git_commit() -> str:
    """Get current git commit hash of scicode-lint."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except Exception:
        return ""


def init_db(db_path: Path | None = None) -> sqlite3.Connection:
    """Initialize database with schema.

    Args:
        db_path: Optional path to database file. Uses default if not provided.

    Returns:
        Connection to the database.
    """
    if db_path is None:
        db_path = get_db_path()

    db_path.parent.mkdir(exist_ok=True, parents=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Create tables with comprehensive schema
    conn.executescript("""
        -- Papers from PapersWithCode
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_url TEXT UNIQUE NOT NULL,
            title TEXT,
            arxiv_id TEXT,
            abstract TEXT,
            authors TEXT,  -- JSON array of author names
            tasks TEXT,  -- JSON array
            matched_domain TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Repositories or data sources containing files
        CREATE TABLE IF NOT EXISTS repos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_url TEXT UNIQUE NOT NULL,
            repo_name TEXT NOT NULL,  -- owner__repo format or data source name
            data_source TEXT DEFAULT 'papersWithCode',  -- data source identifier
            paper_id INTEGER,
            domain TEXT,
            clone_status TEXT DEFAULT 'pending',
            clone_error TEXT,
            cloned_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (paper_id) REFERENCES papers(id)
        );

        -- Python files collected from repos
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,  -- relative to collected_code/files/
            original_path TEXT,       -- path within original repo
            is_notebook BOOLEAN DEFAULT FALSE,
            file_size INTEGER,
            line_count INTEGER,
            ml_imports TEXT,          -- comma-separated
            scientific_imports TEXT,  -- comma-separated
            prefilter_passed BOOLEAN DEFAULT TRUE,
            prefilter_response TEXT,
            has_ml_imports BOOLEAN,           -- deterministic ML check result
            self_contained_class TEXT,           -- self_contained, fragment, uncertain
            self_contained_confidence REAL,      -- LLM confidence in classification
            self_contained_reasoning TEXT,       -- LLM reasoning for classification
            scan_id INTEGER,                     -- scan that classified this file
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (repo_id) REFERENCES repos(id),
            UNIQUE(repo_id, file_path)
        );

        -- Analysis runs (can have multiple per day)
        CREATE TABLE IF NOT EXISTS analysis_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            status TEXT DEFAULT 'running',
            data_source TEXT DEFAULT 'papers_with_code',  -- data source identifier
            total_files INTEGER DEFAULT 0,
            analyzed_files INTEGER DEFAULT 0,
            files_with_findings INTEGER DEFAULT 0,
            total_findings INTEGER DEFAULT 0,
            config TEXT,              -- JSON config (max_files, max_concurrent, etc.)
            git_commit TEXT,          -- scicode-lint git commit
            model_name TEXT,          -- LLM model used
            notes TEXT                -- optional run notes
        );

        -- File analysis results (one per file per run)
        CREATE TABLE IF NOT EXISTS file_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            status TEXT NOT NULL,     -- success, error, timeout, skipped
            error TEXT,               -- error message if failed
            duration_seconds REAL,
            analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (run_id) REFERENCES analysis_runs(id),
            FOREIGN KEY (file_id) REFERENCES files(id),
            UNIQUE(run_id, file_id)
        );

        -- Individual findings
        CREATE TABLE IF NOT EXISTS findings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_analysis_id INTEGER NOT NULL,
            pattern_id TEXT NOT NULL,
            category TEXT,
            severity TEXT,
            confidence REAL,
            issue TEXT,
            explanation TEXT,
            suggestion TEXT,
            reasoning TEXT,
            location_name TEXT,       -- Function/class/method name
            location_type TEXT,       -- function, method, class, module
            lines TEXT,               -- JSON array of line numbers
            focus_line INTEGER,       -- Specific line with the issue
            snippet TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_analysis_id) REFERENCES file_analyses(id)
        );

        -- Finding verifications (Claude evaluation of whether findings are real)
        CREATE TABLE IF NOT EXISTS finding_verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            finding_id INTEGER NOT NULL,
            status TEXT NOT NULL,     -- valid, invalid, uncertain, error
            reasoning TEXT,
            model TEXT,               -- Claude model used
            verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (finding_id) REFERENCES findings(id),
            UNIQUE(finding_id)        -- One verification per finding
        );

        -- Individual pattern run results (tracks success, timeout, etc.)
        CREATE TABLE IF NOT EXISTS pattern_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_analysis_id INTEGER NOT NULL,
            pattern_id TEXT NOT NULL,
            status TEXT NOT NULL,     -- success, timeout, context_length, api_error
            detected TEXT,            -- yes, no, context-dependent (null if failed)
            confidence REAL,          -- null if failed
            reasoning TEXT,
            error_message TEXT,       -- error details if failed
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (file_analysis_id) REFERENCES file_analyses(id),
            UNIQUE(file_analysis_id, pattern_id)
        );

        -- Repository scans (for finding self-contained ML files)
        CREATE TABLE IF NOT EXISTS repo_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repo_id INTEGER NOT NULL,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            status TEXT DEFAULT 'running',
            total_files INTEGER DEFAULT 0,
            passed_ml_import_filter INTEGER DEFAULT 0,
            self_contained INTEGER DEFAULT 0,
            fragments INTEGER DEFAULT 0,
            uncertain INTEGER DEFAULT 0,
            skipped INTEGER DEFAULT 0,
            duration_seconds REAL,
            model_name TEXT,
            git_commit TEXT,
            FOREIGN KEY (repo_id) REFERENCES repos(id)
        );

        -- Prefilter runs (for classifying files as self-contained vs fragments)
        CREATE TABLE IF NOT EXISTS prefilter_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            status TEXT DEFAULT 'running',
            data_source TEXT DEFAULT 'papers_with_code',
            total_files INTEGER DEFAULT 0,
            self_contained INTEGER DEFAULT 0,
            fragments INTEGER DEFAULT 0,
            uncertain INTEGER DEFAULT 0,
            errors INTEGER DEFAULT 0,
            seed INTEGER,
            config TEXT,
            model_name TEXT,
            git_commit TEXT,
            parent_run_id INTEGER
        );

        -- Prefilter file results (per-file classification)
        CREATE TABLE IF NOT EXISTS prefilter_file_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prefilter_run_id INTEGER NOT NULL,
            file_id INTEGER NOT NULL,
            classification TEXT NOT NULL,
            confidence REAL,
            reasoning TEXT,
            classified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (prefilter_run_id) REFERENCES prefilter_runs(id),
            FOREIGN KEY (file_id) REFERENCES files(id),
            UNIQUE(prefilter_run_id, file_id)
        );

        -- Indexes for common queries
        CREATE INDEX IF NOT EXISTS idx_pattern_runs_analysis ON pattern_runs(file_analysis_id);
        CREATE INDEX IF NOT EXISTS idx_pattern_runs_pattern ON pattern_runs(pattern_id);
        CREATE INDEX IF NOT EXISTS idx_pattern_runs_status ON pattern_runs(status);
        CREATE INDEX IF NOT EXISTS idx_repos_domain ON repos(domain);
        CREATE INDEX IF NOT EXISTS idx_verifications_finding ON finding_verifications(finding_id);
        CREATE INDEX IF NOT EXISTS idx_verifications_status ON finding_verifications(status);
        CREATE INDEX IF NOT EXISTS idx_repos_name ON repos(repo_name);
        CREATE INDEX IF NOT EXISTS idx_files_repo ON files(repo_id);
        CREATE INDEX IF NOT EXISTS idx_file_analyses_run ON file_analyses(run_id);
        CREATE INDEX IF NOT EXISTS idx_file_analyses_file ON file_analyses(file_id);
        CREATE INDEX IF NOT EXISTS idx_findings_analysis ON findings(file_analysis_id);
        CREATE INDEX IF NOT EXISTS idx_findings_pattern ON findings(pattern_id);
        CREATE INDEX IF NOT EXISTS idx_findings_category ON findings(category);
        CREATE INDEX IF NOT EXISTS idx_findings_severity ON findings(severity);
        CREATE INDEX IF NOT EXISTS idx_analysis_runs_date ON analysis_runs(started_at);
        CREATE INDEX IF NOT EXISTS idx_repo_scans_repo ON repo_scans(repo_id);
        CREATE INDEX IF NOT EXISTS idx_files_classification ON files(self_contained_class);
        CREATE INDEX IF NOT EXISTS idx_prefilter_results_run
            ON prefilter_file_results(prefilter_run_id);
        CREATE INDEX IF NOT EXISTS idx_prefilter_results_file
            ON prefilter_file_results(file_id);
        CREATE INDEX IF NOT EXISTS idx_prefilter_results_class
            ON prefilter_file_results(classification);
    """)

    # Migration: Add data_source column if missing (for existing databases)
    cursor = conn.execute("PRAGMA table_info(analysis_runs)")
    columns = {row[1] for row in cursor.fetchall()}
    if "data_source" not in columns:
        conn.execute(
            "ALTER TABLE analysis_runs ADD COLUMN data_source TEXT DEFAULT 'papers_with_code'"
        )
        logger.info("Migrated analysis_runs: added data_source column")

    # Migration: Add authors column to papers if missing
    cursor = conn.execute("PRAGMA table_info(papers)")
    paper_columns = {row[1] for row in cursor.fetchall()}
    if "authors" not in paper_columns:
        conn.execute("ALTER TABLE papers ADD COLUMN authors TEXT")
        logger.info("Migrated papers: added authors column")

    # Migration: Add self-contained classification columns to files if missing
    cursor = conn.execute("PRAGMA table_info(files)")
    file_columns = {row[1] for row in cursor.fetchall()}
    if "has_ml_imports" not in file_columns:
        conn.execute("ALTER TABLE files ADD COLUMN has_ml_imports BOOLEAN")
        logger.info("Migrated files: added has_ml_imports column")
    if "self_contained_class" not in file_columns:
        conn.execute("ALTER TABLE files ADD COLUMN self_contained_class TEXT")
        logger.info("Migrated files: added self_contained_class column")
    if "self_contained_confidence" not in file_columns:
        conn.execute("ALTER TABLE files ADD COLUMN self_contained_confidence REAL")
        logger.info("Migrated files: added self_contained_confidence column")
    if "self_contained_reasoning" not in file_columns:
        conn.execute("ALTER TABLE files ADD COLUMN self_contained_reasoning TEXT")
        logger.info("Migrated files: added self_contained_reasoning column")
    if "scan_id" not in file_columns:
        conn.execute("ALTER TABLE files ADD COLUMN scan_id INTEGER")
        logger.info("Migrated files: added scan_id column")

    # Migration: Add prefilter_run_id column to analysis_runs if missing
    cursor = conn.execute("PRAGMA table_info(analysis_runs)")
    run_columns = {row[1] for row in cursor.fetchall()}
    if "prefilter_run_id" not in run_columns:
        conn.execute("ALTER TABLE analysis_runs ADD COLUMN prefilter_run_id INTEGER")
        logger.info("Migrated analysis_runs: added prefilter_run_id column")

    # Migration: Add name-based location columns to findings if missing
    cursor = conn.execute("PRAGMA table_info(findings)")
    finding_columns = {row[1] for row in cursor.fetchall()}
    if "location_name" not in finding_columns:
        conn.execute("ALTER TABLE findings ADD COLUMN location_name TEXT")
        logger.info("Migrated findings: added location_name column")
    if "location_type" not in finding_columns:
        conn.execute("ALTER TABLE findings ADD COLUMN location_type TEXT")
        logger.info("Migrated findings: added location_type column")
    if "focus_line" not in finding_columns:
        conn.execute("ALTER TABLE findings ADD COLUMN focus_line INTEGER")
        logger.info("Migrated findings: added focus_line column")

    conn.commit()
    return conn
