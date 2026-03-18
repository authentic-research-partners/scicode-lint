"""SQLite database for tracking analysis results.

Stores repos, files, findings, and analysis runs for easy querying and reporting.
Supports multiple analysis runs across different dates.

This module is a re-export facade. All implementations live in:
  - db_core: Connection, schema, migrations
  - db_papers: Paper, repo, and file record operations
  - db_analysis: Analysis runs, findings, verifications
  - db_stats: Statistics and query functions
  - db_scans: Repository scan operations
  - db_prefilter: Prefilter run operations
"""

from .db_analysis import (
    complete_analysis_run,
    get_analysis_run_data_source,
    get_finding_verification,
    get_latest_run_id,
    get_timed_out_patterns,
    get_verification_stats,
    insert_file_analysis,
    insert_findings,
    insert_pattern_runs,
    save_verification,
    start_analysis_run,
    update_pattern_run,
)
from .db_core import get_current_git_commit, get_db_path, init_db
from .db_papers import get_file_id, get_or_create_repo, insert_file, insert_paper
from .db_prefilter import (
    complete_prefilter_run,
    get_classified_file_ids,
    get_files_from_analysis_run_repos,
    get_incomplete_prefilter_run,
    get_paper_ids_from_prefilter_runs,
    get_paper_urls_from_prefilter_runs,
    get_prefilter_run,
    get_prefilter_run_files,
    get_prefilter_summary,
    insert_prefilter_result,
    list_prefilter_runs,
    start_prefilter_run,
)
from .db_scans import (
    complete_repo_scan,
    get_latest_scan_for_repo,
    get_scan_stats,
    get_self_contained_files_for_repo,
    start_repo_scan,
    update_file_classification,
)
from .db_stats import (
    get_analyzed_file_ids,
    get_incomplete_run,
    get_run_stats,
    list_runs,
    print_stats,
)

__all__ = [
    # db_core
    "get_db_path",
    "init_db",
    "get_current_git_commit",
    # db_papers
    "insert_paper",
    "get_or_create_repo",
    "insert_file",
    "get_file_id",
    # db_analysis
    "start_analysis_run",
    "get_latest_run_id",
    "complete_analysis_run",
    "insert_file_analysis",
    "get_timed_out_patterns",
    "update_pattern_run",
    "insert_pattern_runs",
    "insert_findings",
    "save_verification",
    "get_verification_stats",
    "get_finding_verification",
    "get_analysis_run_data_source",
    # db_stats
    "get_run_stats",
    "list_runs",
    "get_analyzed_file_ids",
    "get_incomplete_run",
    "print_stats",
    # db_scans
    "start_repo_scan",
    "complete_repo_scan",
    "update_file_classification",
    "get_self_contained_files_for_repo",
    "get_scan_stats",
    "get_latest_scan_for_repo",
    # db_prefilter
    "start_prefilter_run",
    "complete_prefilter_run",
    "insert_prefilter_result",
    "get_paper_ids_from_prefilter_runs",
    "get_paper_urls_from_prefilter_runs",
    "get_prefilter_run_files",
    "get_incomplete_prefilter_run",
    "get_classified_file_ids",
    "get_prefilter_run",
    "list_prefilter_runs",
    "get_files_from_analysis_run_repos",
    "get_prefilter_summary",
]
