"""Real-world demo: analyze scientific ML code with scicode-lint.

This module provides infrastructure for collecting and analyzing real-world
scientific ML code from multiple data sources.

Structure:
    real_world_demo/
    ├── sources/                    # Data source implementations
    │   ├── papers_with_code/       # PapersWithCode repositories
    │   └── leakage_paper/          # Yang et al. ASE'22 notebooks
    ├── run_analysis.py             # Generic analysis runner
    ├── verify_findings.py          # Finding verification with Claude
    ├── generate_report.py          # Report generation from database
    ├── config.py                   # Shared configuration
    ├── database.py                 # SQLite storage
    └── models.py, utils.py         # Shared utilities

Usage:
    # Papers with Code source
    python -m real_world_demo.sources.papers_with_code --run --papers 50

    # Leakage paper source
    python -m real_world_demo.sources.leakage_paper --run
    python -m real_world_demo.sources.leakage_paper.compare_ground_truth

    # Run analysis (works with any source)
    python -m real_world_demo.run_analysis --manifest <path> --base-dir <path>
"""

__all__ = [
    "analysis_reporting",
    "config",
    "database",
    "generate_report",
    "models",
    "report_critical",
    "report_queries",
    "run_analysis",
    "sources",
    "utils",
    "verify_findings",
]
