"""Paper, repo, and file record operations."""

import json
import sqlite3
from typing import Any

__all__ = [
    "insert_paper",
    "get_or_create_repo",
    "insert_file",
    "get_file_id",
]


def insert_paper(conn: sqlite3.Connection, paper_data: dict[str, Any]) -> int:
    """Insert or get paper record.

    Args:
        conn: Database connection.
        paper_data: Paper metadata dict.

    Returns:
        Paper ID.
    """
    paper_url = paper_data.get("paper_url", "")
    if not paper_url:
        return 0

    # Try to get existing
    cursor = conn.execute("SELECT id FROM papers WHERE paper_url = ?", (paper_url,))
    row = cursor.fetchone()
    if row:
        return int(row["id"])

    # Insert new
    tasks = paper_data.get("tasks", [])
    authors = paper_data.get("authors", [])
    cursor = conn.execute(
        """
        INSERT INTO papers (paper_url, title, arxiv_id, abstract, authors, tasks, matched_domain)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            paper_url,
            paper_data.get("paper_title", ""),
            paper_data.get("arxiv_id", ""),
            paper_data.get("abstract", ""),
            json.dumps(authors) if isinstance(authors, list) else authors,
            json.dumps(tasks) if isinstance(tasks, list) else tasks,
            paper_data.get("domain", ""),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.lastrowid else 0


def get_or_create_repo(conn: sqlite3.Connection, repo_data: dict[str, Any]) -> int:
    """Get existing repo or create new one.

    Args:
        conn: Database connection.
        repo_data: Dict with repo metadata.

    Returns:
        Repo ID.
    """
    repo_name = repo_data.get("repo_name", "")
    if not repo_name:
        return 0

    # Try to get existing
    cursor = conn.execute("SELECT id FROM repos WHERE repo_name = ?", (repo_name,))
    row = cursor.fetchone()
    if row:
        return int(row["id"])

    # Get or create paper first
    paper_id = insert_paper(conn, repo_data) if repo_data.get("paper_url") else None

    # Create repo
    cursor = conn.execute(
        """
        INSERT INTO repos (repo_url, repo_name, data_source, paper_id, domain, clone_status)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            repo_data.get("repo_url", ""),
            repo_name,
            repo_data.get("data_source", "papersWithCode"),
            paper_id,
            repo_data.get("domain", ""),
            "success",  # assume cloned if we're analyzing
        ),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.lastrowid else 0


def insert_file(conn: sqlite3.Connection, repo_id: int, file_data: dict[str, Any]) -> int:
    """Insert or update file record.

    Args:
        conn: Database connection.
        repo_id: Parent repo ID.
        file_data: Dict with file metadata.

    Returns:
        File ID.
    """
    file_path = file_data.get("file_path", "")

    # Check for existing
    cursor = conn.execute(
        "SELECT id FROM files WHERE repo_id = ? AND file_path = ?",
        (repo_id, file_path),
    )
    row = cursor.fetchone()
    if row:
        return int(row["id"])

    # Parse imports
    ml_imports = file_data.get("ml_imports", "")
    if isinstance(ml_imports, list):
        ml_imports = ",".join(ml_imports)

    scientific_imports = file_data.get("scientific_imports", "")
    if isinstance(scientific_imports, list):
        scientific_imports = ",".join(scientific_imports)

    cursor = conn.execute(
        """
        INSERT INTO files
        (repo_id, file_path, original_path, is_notebook, file_size, line_count,
         ml_imports, scientific_imports, prefilter_passed, prefilter_response)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            repo_id,
            file_path,
            file_data.get("original_path", ""),
            file_data.get("is_notebook", False),
            file_data.get("file_size", 0),
            file_data.get("line_count", 0),
            ml_imports,
            scientific_imports,
            file_data.get("prefilter_passed", True),
            file_data.get("prefilter_response", ""),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid) if cursor.lastrowid else 0


def get_file_id(conn: sqlite3.Connection, file_path: str) -> int | None:
    """Get file ID by path.

    Args:
        conn: Database connection.
        file_path: File path to look up.

    Returns:
        File ID or None.
    """
    # Try exact match
    cursor = conn.execute("SELECT id FROM files WHERE file_path = ?", (file_path,))
    row = cursor.fetchone()
    if row:
        return int(row["id"])

    # Try partial match (for absolute paths)
    if "/" in file_path:
        # Extract relative path
        parts = file_path.split("files/")
        if len(parts) > 1:
            relative_path = "files/" + parts[-1]
            cursor = conn.execute("SELECT id FROM files WHERE file_path = ?", (relative_path,))
            row = cursor.fetchone()
            if row:
                return int(row["id"])

    return None
