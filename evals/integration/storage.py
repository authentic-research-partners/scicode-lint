"""File operations for saving/loading integration evaluation scenarios."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .models import ScenarioResult

# Directories
INTEGRATION_DIR = Path(__file__).parent
GENERATED_DIR = INTEGRATION_DIR / "generated"


def save_scenarios(
    scenarios: list[ScenarioResult],
    run_dir: Path,
    seed: int = 42,
    generation_stats: dict[str, Any] | None = None,
) -> None:
    """Save scenarios to disk in a run directory.

    Args:
        scenarios: List of generated scenarios
        run_dir: Directory for this run (e.g., generated/20260316_003500/)
        seed: Random seed used for generation
        generation_stats: Optional stats about generation (attempts, rejections, etc.)
    """
    scenarios_dir = run_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    # Metadata section
    verified_count = sum(1 for s in scenarios if s.verified)
    config: dict[str, Any] = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "seed": seed,
            "total_scenarios": len(scenarios),
            "verified_scenarios": verified_count,
            "unverified_scenarios": len(scenarios) - verified_count,
        },
        "scenarios": {},
    }

    # Add generation stats if provided
    if generation_stats:
        config["metadata"]["generation"] = generation_stats

    for scenario in scenarios:
        # Save code file
        file_path = scenarios_dir / f"{scenario.name}.py"
        file_path.write_text(scenario.code)

        # Convert manifest to expected_patterns format
        expected_patterns: dict[str, int] = {}
        for entry in scenario.manifest:
            pid = entry["pattern_id"]
            expected_patterns[pid] = expected_patterns.get(pid, 0) + 1

        config["scenarios"][scenario.name] = {
            "description": f"Generated scenario with {len(scenario.patterns)} patterns",
            "file": f"scenarios/{file_path.name}",
            "expected_patterns": expected_patterns,
            "bugs": scenario.manifest,
            "verified": scenario.verified,
        }

    config["evaluation"] = {
        "run_all_patterns": True,
        "verbose": True,
        "stop_on_first_failure": False,
    }

    expected_path = run_dir / "expected.yaml"
    with open(expected_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def clear_generated_run(run_dir: Path) -> None:
    """Remove a specific generated run."""
    if run_dir.exists():
        shutil.rmtree(run_dir)
        print(f"Cleared {run_dir}")


def clear_all_generated(generated_dir: Path) -> None:
    """Remove all generated runs."""
    if generated_dir.exists():
        shutil.rmtree(generated_dir)
        print(f"Cleared all generated runs in {generated_dir}")


def list_saved_runs() -> list[tuple[str, Path, datetime]]:
    """List all saved runs with their modification times."""
    if not GENERATED_DIR.exists():
        return []

    runs = []
    for run_dir in GENERATED_DIR.iterdir():
        if run_dir.is_dir() and not run_dir.name.startswith("."):
            expected_yaml = run_dir / "expected.yaml"
            if expected_yaml.exists():
                mtime = datetime.fromtimestamp(expected_yaml.stat().st_mtime)
                runs.append((run_dir.name, run_dir, mtime))

    # Sort by modification time (newest first)
    runs.sort(key=lambda x: x[2], reverse=True)
    return runs


def load_scenarios_from_disk(
    run_dir: Path, include_unverified: bool = False
) -> list[tuple[str, str, list[dict[str, Any]], bool]]:
    """Load scenarios from a saved run.

    Args:
        run_dir: Directory containing expected.yaml and scenarios/
        include_unverified: If False, skip scenarios that failed verification

    Returns:
        List of (name, code, manifest, verified) tuples
    """
    expected_yaml = run_dir / "expected.yaml"
    if not expected_yaml.exists():
        return []

    with open(expected_yaml) as f:
        config = yaml.safe_load(f)

    scenarios = []
    skipped = 0
    for name, scenario_config in config.get("scenarios", {}).items():
        verified = scenario_config.get("verified", True)

        # Skip unverified scenarios unless explicitly included
        if not verified and not include_unverified:
            skipped += 1
            continue

        # Load code from file
        code_file = run_dir / scenario_config["file"]
        if code_file.exists():
            code = code_file.read_text()
            manifest = scenario_config.get("bugs", [])
            scenarios.append((name, code, manifest, verified))

    if skipped > 0:
        print(f"  Skipped {skipped} unverified scenarios (use --include-unverified to include)")

    return scenarios
