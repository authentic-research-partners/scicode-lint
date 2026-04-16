#!/usr/bin/env python3
"""Check dependencies and code for security issues.

Combines:
1. pip-audit - dependency vulnerability scanning (PyPI advisory DB)
2. safety - dependency vulnerability scanning (Safety DB)
3. bandit - static code security analysis
4. Fresh venv install - capture deprecation warnings

Usage:
    python scripts/check_dependencies.py
    python scripts/check_dependencies.py --groups dev,dashboard
    python scripts/check_dependencies.py --package pynvml
    python scripts/check_dependencies.py --skip-bandit --skip-warnings
"""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def run_pip_audit() -> tuple[bool, str]:
    """Run pip-audit on current environment."""
    try:
        result = subprocess.run(
            ["pip-audit", "--strict"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        return result.returncode == 0, result.stdout + result.stderr
    except FileNotFoundError:
        return False, "pip-audit not installed. Run: pip install pip-audit"
    except subprocess.TimeoutExpired:
        return False, "pip-audit timed out"


def run_safety() -> tuple[bool, str]:
    """Run safety vulnerability check on installed packages."""
    try:
        result = subprocess.run(
            ["safety", "check", "--output", "text"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        passed = result.returncode == 0
        if passed:
            return True, "No vulnerabilities found."
        return False, output
    except FileNotFoundError:
        return False, "safety not installed. Run: pip install safety"
    except subprocess.TimeoutExpired:
        return False, "safety timed out"


def run_bandit() -> tuple[bool, str]:
    """Run bandit static security analysis on source code."""
    src_path = Path(__file__).parent.parent / "src"
    try:
        result = subprocess.run(
            [
                "bandit",
                "-r",
                str(src_path),
                "-f",
                "txt",
                "--severity-level",
                "medium",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        # Bandit returns 0 if no issues, 1 if issues found
        passed = result.returncode == 0
        if passed:
            return True, "No security issues found."
        return False, output
    except FileNotFoundError:
        return False, "bandit not installed. Run: pip install bandit"
    except subprocess.TimeoutExpired:
        return False, "bandit timed out"


def check_package_warnings(packages: list[str]) -> dict[str, list[str]]:
    """Install packages in fresh venv and capture warnings.

    Args:
        packages: List of package specs (e.g., ["pynvml", "requests>=2.0"])

    Returns:
        Dict mapping package to list of warnings found
    """
    warnings: dict[str, list[str]] = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        venv_path = Path(tmpdir) / "venv"

        # Create fresh venv
        print("Creating temporary venv...")
        result = subprocess.run(
            [sys.executable, "-m", "venv", str(venv_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"Failed to create venv: {result.stderr}")
            return warnings

        # Get pip path in venv
        if sys.platform == "win32":
            pip_path = venv_path / "Scripts" / "pip"
        else:
            pip_path = venv_path / "bin" / "pip"

        # Upgrade pip first (quietly)
        subprocess.run(
            [str(pip_path), "install", "--upgrade", "pip"],
            capture_output=True,
            timeout=60,
        )

        # Install each package and capture output
        for package in packages:
            print(f"  Checking {package}...")
            result = subprocess.run(
                [str(pip_path), "install", package],
                capture_output=True,
                text=True,
                timeout=120,
            )

            output = result.stdout + result.stderr
            pkg_warnings = []

            # Look for common warning patterns
            warning_patterns = [
                r"DEPRECAT\w*:.*",
                r"WARNING:.*deprecat.*",
                r"is deprecated.*",
                r"will be removed.*",
                r"no longer maintained.*",
                r"please use .* instead.*",
                r"FutureWarning:.*",
            ]

            for line in output.split("\n"):
                for pattern in warning_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        pkg_warnings.append(line.strip())
                        break

            if pkg_warnings:
                warnings[package] = pkg_warnings

    return warnings


def get_packages_from_pyproject(groups: list[str] | None = None) -> list[str]:
    """Extract packages from pyproject.toml."""
    import tomllib

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"

    if not pyproject_path.exists():
        return []

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    packages = []

    # Main dependencies
    if groups is None or "main" in groups:
        packages.extend(data.get("project", {}).get("dependencies", []))

    # Optional dependencies
    optional = data.get("project", {}).get("optional-dependencies", {})
    if groups is None:
        # All groups
        for group_packages in optional.values():
            packages.extend(group_packages)
    else:
        for group in groups:
            if group in optional:
                packages.extend(optional[group])

    # Clean up package specs (remove version constraints for warning check)
    return packages


def main() -> int:
    parser = argparse.ArgumentParser(description="Check dependencies for issues")
    parser.add_argument(
        "--groups",
        help="Comma-separated optional dependency groups (e.g., dev,dashboard)",
    )
    parser.add_argument(
        "--package",
        action="append",
        dest="packages",
        help="Specific package to check (can be used multiple times)",
    )
    parser.add_argument(
        "--skip-audit",
        action="store_true",
        help="Skip pip-audit dependency vulnerability check",
    )
    parser.add_argument(
        "--skip-safety",
        action="store_true",
        help="Skip safety dependency vulnerability check",
    )
    parser.add_argument(
        "--skip-bandit",
        action="store_true",
        help="Skip bandit static code security analysis",
    )
    parser.add_argument(
        "--skip-warnings",
        action="store_true",
        help="Skip fresh venv deprecation warning check",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("DEPENDENCY HEALTH CHECK")
    print("=" * 60)

    has_issues = False

    # 1. Dependency vulnerability audit (pip-audit)
    if not args.skip_audit:
        print("\n[1/4] Dependency Vulnerabilities (pip-audit)")
        print("-" * 40)
        passed, output = run_pip_audit()
        print(output)
        if not passed and "not installed" not in output:
            has_issues = True

    # 2. Dependency vulnerability audit (safety)
    if not args.skip_safety:
        print("\n[2/4] Dependency Vulnerabilities (safety)")
        print("-" * 40)
        passed, output = run_safety()
        print(output)
        if not passed and "not installed" not in output:
            has_issues = True

    # 3. Static code security analysis
    if not args.skip_bandit:
        print("\n[3/4] Code Security Analysis (bandit)")
        print("-" * 40)
        passed, output = run_bandit()
        print(output)
        if not passed and "not installed" not in output:
            has_issues = True

    # 4. Deprecation warnings from fresh install
    if not args.skip_warnings:
        print("\n[4/4] Deprecation Warnings (fresh venv install)")
        print("-" * 40)

        if args.packages:
            packages = args.packages
        else:
            groups = args.groups.split(",") if args.groups else None
            packages = get_packages_from_pyproject(groups)

        if not packages:
            print("No packages to check.")
        else:
            print(f"Checking {len(packages)} packages...")
            warnings = check_package_warnings(packages)

            if warnings:
                has_issues = True
                print("\nWarnings found:")
                for pkg, pkg_warnings in warnings.items():
                    print(f"\n  {pkg}:")
                    for w in pkg_warnings:
                        print(f"    - {w}")
            else:
                print("\nNo deprecation warnings found.")

    print("\n" + "=" * 60)
    if has_issues:
        print("ISSUES FOUND - Review above")
        return 1
    else:
        print("ALL CHECKS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
