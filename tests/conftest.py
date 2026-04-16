"""Pytest configuration for scicode-lint tests."""

import sys

# Prevent Python from writing .pyc bytecode files during tests.
# This ensures tests always read fresh source, never stale cached bytecode.
sys.dont_write_bytecode = True
