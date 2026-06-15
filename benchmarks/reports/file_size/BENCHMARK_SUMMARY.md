# File Size Benchmark Report

**Date:** 2026-05-31
**Mode:** full scan (all patterns)
**Runs per file:** 3

## Results

| File | Lines | Findings | Mean (s) | Min (s) | Max (s) | Spread (s) |
|------|------:|--------:|---------:|--------:|--------:|-----------:|
| small_30_lines.py | 30 | 6 | 62.9 | 47.75 | 88.84 | 41.09 |
| medium_200_lines.py | 203 | 7 | 96.77 | 92.32 | 103.91 | 11.59 |
| large_500_lines.py | 487 | 8 | 127.39 | 116.53 | 146.29 | 29.76 |
| xlarge_1000_lines.py | 984 | 4 | 154.18 | 141.74 | 162.24 | 20.5 |

## Summary

- **Smallest file:** 30 lines → 62.9s
- **Largest file:** 984 lines → 154.18s
- **Line count ratio:** 33x
- **Time ratio:** 2.5x
- **Scaling:** Sub-linear (2.5x time for 33x lines)