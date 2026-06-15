---
title: 'scicode-lint: Detecting Methodology Bugs in Scientific Python Code with LLM-Generated Patterns'
tags:
  - Python
  - research software engineering
  - machine learning
  - static analysis
  - linter
  - reproducibility
  - data leakage
  - large language models
authors:
  - name: Sergey V. Samsonau
    orcid: 0000-0002-0835-2970
    affiliation: 1
affiliations:
  - name: Authentic Research Partners, Princeton, NJ, USA
    index: 1
date: 15 June 2026
bibliography: paper.bib
---

# Summary

Methodology bugs in scientific Python code do not crash programs; they produce
plausible but incorrect results. Data leakage, missing random seeds, incorrect
cross-validation, and silent numerical errors let code run cleanly, pass tests,
and still yield conclusions that are wrong. Traditional linters and static
analysis tools cannot catch these bugs, because detecting them requires semantic
understanding of scientific methodology rather than syntax or style checks.

`scicode-lint` is a command-line linter and Python library that detects such
bugs in scientific Python scripts and Jupyter notebooks. It ships 66 detection
patterns across five categories (AI training, AI inference, scientific-numerical,
scientific-performance, and scientific-reproducibility) and reports what is wrong
and why it matters, with links to the relevant library documentation. It performs
detection only and never rewrites code, so the researcher stays in control.

The tool runs a small local large language model (LLM) rather than a cloud API:
code never leaves the machine; there are no per-query costs; and results stay
reproducible because open-weight models remain available over time. A companion
preprint [@samsonau2026scicodelint] describes the architecture and evaluation in
detail.

# Statement of need

Modern research depends on software at every stage, and over half of researchers
develop their own [@hettrick2014survey]. Most of that code is written by domain
scientists for whom software engineering is a means to an end and who rarely have
access to expert code review. As machine learning becomes a core component of
this work, a specific and damaging class of bug appears: methodology errors that
inflate performance metrics instead of producing obviously wrong output. Data
leakage alone has been documented across 329 papers in 17 fields
[@kapoor2023leakage]. AI coding assistants, trained on public repositories full of
the same mistakes, accelerate the problem by generating more code than humans can
review.

Several ML-specific linters have demonstrated that automated methodology checking
is feasible [@haakman2022dslinter; @shivashankar2025mlscent; @vanoort2022mllint;
@eghbali2025dylin], but they share a sustainability problem: dependence on
specific `pylint` or Python versions, no PyPI packaging, and reliance on manual
engineering to author and maintain every detection rule. General-purpose AI code
review tools, meanwhile, catch logic and security bugs but not domain-specific
methodology errors. `scicode-lint` targets this gap with an architecture designed
so that maintenance costs tokens rather than engineering hours, making the tool
viable to sustain beyond a single project or graduate thesis.

# Key features

- **Two-tier architecture.** Frontier models design detection patterns at build
  time (analyzing library documentation, writing focused detection questions, and
  generating test cases); a small local model (`RedHatAI/Qwen3-8B-FP8-dynamic`,
  fits in 16 GB VRAM) executes those questions at runtime. Adapting to new library
  versions is a build-time step against updated documentation, not a rewrite.
- **Local, private execution** via vLLM [@kwon2023vllm]. Code-first prompt ordering
  lets all 66 patterns share a cached prefix and run concurrently, so scanning a
  file against all patterns takes roughly the time of a single pattern.
- **Dual-audience output.** Structured JSON for AI coding agents and CI, plus
  human-readable explanations with documentation links. Exit codes follow linter
  convention so pipelines can branch on the outcome.
- **Flexible deployment.** A workstation GPU, a shared institutional vLLM endpoint
  used from a laptop, a CI/CD gate, or an AI-agent feedback loop.

# State of the field

Existing ML-linting tools cover an estimated 9-14% of `scicode-lint`'s patterns
and leave categories such as PyTorch inference-mode errors, temporal leakage, and
numerical-stability checks unaddressed [@samsonau2026scicodelint]. Static
data-flow approaches achieve high precision on the narrow leakage types they target
[@yang2022leakage; @drobnjakovic2024abstract] but are difficult to extend. Hybrid
systems that pair LLMs with static analysis have validated the paradigm for
security bugs in systems code [@li2024lllift; @yang2025knighter]; `scicode-lint`
applies it to scientific methodology under the additional constraint of local
execution on commodity hardware.

On controlled per-pattern tests `scicode-lint` reaches 97.7% accuracy; on
human-labeled Kaggle notebooks [@yang2022leakage] preprocessing-leakage detection
reaches 65% precision at 100% recall; on code from published scientific papers,
LLM-judged precision is 54-62% across categories. These figures are a baseline for
the smallest viable runtime model and first-generation patterns; the architecture
is designed so that better patterns, larger models, and multi-file context each
improve results independently.

# Acknowledgements

`scicode-lint` is developed with AI assistance. Detection patterns (detection
questions, test files, metadata) and code are created and revised with Claude
Opus via Claude Code; automated semantic validation, integration-test generation
and judging, and finding verification use Claude Sonnet (Claude models in the
4.5-4.8 version range). The runtime detection model is
`RedHatAI/Qwen3-8B-FP8-dynamic`, served via vLLM. This paper was drafted with
Claude via Claude Code. The author wrote the specifications, evaluation
harnesses, and quality gates, made all core design decisions, and reviewed,
edited, and validated all AI-generated patterns, code, and text.

# References
