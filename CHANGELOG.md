# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-26

### Added

- Document loaders for PDF, DOCX, TXT and Markdown, with text normalisation
  (NFKC, hyphenation repair, whitespace tidy-up and optional Markdown stripping).
- Curated taxonomy of 153 software and AI skills, with alias, punctuation-variant
  and conservative fuzzy extraction.
- `match_cv()` overall score from skill overlap (default 60%) and text
  similarity (40%). TF-IDF is the default backend; sentence-transformers
  embeddings are available through the optional `[embeddings]` extra.
- Pluggable gap analysis: a deterministic offline rule-based analyser, plus an
  OpenAI-compatible provider used only when an API key is set.
- `cv-match` CLI: scores, matched / missing / extra skills and suggestions,
  with `--json`, `--backend`, `--provider` and `-o` output options.
- FastAPI web demo: `POST /match`, `GET /health`, OpenAPI docs and a
  drag-and-drop upload page.
- Dockerfile (non-root user and health check) for the web demo.
- Test suite with a 95% coverage floor, strict mypy, ruff linting/formatting,
  pre-commit hooks and GitHub Actions CI on Python 3.10–3.13.

[Unreleased]: https://github.com/mzhyhan/cv-job-matcher/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mzhyhan/cv-job-matcher/releases/tag/v0.1.0
