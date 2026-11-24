# CV Job Matcher

Match a curriculum vitae against a job description: extract skills, score the
fit, and suggest targeted improvements.

> This project is a work in progress. The public API, CLI and web interface are
> not stable yet.

## Current status

**Document loading, skill extraction, match scoring, gap analysis and the
`cv-match` CLI are in place.** You can extract and normalise text from CVs and
job descriptions in PDF, DOCX, TXT and Markdown formats, extract skills from a
curated taxonomy, compute an overall match score from skill overlap and text
similarity, and get improvement suggestions. TF-IDF is the default similarity
backend; sentence-transformers embeddings are available through the optional
`embeddings` extra (`pip install -e ".[embeddings]"`). Suggestions come from an
offline rule-based analyser by default, or from any OpenAI-compatible endpoint
when `OPENAI_API_KEY` is set.

Coming next:

- A small web app and API
- Docker image and CI

## Requirements

- Python 3.10 or later

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Library usage

```python
from cv_matcher import DocumentKind, load_document

cv = load_document("path/to/cv.pdf", kind=DocumentKind.CV)
job = load_document("path/to/job.md", kind=DocumentKind.JOB_DESCRIPTION)
print(cv.text)
print(job.text)
```

```python
from cv_matcher import match_cv

result = match_cv(cv, job)  # LoadedDocument or plain text
print(f"{result.overall_score:.0%}", result.backend)
print([m.name for m in result.matched_skills])
print([s.name for s in result.missing_skills])
```

```python
from cv_matcher import analyse_gap

analysis = analyse_gap(cv, job)  # rule-based unless OPENAI_API_KEY is set
print(analysis.provider, analysis.summary)
for suggestion in analysis.suggestions:
    print(f"- {suggestion.title}: {suggestion.detail}")
```

`load_from_bytes` accepts in-memory file contents (for example an upload) and
returns the same `LoadedDocument` structure.

## Command line

```bash
cv-match path/to/cv.pdf path/to/job.md
cv-match cv.pdf job.md --json --provider rules
cv-match cv.docx job.txt --provider openai --model gpt-4o-mini -o report.txt
```

The `openai` provider reads `OPENAI_API_KEY` (and optionally `OPENAI_BASE_URL`
and `OPENAI_MODEL`); with `--provider auto` the offline rules are used when no
key is set.

## Development

```bash
ruff check .
mypy
pytest --cov
pre-commit install
```

## License

MIT — see [LICENSE](LICENSE).
