# CV Job Matcher

Match a curriculum vitae against a job description: extract skills, score the
fit, and suggest targeted improvements.

> This project is a work in progress. The public API, CLI and web interface are
> not stable yet.

## Current status

**Document loading is in place.** You can extract and normalise text from CVs
and job descriptions in PDF, DOCX, TXT and Markdown formats.

Coming next:

- Skill extraction and similarity scoring
- Gap analysis with improvement suggestions
- A `cv-match` command-line interface
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

`load_from_bytes` accepts in-memory file contents (for example an upload) and
returns the same `LoadedDocument` structure.

## Development

```bash
ruff check .
mypy
pytest --cov
pre-commit install
```

## License

MIT — see [LICENSE](LICENSE).
