# Contributing

Thanks for your interest in improving cv-job-matcher!

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

The optional `embeddings` extra (`pip install -e ".[embeddings]"`) is only
needed if you want to run the sentence-transformers backend locally. Tests stay
offline and do not download a model.

## Checks

Every pull request must pass the same checks CI runs:

```bash
ruff check . && ruff format --check .   # lint + formatting
mypy                                    # strict type checking
pytest --cov                            # tests, doctests and coverage (>= 95%)
```

`pre-commit run --all-files` runs ruff, ruff-format, mypy and basic file hygiene hooks in one go.

## Guidelines

- Write Google-style docstrings for public modules, classes and functions.
- Add or update tests for any behaviour change. Network calls and model
  downloads should be stubbed so the suite can run offline.
- Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, ...).
- Record user-facing changes in [CHANGELOG.md](CHANGELOG.md) under *Unreleased*.
