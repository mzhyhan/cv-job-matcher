"""Command-line interface for cv-job-matcher.

Installed as the ``cv-match`` console script (also runnable as
``python -m cv_matcher``).

Examples:
    Score a CV against a job description::

        $ cv-match cv.pdf job.md

    Machine-readable output, offline suggestions only::

        $ cv-match cv.pdf job.md --json --provider rules

    Use an OpenAI-compatible model and save the report::

        $ OPENAI_API_KEY=... cv-match cv.docx job.txt --provider openai -o report.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from cv_matcher import __version__
from cv_matcher.analysis import DEFAULT_BASE_URL, DEFAULT_OPENAI_MODEL, analyse_gap
from cv_matcher.loaders import load_document
from cv_matcher.models import (
    AnalysisError,
    AnalysisResult,
    DocumentKind,
    EmbeddingsUnavailableError,
    LoadError,
)

__all__ = ["build_parser", "main"]

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE_ERROR = 2


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for ``cv-match``.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="cv-match",
        description="Match a CV against a job description and suggest improvements.",
    )
    parser.add_argument("cv", type=Path, help="CV file (PDF, DOCX, TXT or Markdown)")
    parser.add_argument("job", type=Path, help="job description file (PDF, DOCX, TXT or Markdown)")
    parser.add_argument("--json", action="store_true", help="output machine-readable JSON")
    parser.add_argument(
        "--backend",
        choices=["auto", "tfidf", "embeddings"],
        default="auto",
        help="text-similarity backend (embeddings need the optional extra)",
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "rules", "openai"],
        default="auto",
        help="suggestion provider; auto uses OpenAI only if an API key is set",
    )
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL, help="OpenAI model name")
    parser.add_argument("--api-key", help="API key (default: OPENAI_API_KEY environment variable)")
    parser.add_argument("-o", "--output", type=Path, help="write output to a file")
    parser.add_argument("-v", "--verbose", action="store_true", help="log pipeline details")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _to_dict(result: AnalysisResult) -> dict[str, object]:
    """Convert an analysis to a JSON-serialisable dictionary.

    Args:
        result: The analysis to convert.

    Returns:
        A dictionary with deterministic key order.
    """
    match = result.match
    return {
        "overall_score": round(match.overall_score, 4),
        "skill_score": round(match.skill_score, 4),
        "text_score": round(match.text_score, 4),
        "backend": match.backend,
        "matched_skills": [m.name for m in match.matched_skills],
        "missing_skills": [h.name for h in match.missing_skills],
        "extra_skills": [h.name for h in match.extra_skills],
        "provider": result.provider,
        "summary": result.summary,
        "suggestions": [
            {"title": s.title, "detail": s.detail, "skill": s.skill, "category": s.category}
            for s in result.suggestions
        ],
    }


def _format_text(result: AnalysisResult) -> str:
    """Render an analysis as a plain-text report.

    Args:
        result: The analysis to render.

    Returns:
        The report text.
    """
    match = result.match

    def names(items: Sequence[str]) -> str:
        return ", ".join(items) if items else "none"

    lines = [
        f"Overall score: {match.overall_score:.0%}",
        f"  Skill score: {match.skill_score:.0%}",
        f"  Text score:  {match.text_score:.0%} ({match.backend})",
        "",
        f"Matched skills: {names([m.name for m in match.matched_skills])}",
        f"Missing skills: {names([h.name for h in match.missing_skills])}",
        f"Extra skills:   {names([h.name for h in match.extra_skills])}",
        "",
        f"Summary: {result.summary}",
        "",
        f"Suggestions (provider: {result.provider}):",
    ]
    for number, suggestion in enumerate(result.suggestions, start=1):
        lines.append(f"  {number}. {suggestion.title}")
        lines.append(f"     {suggestion.detail}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ``cv-match`` command.

    Args:
        argv: Command-line arguments without the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: ``0`` on success, ``1`` if loading or analysis failed,
        ``2`` for usage errors such as a missing input file.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger("cv_matcher").setLevel(logging.DEBUG if args.verbose else logging.WARNING)

    for path in (args.cv, args.job):
        if not path.is_file():
            print(f"error: file not found: {path}", file=sys.stderr)
            return EXIT_USAGE_ERROR

    try:
        cv = load_document(args.cv, DocumentKind.CV)
        job = load_document(args.job, DocumentKind.JOB_DESCRIPTION)
        result = analyse_gap(
            cv,
            job,
            provider=args.provider,
            api_key=args.api_key,
            base_url=DEFAULT_BASE_URL,
            model=args.model,
            backend=args.backend,
        )
    except FileNotFoundError as exc:
        print(f"error: file not found: {exc.filename or exc}", file=sys.stderr)
        return EXIT_USAGE_ERROR
    except (LoadError, AnalysisError, EmbeddingsUnavailableError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    output = json.dumps(_to_dict(result), indent=2) if args.json else _format_text(result)
    if args.output:
        args.output.write_text(output + "\n", encoding="utf-8")
        logger.info("Wrote output to %s", args.output)
    else:
        print(output)
    return EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
