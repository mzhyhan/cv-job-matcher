"""Tests for document text normalisation."""

from __future__ import annotations

from cv_matcher import normalise_text


def test_nfkc_canonicalises_ligatures() -> None:
    assert normalise_text("ef\ufb01cient") == "efficient"


def test_bom_and_zero_width_characters_are_removed() -> None:
    raw = "\ufeffHello\u200b world"
    assert normalise_text(raw) == "Hello world"


def test_crlf_becomes_unix_newlines() -> None:
    assert normalise_text("one\r\ntwo\rthree") == "one\ntwo\nthree"


def test_trailing_spaces_and_blank_line_runs() -> None:
    raw = "title  \n\n\n\nbody   \n"
    assert normalise_text(raw) == "title\n\nbody"


def test_hyphenation_across_line_breaks_is_joined() -> None:
    assert normalise_text("experi-\nence with Python") == "experience with Python"


def test_hyphenation_is_not_joined_for_regular_lists() -> None:
    text = "Skills:\n- Python\n- Docker"
    assert normalise_text(text) == text


def test_strip_markdown_keeps_wording() -> None:
    raw = (
        "# AI Engineer\n\n"
        "We need a **Python** engineer who knows [NLP](https://example.com) "
        "and `FastAPI`.\n\n"
        "![Team photo](photo.png)\n\n"
        "```python\nprint('hello')\n```\n"
        "- embeddings\n"
        "1. evaluation\n"
    )
    cleaned = normalise_text(raw, strip_markdown=True)
    assert "#" not in cleaned
    assert "**" not in cleaned
    assert "https://example.com" not in cleaned
    assert "Python" in cleaned
    assert "NLP" in cleaned
    assert "FastAPI" in cleaned
    assert "Team photo" in cleaned
    assert "print('hello')" in cleaned
    assert "embeddings" in cleaned
    assert "evaluation" in cleaned


def test_whitespace_only_becomes_empty() -> None:
    assert normalise_text("  \n\t  ") == ""


def test_clean_text_is_unchanged() -> None:
    text = "Python engineer\n\nBuilt NLP pipelines."
    assert normalise_text(text) == text
