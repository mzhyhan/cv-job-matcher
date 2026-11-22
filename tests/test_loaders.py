"""Tests for CV and job-description document loading."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from pypdf import PdfWriter
from pypdf.errors import PdfReadError

from cv_matcher import (
    DocumentKind,
    EmptyDocumentError,
    LoadedDocument,
    LoadError,
    SourceFormat,
    UnsupportedFormatError,
    detect_format,
    load_document,
    load_from_bytes,
)
from tests.conftest import SAMPLE_CV_TEXT, build_docx, build_simple_pdf


def test_load_txt_cv(txt_cv_path: Path) -> None:
    document = load_document(txt_cv_path, kind=DocumentKind.CV)
    assert document.kind is DocumentKind.CV
    assert document.source_format is SourceFormat.TXT
    assert document.source_path == txt_cv_path.resolve()
    assert document.source_name == "cv.txt"
    assert document.page_count is None
    assert "Python and AI Engineer" in document.text
    assert document.character_count == len(document.text)


def test_load_markdown_job_strips_syntax(md_job_path: Path) -> None:
    document = load_document(md_job_path, kind="job_description")
    assert document.kind is DocumentKind.JOB_DESCRIPTION
    assert document.source_format is SourceFormat.MD
    assert "Python" in document.text
    assert "NLP" in document.text
    assert "FastAPI" in document.text
    assert "**" not in document.text
    assert "https://example.com/nlp" not in document.text
    assert not document.text.startswith("#")


def test_load_pdf_cv(pdf_cv_path: Path) -> None:
    document = load_document(pdf_cv_path, kind=DocumentKind.CV)
    assert document.source_format is SourceFormat.PDF
    assert document.page_count == 1
    assert "Mykhaylo Zhyhan" in document.text
    assert "Python" in document.text


def test_load_multipage_pdf(tmp_path: Path) -> None:
    path = tmp_path / "job.pdf"
    path.write_bytes(build_simple_pdf(["AI Engineer", "Required: Python and NLP"]))
    document = load_document(path, kind=DocumentKind.JOB_DESCRIPTION)
    assert document.page_count == 2
    assert "AI Engineer" in document.text
    assert "Python and NLP" in document.text


def test_load_docx_includes_paragraphs_and_tables(docx_cv_path: Path) -> None:
    document = load_document(docx_cv_path, kind=DocumentKind.CV)
    assert document.source_format is SourceFormat.DOCX
    assert "Python and AI Engineer" in document.text
    assert "Skill" in document.text
    assert "Python" in document.text
    assert document.page_count is None


def test_load_from_bytes_for_each_format(txt_cv_path: Path) -> None:
    text_bytes = txt_cv_path.read_bytes()
    loaded = load_from_bytes(
        text_bytes,
        kind=DocumentKind.CV,
        source_format="txt",
        source_name="uploaded.txt",
    )
    assert loaded.source_path is None
    assert loaded.source_name == "uploaded.txt"
    assert "document processing pipelines" in loaded.text


def test_load_from_bytes_infers_format_from_name() -> None:
    data = build_docx("Job: AI Engineer in London.")
    document = load_from_bytes(data, kind="job_description", source_name="role.docx")
    assert document.source_format is SourceFormat.DOCX
    assert "AI Engineer" in document.text


def test_detect_format_is_case_insensitive() -> None:
    assert detect_format("CV.PDF") is SourceFormat.PDF
    assert detect_format("notes.Markdown") is SourceFormat.MD
    assert detect_format("readme.md") is SourceFormat.MD


def test_missing_file_raises_file_not_found(tmp_path: Path) -> None:
    missing = tmp_path / "absent.txt"
    with pytest.raises(FileNotFoundError, match="not found"):
        load_document(missing, kind=DocumentKind.CV)


def test_directory_raises_load_error(tmp_path: Path) -> None:
    with pytest.raises(LoadError, match="directory"):
        load_document(tmp_path, kind=DocumentKind.CV)


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "cv.xlsx"
    path.write_bytes(b"not a spreadsheet")
    with pytest.raises(UnsupportedFormatError, match="xlsx"):
        load_document(path, kind=DocumentKind.CV)


def test_unknown_kind_raises_value_error(txt_cv_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown document kind"):
        load_document(txt_cv_path, kind="cover_letter")


def test_empty_text_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("   \n\n", encoding="utf-8")
    with pytest.raises(EmptyDocumentError, match="No extractable text"):
        load_document(path, kind=DocumentKind.CV)


def test_blank_pdf_raises(tmp_path: Path) -> None:
    path = tmp_path / "blank.pdf"
    path.write_bytes(build_simple_pdf([""]))
    with pytest.raises(EmptyDocumentError):
        load_document(path, kind=DocumentKind.JOB_DESCRIPTION)


def test_encrypted_pdf_raises(tmp_path: Path) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.encrypt("secret")
    path = tmp_path / "locked.pdf"
    with path.open("wb") as handle:
        writer.write(handle)
    with pytest.raises(LoadError, match="encrypted"):
        load_document(path, kind=DocumentKind.CV)


def test_corrupt_pdf_raises() -> None:
    with pytest.raises(LoadError, match="Could not parse PDF"):
        load_from_bytes(b"%PDF-this is not valid", kind=DocumentKind.CV, source_name="broken.pdf")


def test_corrupt_docx_raises() -> None:
    with pytest.raises(LoadError, match="Could not parse DOCX"):
        load_from_bytes(b"not a zip file", kind=DocumentKind.CV, source_name="broken.docx")


def test_invalid_encoding_raises(tmp_path: Path) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes("café".encode("latin-1"))
    with pytest.raises(LoadError, match="Could not decode"):
        load_document(path, kind=DocumentKind.CV, encoding="utf-8")


def test_latin1_encoding_can_be_selected(tmp_path: Path) -> None:
    path = tmp_path / "latin1.txt"
    path.write_bytes("café".encode("latin-1"))
    document = load_document(path, kind=DocumentKind.CV, encoding="latin-1")
    assert "café" in document.text


def test_utf8_bom_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "bom.txt"
    path.write_bytes("Hello CV".encode("utf-8-sig"))
    document = load_document(path, kind=DocumentKind.CV)
    assert document.text == "Hello CV"


def test_normalise_can_be_disabled(tmp_path: Path) -> None:
    path = tmp_path / "raw.txt"
    path.write_text("experi-\nence   \n", encoding="utf-8")
    document = load_document(path, kind=DocumentKind.CV, normalise=False)
    assert document.text == "experi-\nence   \n"


def test_load_from_bytes_without_format_or_suffix_raises() -> None:
    with pytest.raises(UnsupportedFormatError):
        load_from_bytes(b"hello", kind=DocumentKind.CV, source_name="document")


def test_unknown_source_format_string_raises() -> None:
    with pytest.raises(UnsupportedFormatError, match="Unknown document format"):
        load_from_bytes(b"hello", kind=DocumentKind.CV, source_format="rtf")


def test_markdown_alias_and_without_strip(tmp_path: Path) -> None:
    path = tmp_path / "job.markdown"
    path.write_text("# Title\n\n**Python**", encoding="utf-8")
    stripped = load_document(path, kind=DocumentKind.JOB_DESCRIPTION)
    kept = load_document(path, kind=DocumentKind.JOB_DESCRIPTION, strip_markdown=False)
    assert stripped.text == "Title\n\nPython"
    assert "**Python**" in kept.text


def test_loaded_document_character_count_defaults() -> None:
    document = LoadedDocument(
        text="Hello",
        kind=DocumentKind.CV,
        source_format=SourceFormat.TXT,
        source_name="cv.txt",
    )
    assert document.character_count == 5


def test_sample_cv_text_roundtrip(txt_cv_path: Path) -> None:
    document = load_document(txt_cv_path, kind=DocumentKind.CV)
    assert SAMPLE_CV_TEXT in document.text or document.text == SAMPLE_CV_TEXT


def test_docx_skips_blank_table_rows() -> None:
    data = build_docx("Summary", table=[["Skill", "Years"], ["", ""], ["Python", "3"]])
    document = load_from_bytes(data, kind=DocumentKind.CV, source_name="cv.docx")
    assert "Summary" in document.text
    assert "Skill" in document.text
    assert "Python" in document.text


def test_pdf_with_empty_user_password_is_readable() -> None:
    writer = PdfWriter()
    writer.append(BytesIO(build_simple_pdf(["Python engineer"])))
    writer.encrypt(user_password="", owner_password="owner")
    buffer = BytesIO()
    writer.write(buffer)
    document = load_from_bytes(buffer.getvalue(), kind=DocumentKind.CV, source_name="open.pdf")
    assert "Python engineer" in document.text


def test_decrypt_failure_is_wrapped_as_load_error() -> None:
    with patch("cv_matcher.loaders.PdfReader") as reader_cls:
        reader = reader_cls.return_value
        reader.is_encrypted = True
        reader.decrypt.side_effect = ValueError("unsupported filter")
        with pytest.raises(LoadError, match="encrypted"):
            load_from_bytes(b"%PDF-fake", kind=DocumentKind.CV, source_name="locked.pdf")


def test_page_extraction_failure_is_wrapped_as_load_error() -> None:
    with patch("cv_matcher.loaders.PdfReader") as reader_cls:
        reader = reader_cls.return_value
        reader.is_encrypted = False

        class BrokenPage:
            def extract_text(self) -> str:
                raise PdfReadError("boom")

        reader.pages = [BrokenPage()]
        with pytest.raises(LoadError, match="Could not extract text"):
            load_from_bytes(b"%PDF-fake", kind=DocumentKind.CV, source_name="bad.pdf")
