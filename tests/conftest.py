"""Shared fixtures: sample documents are generated in a temporary directory."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from docx import Document as DocxDocument

SAMPLE_CV_TEXT = (
    "Mykhaylo Zhyhan\n"
    "Python and AI Engineer\n\n"
    "Experience\n"
    "Built document processing pipelines in Python."
)
"""Plain text used in generated CV fixtures."""

SAMPLE_JOB_TEXT = (
    "AI Engineer\n\n"
    "We are looking for a Python engineer with experience in NLP "
    "and document processing."
)
"""Plain text used in generated job-description fixtures."""


def build_simple_pdf(pages: list[str]) -> bytes:
    """Build a minimal PDF 1.4 document with one Helvetica line per page.

    The content stream uses a standard Type1 font so pypdf can extract the
    literal text without needing a full typesetting library.
    """
    n = len(pages)
    page_objs = [4 + 2 * i for i in range(n)]
    content_objs = [5 + 2 * i for i in range(n)]

    parts: dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        3: b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    }
    kids = " ".join(f"{obj} 0 R" for obj in page_objs)
    parts[2] = f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode()

    for i, text in enumerate(pages):
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1", "replace")
        parts[content_objs[i]] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )
        parts[page_objs[i]] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {content_objs[i]} 0 R "
            f"/Resources << /Font << /F1 3 0 R >> >> >>"
        ).encode()

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = {0: 0}
    max_obj = max(parts)
    for num in range(1, max_obj + 1):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode() + parts[num] + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {max_obj + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for num in range(1, max_obj + 1):
        out += f"{offsets[num]:010d} 00000 n \n".encode()
    out += (
        f"trailer << /Size {max_obj + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


def build_docx(text: str, *, table: list[list[str]] | None = None) -> bytes:
    """Return the bytes of a short .docx document."""
    document = DocxDocument()
    for paragraph in text.split("\n"):
        document.add_paragraph(paragraph)
    if table:
        grid = document.add_table(rows=len(table), cols=len(table[0]))
        for row_idx, row in enumerate(table):
            for col_idx, value in enumerate(row):
                grid.rows[row_idx].cells[col_idx].text = value
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()


@pytest.fixture
def sample_cv_text() -> str:
    return SAMPLE_CV_TEXT


@pytest.fixture
def sample_job_text() -> str:
    return SAMPLE_JOB_TEXT


@pytest.fixture
def txt_cv_path(tmp_path: Path) -> Path:
    path = tmp_path / "cv.txt"
    path.write_text(SAMPLE_CV_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def md_job_path(tmp_path: Path) -> Path:
    markdown = (
        "# AI Engineer\n\n"
        "We are looking for a **Python** engineer with experience in "
        "[NLP](https://example.com/nlp) and document processing.\n\n"
        "- FastAPI\n"
        "- embeddings\n"
    )
    path = tmp_path / "job.md"
    path.write_text(markdown, encoding="utf-8")
    return path


@pytest.fixture
def pdf_cv_path(tmp_path: Path) -> Path:
    path = tmp_path / "cv.pdf"
    path.write_bytes(build_simple_pdf(["Mykhaylo Zhyhan Python and AI Engineer"]))
    return path


@pytest.fixture
def docx_cv_path(tmp_path: Path) -> Path:
    path = tmp_path / "cv.docx"
    path.write_bytes(
        build_docx(
            SAMPLE_CV_TEXT,
            table=[["Skill", "Years"], ["Python", "3"]],
        )
    )
    return path
