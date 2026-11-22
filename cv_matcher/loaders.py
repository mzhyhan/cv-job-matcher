"""Load CVs and job descriptions from PDF, DOCX, TXT and Markdown files.

The main entry points are :func:`load_document` (from a path) and
:func:`load_from_bytes` (from in-memory content, as a web upload will later
need). Both return a :class:`~cv_matcher.models.LoadedDocument`.

Example:
    ::

        from cv_matcher import DocumentKind, load_document

        cv = load_document("cv.pdf", kind=DocumentKind.CV)
        print(cv.text)
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile

from docx import Document as DocxDocument
from pypdf import PdfReader
from pypdf.errors import FileNotDecryptedError, PdfReadError

from cv_matcher.models import (
    DocumentKind,
    EmptyDocumentError,
    LoadedDocument,
    LoadError,
    SourceFormat,
    UnsupportedFormatError,
)
from cv_matcher.text import normalise_text

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "detect_format",
    "load_document",
    "load_from_bytes",
]

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: dict[str, SourceFormat] = {
    ".pdf": SourceFormat.PDF,
    ".docx": SourceFormat.DOCX,
    ".txt": SourceFormat.TXT,
    ".md": SourceFormat.MD,
    ".markdown": SourceFormat.MD,
}
"""Mapping of lower-cased file suffixes to a :class:`SourceFormat`."""


def detect_format(path: str | Path) -> SourceFormat:
    """Return the source format implied by ``path``'s suffix.

    Args:
        path: A file path or name. Only the suffix is inspected.

    Returns:
        The matching :class:`SourceFormat`.

    Raises:
        UnsupportedFormatError: If the suffix is missing or not supported.
    """
    suffix = Path(path).suffix.lower()
    try:
        return SUPPORTED_EXTENSIONS[suffix]
    except KeyError as exc:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        label = Path(path).name or str(path)
        raise UnsupportedFormatError(
            f"Unsupported document format for {label!r}. Expected one of: {supported}."
        ) from exc


def load_document(
    path: str | Path,
    kind: DocumentKind | str,
    *,
    encoding: str = "utf-8",
    normalise: bool = True,
    strip_markdown: bool | None = None,
) -> LoadedDocument:
    """Extract text from a CV or job description on disk.

    Args:
        path: Path to a PDF, DOCX, TXT or Markdown file.
        kind: ``"cv"`` or ``"job_description"``, or a :class:`DocumentKind`.
        encoding: Text encoding used for TXT and Markdown files. ``"utf-8"``
            also accepts a leading byte-order mark.
        normalise: If True (the default), clean the extracted text with
            :func:`~cv_matcher.text.normalise_text`.
        strip_markdown: Whether to drop Markdown syntax during normalisation.
            ``None`` (the default) enables this only for Markdown files.
            Ignored when ``normalise`` is False.

    Returns:
        A :class:`LoadedDocument` with the extracted text and source metadata.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        LoadError: If ``path`` is not a regular file, cannot be parsed, or
            yields no text.
        UnsupportedFormatError: If the file suffix is not supported.
        EmptyDocumentError: If the file contains no extractable text.
    """
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")
    if not file_path.is_file():
        raise LoadError(f"Expected a file, got a directory: {file_path}")

    source_format = detect_format(file_path)
    data = file_path.read_bytes()
    document = load_from_bytes(
        data,
        kind,
        source_format=source_format,
        source_name=file_path.name,
        encoding=encoding,
        normalise=normalise,
        strip_markdown=strip_markdown,
    )
    return LoadedDocument(
        text=document.text,
        kind=document.kind,
        source_format=document.source_format,
        source_name=document.source_name,
        source_path=file_path.resolve(),
        page_count=document.page_count,
        character_count=document.character_count,
        metadata=document.metadata,
    )


def load_from_bytes(
    data: bytes,
    kind: DocumentKind | str,
    *,
    source_format: SourceFormat | str | None = None,
    source_name: str = "document",
    encoding: str = "utf-8",
    normalise: bool = True,
    strip_markdown: bool | None = None,
) -> LoadedDocument:
    """Extract text from in-memory document bytes.

    Args:
        data: The raw file contents.
        kind: ``"cv"`` or ``"job_description"``, or a :class:`DocumentKind`.
        source_format: Format of ``data``. If omitted, it is inferred from
            ``source_name``.
        source_name: Label used in error messages and, when ``source_format``
            is omitted, for format detection.
        encoding: Text encoding used for TXT and Markdown bytes.
        normalise: If True (the default), clean the extracted text.
        strip_markdown: Whether to drop Markdown syntax during normalisation.
            ``None`` (the default) enables this only for Markdown input.
            Ignored when ``normalise`` is False.

    Returns:
        A :class:`LoadedDocument` with the extracted text and source metadata.

    Raises:
        UnsupportedFormatError: If the format cannot be determined or is
            not supported.
        EmptyDocumentError: If the bytes contain no extractable text.
        LoadError: If the bytes cannot be parsed as the given format.
    """
    document_kind = _as_kind(kind)
    fmt = _as_format(source_format, source_name)
    if strip_markdown is None:
        strip_markdown = fmt is SourceFormat.MD

    raw_text, page_count = _extract(data, fmt, encoding=encoding, source_name=source_name)
    text = normalise_text(raw_text, strip_markdown=strip_markdown) if normalise else raw_text
    if not text.strip():
        raise EmptyDocumentError(f"No extractable text in {source_name!r}.")

    logger.debug(
        "Loaded %s %s from %s (%s characters)",
        document_kind.value,
        fmt.value,
        source_name,
        len(text),
    )
    return LoadedDocument(
        text=text,
        kind=document_kind,
        source_format=fmt,
        source_name=source_name,
        source_path=None,
        page_count=page_count,
        character_count=len(text),
    )


def _as_kind(kind: DocumentKind | str) -> DocumentKind:
    """Coerce a string or enum value to :class:`DocumentKind`."""
    if isinstance(kind, DocumentKind):
        return kind
    try:
        return DocumentKind(kind)
    except ValueError as exc:
        expected = ", ".join(item.value for item in DocumentKind)
        raise ValueError(f"Unknown document kind {kind!r}. Expected one of: {expected}.") from exc


def _as_format(source_format: SourceFormat | str | None, source_name: str) -> SourceFormat:
    """Resolve a source format from an explicit value or a file name."""
    if source_format is None:
        return detect_format(source_name)
    if isinstance(source_format, SourceFormat):
        return source_format
    try:
        return SourceFormat(source_format)
    except ValueError as exc:
        expected = ", ".join(item.value for item in SourceFormat)
        raise UnsupportedFormatError(
            f"Unknown document format {source_format!r}. Expected one of: {expected}."
        ) from exc


def _extract(
    data: bytes,
    source_format: SourceFormat,
    *,
    encoding: str,
    source_name: str,
) -> tuple[str, int | None]:
    """Return ``(raw_text, page_count)`` for a single format."""
    if source_format is SourceFormat.PDF:
        return _extract_pdf(data, source_name=source_name)
    if source_format is SourceFormat.DOCX:
        return _extract_docx(data, source_name=source_name), None
    return _decode_text(data, encoding=encoding, source_name=source_name), None


def _extract_pdf(data: bytes, *, source_name: str) -> tuple[str, int]:
    """Extract text and the page count from a PDF."""
    try:
        reader = PdfReader(BytesIO(data))
    except PdfReadError as exc:
        raise LoadError(f"Could not parse PDF {source_name!r}: {exc}") from exc

    if reader.is_encrypted:
        try:
            # Some writers set an empty user password; try that before failing.
            unlocked = reader.decrypt("")
        except (
            PdfReadError,
            FileNotDecryptedError,
            NotImplementedError,
            OSError,
            ValueError,
        ) as exc:
            raise LoadError(
                f"PDF {source_name!r} is encrypted. Decrypt it before loading."
            ) from exc
        if int(unlocked) == 0:
            raise LoadError(f"PDF {source_name!r} is encrypted. Decrypt it before loading.")

    try:
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except (PdfReadError, FileNotDecryptedError, OSError, ValueError) as exc:
        raise LoadError(f"Could not extract text from PDF {source_name!r}: {exc}") from exc

    return "\n\n".join(part for part in pages if part), len(reader.pages)


def _extract_docx(data: bytes, *, source_name: str) -> str:
    """Extract paragraphs and table cells from a Word document."""
    try:
        document = DocxDocument(BytesIO(data))
    except (BadZipFile, KeyError, ValueError, OSError) as exc:
        raise LoadError(f"Could not parse DOCX {source_name!r}: {exc}") from exc

    parts: list[str] = []
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            parts.append(paragraph.text)
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append("\t".join(cells))
    return "\n".join(parts)


def _decode_text(data: bytes, *, encoding: str, source_name: str) -> str:
    """Decode TXT or Markdown bytes, accepting a UTF-8 BOM when encoding is UTF-8."""
    codec = "utf-8-sig" if encoding.lower().replace("_", "-") in {"utf-8", "utf8"} else encoding
    try:
        return data.decode(codec)
    except UnicodeDecodeError as exc:
        raise LoadError(
            f"Could not decode {source_name!r} as {encoding}. "
            "Pass encoding=... to load_document or load_from_bytes."
        ) from exc
