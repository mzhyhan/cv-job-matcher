"""cv-job-matcher: match a CV against a job description.

Public API (document loading):
    - :func:`load_document` and :func:`load_from_bytes` - extract text from a
      CV or job description in PDF, DOCX, TXT or Markdown.
    - :func:`normalise_text` - clean extracted text for later matching.
    - :class:`LoadedDocument`, :class:`DocumentKind`, :class:`SourceFormat`.
    - :class:`LoadError`, :class:`UnsupportedFormatError`,
      :class:`EmptyDocumentError`.

Example:
    >>> import cv_matcher
    >>> cv_matcher.__version__
    '0.1.0'
    >>> from cv_matcher import DocumentKind, load_document  # doctest: +SKIP
    >>> cv = load_document("cv.pdf", kind=DocumentKind.CV)  # doctest: +SKIP
"""

import logging

from cv_matcher.loaders import (
    SUPPORTED_EXTENSIONS,
    detect_format,
    load_document,
    load_from_bytes,
)
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
    "DocumentKind",
    "EmptyDocumentError",
    "LoadError",
    "LoadedDocument",
    "SourceFormat",
    "UnsupportedFormatError",
    "__version__",
    "detect_format",
    "load_document",
    "load_from_bytes",
    "normalise_text",
]
__version__ = "0.1.0"

# Library best practice: let the application decide how (and whether) to log.
logging.getLogger(__name__).addHandler(logging.NullHandler())
