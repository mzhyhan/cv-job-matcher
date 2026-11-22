"""cv-job-matcher: match a CV against a job description.

The public API will grow as the project takes shape. See the README for the
current status.

Example:
    >>> import cv_matcher
    >>> cv_matcher.__version__
    '0.1.0'
"""

import logging

__all__ = [
    "__version__",
]
__version__ = "0.1.0"

# Library best practice: let the application decide how (and whether) to log.
logging.getLogger(__name__).addHandler(logging.NullHandler())
