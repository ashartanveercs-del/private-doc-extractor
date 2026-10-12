"""Exception types.

Every message is written to be shown to a user as-is: no API keys, no document text and no full
URLs (signed links carry tokens in their query string). Each class carries the CLI exit code it
maps to.
"""

from __future__ import annotations


class PrivdocError(Exception):
    """Base class. Exit code 3: the document could not be read."""

    exit_code = 3


class UsageError(PrivdocError):
    """Bad arguments, configuration or schema. Exit code 2."""

    exit_code = 2


class SchemaError(UsageError):
    """The JSON schema uses something this tool cannot enforce strictly."""


class PrivacyViolation(PrivdocError):
    """Refused by the privacy tier before anything was sent. Exit code 2."""

    exit_code = 2


class EgressBlocked(PrivacyViolation):
    """A network send to a destination the tier does not allow."""


class UnsafeURL(PrivdocError):
    """A link was refused by the SSRF guard. Exit code 2."""

    exit_code = 2


class FetchError(PrivdocError):
    """A safe link could not be downloaded (status, size, type, timeout)."""


class DocumentError(PrivdocError):
    """The input is not a document this tool can open."""


class EngineError(PrivdocError):
    """One OCR/LLM call failed. The pipeline records it against the page and moves on."""


class DatasetError(PrivdocError):
    """A benchmark dataset is missing, unsafe to unpack, or in an unexpected layout."""
