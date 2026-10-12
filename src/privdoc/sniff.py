"""Identify a file by its bytes, never by its name or declared content type."""

from __future__ import annotations

IMAGE_KINDS = frozenset({"png", "jpeg", "tiff", "webp", "gif", "bmp"})
DOCUMENT_KINDS = frozenset({"pdf"}) | IMAGE_KINDS

MEDIA_TYPES: dict[str, frozenset[str]] = {
    "pdf": frozenset({"application/pdf", "application/x-pdf"}),
    "png": frozenset({"image/png"}),
    "jpeg": frozenset({"image/jpeg", "image/jpg", "image/pjpeg"}),
    "tiff": frozenset({"image/tiff", "image/tiff-fx"}),
    "webp": frozenset({"image/webp"}),
    "gif": frozenset({"image/gif"}),
    "bmp": frozenset({"image/bmp", "image/x-ms-bmp"}),
    "zip": frozenset({"application/zip", "application/x-zip-compressed"}),
}


def sniff(data: bytes) -> str | None:
    """Return the file kind ("pdf", "png", ...) or None when it is not one we know."""
    if b"%PDF-" in data[:1024]:  # the PDF spec tolerates junk before the header
        return "pdf"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
        return "tiff"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if data[:2] == b"BM" and len(data) >= 18 and data[14] in (12, 40, 52, 56, 64, 108, 124):
        return "bmp"  # byte 14 is the DIB header size, which rules out text starting "BM"
    if data[:4] == b"PK\x03\x04":
        return "zip"
    return None
