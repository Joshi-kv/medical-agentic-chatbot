class PDFIngestionError(Exception):
    """Base exception for PDF ingestion failures."""

    def __init__(self, message: str, path: str | None = None) -> None:
        self.path = path
        detail = f"{message} (file: {path})" if path else message
        super().__init__(detail)


class PDFNotFoundError(PDFIngestionError):
    """Raised when the requested PDF does not exist."""

    def __init__(self, path: str) -> None:
        super().__init__("PDF not found", path=path)


class InvalidPDFError(PDFIngestionError):
    """Raised when a file is not a valid PDF."""

    def __init__(self, path: str, reason: str | None = None) -> None:
        message = f"Invalid PDF: {reason}" if reason else "Invalid PDF"
        super().__init__(message, path=path)


class EncryptedPDFError(PDFIngestionError):
    """Raised when a PDF requires a password."""

    def __init__(self, path: str) -> None:
        super().__init__("PDF is encrypted and requires a password", path=path)


class PDFTooLargeError(PDFIngestionError):
    """Raised when a PDF exceeds the configured size limit."""

    def __init__(self, path: str, size_bytes: int, limit_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes
        message = (
            f"PDF size {size_bytes:,} bytes exceeds limit of {limit_bytes:,} bytes"
        )
        super().__init__(message, path=path)


class EmptyPDFError(PDFIngestionError):
    """Raised when a PDF contains no extractable pages."""

    def __init__(self, path: str) -> None:
        super().__init__("PDF contains no extractable pages", path=path)
