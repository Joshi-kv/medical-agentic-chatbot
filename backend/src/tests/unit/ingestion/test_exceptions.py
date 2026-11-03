from ingestion.exceptions import (
    EmptyPDFError,
    EncryptedPDFError,
    InvalidPDFError,
    PDFIngestionError,
    PDFNotFoundError,
    PDFTooLargeError,
)


def test_base_pdf_ingestion_error() -> None:
    error_without_path = PDFIngestionError("Generic failure")
    assert str(error_without_path) == "Generic failure"
    assert error_without_path.path is None

    error_with_path = PDFIngestionError("Generic failure", path="/path/to/doc.pdf")
    assert str(error_with_path) == "Generic failure (file: /path/to/doc.pdf)"
    assert error_with_path.path == "/path/to/doc.pdf"


def test_pdf_not_found_error() -> None:
    error = PDFNotFoundError(path="missing.pdf")
    assert isinstance(error, PDFIngestionError)
    assert error.path == "missing.pdf"
    assert "PDF not found" in str(error)


def test_invalid_pdf_error() -> None:
    error_no_reason = InvalidPDFError(path="bad.pdf")
    assert isinstance(error_no_reason, PDFIngestionError)
    assert error_no_reason.path == "bad.pdf"
    assert str(error_no_reason) == "Invalid PDF (file: bad.pdf)"

    error_with_reason = InvalidPDFError(path="bad.pdf", reason="corrupt header")
    assert error_with_reason.path == "bad.pdf"
    assert "Invalid PDF: corrupt header" in str(error_with_reason)


def test_encrypted_pdf_error() -> None:
    error = EncryptedPDFError(path="protected.pdf")
    assert isinstance(error, PDFIngestionError)
    assert error.path == "protected.pdf"
    assert "encrypted" in str(error).lower()


def test_pdf_too_large_error() -> None:
    error = PDFTooLargeError(
        path="huge.pdf", size_bytes=25_000_000, limit_bytes=20_000_000
    )
    assert isinstance(error, PDFIngestionError)
    assert error.path == "huge.pdf"
    assert error.size_bytes == 25_000_000
    assert error.limit_bytes == 20_000_000
    assert "25,000,000" in str(error)
    assert "20,000,000" in str(error)


def test_empty_pdf_error() -> None:
    error = EmptyPDFError(path="empty.pdf")
    assert isinstance(error, PDFIngestionError)
    assert error.path == "empty.pdf"
    assert "empty" in str(error).lower() or "no extractable pages" in str(error).lower()
