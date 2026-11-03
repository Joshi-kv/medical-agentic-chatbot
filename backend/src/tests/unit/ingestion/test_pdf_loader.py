import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pymupdf
import pytest

from ingestion.exceptions import (
    EmptyPDFError,
    EncryptedPDFError,
    InvalidPDFError,
    PDFNotFoundError,
    PDFTooLargeError,
)
from ingestion.pdf_loader import DEFAULT_OCR_TEXT_THRESHOLD, load_pdf
from ingestion.schemas import SourceDocument


@pytest.fixture
def sample_source_document() -> SourceDocument:
    return SourceDocument(
        document_id="doc-123",
        filename="test.pdf",
        source="test-suite",
        tenant_id="tenant-1",
        size_bytes=1024,
        checksum="a" * 64,
    )


def create_pdf(
    file_path: Path,
    pages_text: list[str],
    password: str | None = None,
    with_image: bool = False,
) -> Path:
    doc = pymupdf.open()
    for text in pages_text:
        page = doc.new_page(width=595, height=842)
        if text:
            page.insert_text((50, 72), text)

        if with_image:
            # 10x10 pixmap
            pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
            pix.clear_with(255)
            page.insert_image(pymupdf.Rect(100, 100, 200, 200), pixmap=pix)

    if password:
        doc.save(
            file_path,
            encryption=pymupdf.PDF_ENCRYPT_AES_256,
            user_pw=password,
            owner_pw=password + "_owner",
        )
    else:
        doc.save(file_path)
    doc.close()
    return file_path


def test_load_pdf_raises_not_found(
    sample_source_document: SourceDocument, tmp_path: Path
) -> None:
    non_existent = tmp_path / "does_not_exist.pdf"
    with pytest.raises(PDFNotFoundError) as exc_info:
        load_pdf(document=sample_source_document, file_path=non_existent)

    assert str(non_existent) in exc_info.value.path


def test_load_pdf_raises_empty_file(
    sample_source_document: SourceDocument, tmp_path: Path
) -> None:
    empty_file = tmp_path / "zero_bytes.pdf"
    empty_file.write_bytes(b"")

    with pytest.raises(InvalidPDFError) as exc_info:
        load_pdf(document=sample_source_document, file_path=empty_file)

    assert "empty" in str(exc_info.value).lower()


def test_load_pdf_raises_invalid_signature(
    sample_source_document: SourceDocument, tmp_path: Path
) -> None:
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.write_bytes(b"NOT_A_PDF_CONTENT")

    with pytest.raises(InvalidPDFError) as exc_info:
        load_pdf(document=sample_source_document, file_path=fake_pdf)

    assert "signature" in str(exc_info.value).lower()


def test_load_pdf_raises_too_large(
    sample_source_document: SourceDocument, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "large.pdf"
    create_pdf(pdf_path, ["Short text"])

    file_size = pdf_path.stat().st_size
    max_size = file_size - 1

    with pytest.raises(PDFTooLargeError) as exc_info:
        load_pdf(
            document=sample_source_document,
            file_path=pdf_path,
            max_file_size=max_size,
        )

    assert exc_info.value.size_bytes == file_size
    assert exc_info.value.limit_bytes == max_size


def test_load_pdf_raises_encrypted(
    sample_source_document: SourceDocument, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "encrypted.pdf"
    create_pdf(pdf_path, ["Classified information"], password="secretpassword")

    with pytest.raises(EncryptedPDFError):
        load_pdf(document=sample_source_document, file_path=pdf_path)


def test_load_pdf_raises_empty_pdf(
    sample_source_document: SourceDocument, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "empty_doc.pdf"
    create_pdf(pdf_path, ["Placeholder page"])

    mock_doc = MagicMock()
    mock_doc.needs_pass = False
    mock_doc.page_count = 0
    mock_doc.__enter__.return_value = mock_doc

    with patch("pymupdf.open", return_value=mock_doc):
        with pytest.raises(EmptyPDFError):
            load_pdf(document=sample_source_document, file_path=pdf_path)


def test_load_pdf_success_multi_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "medical_doc.pdf"
    page_texts = [
        "Patient presents with fever and fatigue.",
        "Differential diagnosis includes influenza and COVID-19.",
    ]
    create_pdf(pdf_path, page_texts)

    raw_bytes = pdf_path.read_bytes()
    doc = SourceDocument(
        document_id="doc-456",
        filename=pdf_path.name,
        source="clinic-records",
        tenant_id="hospital-a",
        size_bytes=len(raw_bytes),
        checksum=hashlib.sha256(raw_bytes).hexdigest(),
    )

    pages = load_pdf(document=doc, file_path=pdf_path)

    assert len(pages) == 2

    assert pages[0].document_id == "doc-456"
    assert pages[0].page_number == 1
    assert "Patient presents with fever" in pages[0].text
    assert not pages[0].requires_ocr
    assert pages[0].metadata["filename"] == pdf_path.name
    assert pages[0].metadata["source"] == "clinic-records"
    assert pages[0].metadata["tenant_id"] == "hospital-a"
    assert pages[0].metadata["page_width"] == 595.0
    assert pages[0].metadata["page_height"] == 842.0

    assert pages[1].document_id == "doc-456"
    assert pages[1].page_number == 2
    assert "Differential diagnosis" in pages[1].text
    assert not pages[1].requires_ocr


def test_load_pdf_ocr_flag_triggered_for_scanned_page(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scanned_page.pdf"
    create_pdf(pdf_path, ["Hi"], with_image=True)

    raw_bytes = pdf_path.read_bytes()
    doc = SourceDocument(
        document_id="doc-ocr",
        filename=pdf_path.name,
        source="scanner",
        size_bytes=len(raw_bytes),
        checksum=hashlib.sha256(raw_bytes).hexdigest(),
    )

    pages = load_pdf(
        document=doc,
        file_path=pdf_path,
        ocr_text_threshold=DEFAULT_OCR_TEXT_THRESHOLD,
    )

    assert len(pages) == 1
    assert pages[0].requires_ocr is True
    assert pages[0].metadata["image_count"] >= 1


def test_load_pdf_corrupt_file_raises_invalid_pdf(
    sample_source_document: SourceDocument, tmp_path: Path
) -> None:
    corrupt_pdf = tmp_path / "corrupt.pdf"
    corrupt_pdf.write_bytes(b"%PDF-1.7\nCorrupted garbage data")

    with pytest.raises(InvalidPDFError):
        load_pdf(document=sample_source_document, file_path=corrupt_pdf)
