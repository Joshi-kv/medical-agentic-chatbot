from pathlib import Path

import pymupdf

from ingestion.exceptions import (
    EmptyPDFError,
    EncryptedPDFError,
    InvalidPDFError,
    PDFNotFoundError,
    PDFTooLargeError,
)
from ingestion.schemas import ExtractedPage, SourceDocument

DEFAULT_MAX_FILE_SIZE = 20 * 1024 * 1024
DEFAULT_OCR_TEXT_THRESHOLD = 20


def load_pdf(
    document: SourceDocument,
    file_path: Path,
    max_file_size: int = DEFAULT_MAX_FILE_SIZE,
    ocr_text_threshold: int = DEFAULT_OCR_TEXT_THRESHOLD,
) -> list[ExtractedPage]:
    """Load a PDF file and extract its text content, page by page.
    Args:
        document: SourceDocument object
        file_path: Path to the PDF file
        max_file_size: Maximum file size in bytes
        ocr_text_threshold: Threshold for OCR text extraction
    """
    path = Path(file_path)

    _validate_file(path, max_file_size=max_file_size)

    try:
        with pymupdf.open(path) as pdf:
            if pdf.needs_pass:
                raise EncryptedPDFError(path=str(path))

            if pdf.page_count == 0:
                raise EmptyPDFError(path=str(path))

            return [
                _extract_page(
                    document=document,
                    page=page,
                    page_number=page_index + 1,
                    ocr_text_threshold=ocr_text_threshold,
                )
                for page_index, page in enumerate(pdf)
            ]
    except EncryptedPDFError:
        raise
    except EmptyPDFError:
        raise
    except (pymupdf.FileDataError, RuntimeError) as error:
        raise InvalidPDFError(path=str(path), reason=str(error)) from error
    except Exception as error:
        raise InvalidPDFError(path=str(path), reason=str(error)) from error


def _validate_file(
    file_path: Path,
    max_file_size: int,
) -> None:
    if not file_path.is_file():
        raise PDFNotFoundError(path=str(file_path))

    file_size = file_path.stat().st_size

    if file_size == 0:
        raise InvalidPDFError(path=str(file_path), reason="file is empty")

    if file_size > max_file_size:
        raise PDFTooLargeError(
            path=str(file_path),
            size_bytes=file_size,
            limit_bytes=max_file_size,
        )

    with file_path.open("rb") as pdf_file:
        signature = pdf_file.read(5)

    if signature != b"%PDF-":
        raise InvalidPDFError(path=str(file_path), reason="missing PDF signature")


def _extract_page(
    document: SourceDocument,
    page: pymupdf.Page,
    page_number: int,
    ocr_text_threshold: int,
) -> ExtractedPage:
    text = page.get_text("text", sort=True).strip()

    image_count = len(page.get_images(full=True))
    meaningful_character_count = len(
        "".join(character for character in text if not character.isspace())
    )

    requires_ocr = image_count > 0 and meaningful_character_count < ocr_text_threshold

    return ExtractedPage(
        document_id=document.document_id,
        page_number=page_number,
        text=text,
        requires_ocr=requires_ocr,
        metadata={
            "filename": document.filename,
            "source": document.source,
            "tenant_id": document.tenant_id,
            "page_width": float(page.rect.width),
            "page_height": float(page.rect.height),
            "page_rotation": page.rotation,
            "image_count": image_count,
        },
    )


if __name__ == "__main__":
    import hashlib

    pdf_path = Path(
        "/home/joshi/Desktop/personal-projects/medical-agentic-chatbot/backend/data/Medical_Book.pdf"
    )
    pdf_bytes = pdf_path.read_bytes()

    source_document = SourceDocument(
        document_id="123",
        filename=pdf_path.name,
        source="test",
        tenant_id="123",
        size_bytes=len(pdf_bytes),
        checksum=hashlib.sha256(pdf_bytes).hexdigest(),
    )

    pages = load_pdf(document=source_document, file_path=pdf_path)

    print(pages)
