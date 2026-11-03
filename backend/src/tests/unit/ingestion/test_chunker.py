from hashlib import sha256

import pytest

from ingestion.chunker import chunk_page, chunk_pages
from ingestion.schemas import DocumentChunk, ExtractedPage, SourceDocument


def make_document(**updates: object) -> SourceDocument:
    values = {
        "document_id": "document-1",
        "filename": "medical-guide.pdf",
        "size_bytes": 1_024,
        "checksum": "a" * 64,
        "source": "medical-guide.pdf",
        "title": "Medical Guide",
        "tenant_id": "tenant-1",
        "metadata": {"publisher": "Example Health"},
    }
    values.update(updates)
    return SourceDocument(**values)


def make_page(text: str, **updates: object) -> ExtractedPage:
    values = {
        "document_id": "document-1",
        "page_number": 1,
        "text": text,
        "metadata": {"section": "Dosage"},
    }
    values.update(updates)
    return ExtractedPage(**values)


def test_short_page_produces_complete_document_chunk() -> None:
    document = make_document()
    page = make_page("Aspirin may reduce platelet aggregation.")

    chunks = chunk_page(page, document)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert isinstance(chunk, DocumentChunk)
    assert chunk.chunk_id == "document-1:page-1:chunk-0"
    assert chunk.content_hash == sha256(chunk.text.encode("utf-8")).hexdigest()
    assert chunk.source == document.source
    assert chunk.title == document.title
    assert chunk.tenant_id == document.tenant_id
    assert chunk.metadata == {
        "publisher": "Example Health",
        "section": "Dosage",
    }


@pytest.mark.parametrize(
    "page",
    [
        make_page("   "),
        make_page("unreliable extracted text", requires_ocr=True),
    ],
)
def test_empty_and_ocr_pages_are_skipped(page: ExtractedPage) -> None:
    assert chunk_page(page, make_document()) == []


def test_chunking_does_not_create_overlap_only_tail() -> None:
    chunks = chunk_page(
        make_page("x" * 1_200),
        make_document(),
        chunk_size=1_200,
        chunk_overlap=200,
    )

    assert len(chunks) == 1
    assert len(chunks[0].text) == 1_200


def test_long_page_has_bounded_chunks_with_overlap() -> None:
    text = "0123456789" * 220

    chunks = chunk_page(
        make_page(text),
        make_document(),
        chunk_size=1_200,
        chunk_overlap=200,
    )

    assert [len(chunk.text) for chunk in chunks] == [1_200, 1_200]
    assert chunks[0].text[-200:] == chunks[1].text[:200]


def test_paragraph_boundary_is_preferred() -> None:
    text = "First paragraph has useful details.\n\nSecond paragraph continues here."

    chunks = chunk_page(
        make_page(text),
        make_document(),
        chunk_size=45,
        chunk_overlap=5,
    )

    assert chunks[0].text == "First paragraph has useful details."
    assert chunks[1].text.endswith("Second paragraph continues here.")


def test_chunk_pages_preserves_page_order_and_restarts_numbering() -> None:
    pages = [
        make_page("a" * 60),
        make_page("b" * 60, page_number=2),
    ]

    chunks = chunk_pages(
        pages,
        make_document(),
        chunk_size=40,
        chunk_overlap=10,
    )

    assert [chunk.page_number for chunk in chunks] == [1, 1, 2, 2]
    assert [chunk.chunk_number for chunk in chunks] == [0, 1, 0, 1]


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)],
)
def test_invalid_chunk_configuration_is_rejected(
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        chunk_page(
            make_page("text"),
            make_document(),
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def test_page_and_document_ids_must_match() -> None:
    with pytest.raises(ValueError, match="IDs must match"):
        chunk_page(
            make_page("text", document_id="different-document"),
            make_document(),
        )
