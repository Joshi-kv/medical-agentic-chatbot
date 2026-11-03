from hashlib import sha256

from ingestion.schemas import DocumentChunk, ExtractedPage, SourceDocument

DEFAULT_CHUNK_SIZE = 1_200
DEFAULT_CHUNK_OVERLAP = 200

PREFERRED_BOUNDARIES = ("\n\n", "\n", ". ", "? ", "! ", " ")


def chunk_page(
    page: ExtractedPage,
    document: SourceDocument,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Split one page into deterministic, citation-safe document chunks."""
    _validate_chunk_configuration(chunk_size, chunk_overlap)

    if page.document_id != document.document_id:
        raise ValueError("page and document IDs must match")

    if page.requires_ocr or not page.text.strip():
        return []

    text_segments = _split_text(page.text.strip(), chunk_size, chunk_overlap)
    metadata = {**document.metadata, **page.metadata}

    return [
        DocumentChunk(
            chunk_id=(
                f"{document.document_id}:page-{page.page_number}:chunk-{chunk_number}"
            ),
            document_id=document.document_id,
            page_number=page.page_number,
            chunk_number=chunk_number,
            text=text,
            content_hash=sha256(text.encode("utf-8")).hexdigest(),
            source=document.source,
            title=document.title,
            tenant_id=document.tenant_id,
            metadata=metadata.copy(),
        )
        for chunk_number, text in enumerate(text_segments)
    ]


def chunk_pages(
    pages: list[ExtractedPage],
    document: SourceDocument,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Chunk pages in order, restarting chunk numbering for each page."""
    _validate_chunk_configuration(chunk_size, chunk_overlap)

    chunks: list[DocumentChunk] = []
    for page in pages:
        chunks.extend(
            chunk_page(
                page,
                document,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
        )
    return chunks


def _split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0

    while start < len(text):
        maximum_end = min(start + chunk_size, len(text))
        end = maximum_end

        if maximum_end < len(text):
            end = _find_preferred_boundary(text, start, maximum_end, chunk_size)

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = max(end - chunk_overlap, start + 1)

    return chunks


def _find_preferred_boundary(
    text: str,
    start: int,
    maximum_end: int,
    chunk_size: int,
) -> int:
    minimum_end = start + max(1, chunk_size // 2)

    for boundary in PREFERRED_BOUNDARIES:
        position = text.rfind(boundary, minimum_end, maximum_end)
        if position != -1:
            return position + len(boundary)

    return maximum_end


def _validate_chunk_configuration(chunk_size: int, chunk_overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and less than chunk_size")
