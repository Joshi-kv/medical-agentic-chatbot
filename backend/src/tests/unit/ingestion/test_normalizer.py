from ingestion.normalizer import normalize_page, normalize_pages
from ingestion.schemas import ExtractedPage


def make_page(text: str, **kwargs: object) -> ExtractedPage:
    return ExtractedPage(
        document_id="document-1",
        page_number=1,
        text=text,
        **kwargs,
    )


def test_normalize_page_cleans_control_characters_and_whitespace() -> None:
    page = make_page("  First\t line\x00\r\n\r\n\r\n\r\n Second\x07  line  ")

    normalized = normalize_page(page)

    assert normalized.text == "First line\n\nSecond line"


def test_normalize_page_preserves_medical_notation_and_lists() -> None:
    text = "Dosage: 5 mg/kg\n• Monitor BP ≥ 90 mmHg\n- Do not exceed 500 mg"

    normalized = normalize_page(make_page(text))

    assert normalized.text == text


def test_normalize_page_does_not_mutate_original_page_or_metadata() -> None:
    page = make_page("  original  ", metadata={"source": "guide.pdf"})

    normalized = normalize_page(page)

    assert page.text == "  original  "
    assert normalized.text == "original"
    assert normalized.metadata == page.metadata


def test_normalize_page_preserves_empty_ocr_candidate() -> None:
    page = make_page("", requires_ocr=True)

    normalized = normalize_page(page)

    assert normalized.text == ""
    assert normalized.requires_ocr is True


def test_normalize_pages_preserves_page_order() -> None:
    pages = [
        make_page("  page one  "),
        make_page("  page two  ").model_copy(update={"page_number": 2}),
    ]

    normalized = normalize_pages(pages)

    assert [page.page_number for page in normalized] == [1, 2]
    assert [page.text for page in normalized] == ["page one", "page two"]
