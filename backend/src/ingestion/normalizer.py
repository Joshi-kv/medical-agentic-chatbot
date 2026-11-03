import re

from ingestion.schemas import ExtractedPage

CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
SURROUNDING_NEWLINE_WHITESPACE = re.compile(r" *\n *")
EXCESSIVE_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_page(page: ExtractedPage) -> ExtractedPage:
    """Return a copy of a page with conservatively normalized text."""
    text = page.text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARACTERS.sub("", text)
    text = HORIZONTAL_WHITESPACE.sub(" ", text)
    text = SURROUNDING_NEWLINE_WHITESPACE.sub("\n", text)
    text = EXCESSIVE_BLANK_LINES.sub("\n\n", text)

    return page.model_copy(update={"text": text.strip()})


def normalize_pages(pages: list[ExtractedPage]) -> list[ExtractedPage]:
    """Normalize pages while preserving their order."""
    return [normalize_page(page) for page in pages]
