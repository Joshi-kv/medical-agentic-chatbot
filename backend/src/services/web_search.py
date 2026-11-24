from hashlib import sha256
from urllib.parse import urlparse

from anyio import to_thread
from tavily import TavilyClient

from services.evidence import Evidence, EvidencePackage

MAX_CONTENT_LENGTH = 4000


class WebSearchError(Exception):
    pass


class WebSearchService:
    """Collect evidence from an allowlist of trusted medical websites."""

    def __init__(
        self,
        client: TavilyClient,
        trusted_domains: list[str],
        max_results: int = 3,
        search_depth: str = "basic",
    ) -> None:
        if not trusted_domains:
            raise ValueError("At least one trusted medical domain is required.")

        if max_results < 1 or max_results > 10:
            raise ValueError("max_results must be between 1 and 10.")

        if search_depth not in {
            "basic",
            "advanced",
            "fast",
            "ultra-fast",
        }:
            raise ValueError("Unsupported Tavily search depth.")

        self._client = client
        self._trusted_domains = self._normalize_domains(trusted_domains)
        self._max_results = max_results
        self._search_depth = search_depth

    async def collect_evidence(
        self,
        query: str,
    ) -> EvidencePackage:
        normalized_query = query.strip()

        if not normalized_query:
            raise WebSearchError("empty query is not allowed")

        try:
            response = await to_thread.run_sync(
                lambda: self._client.search(
                    query=f"{normalized_query} medical patient information",
                    topic="general",
                    search_depth=self._search_depth,
                    max_results=self._max_results,
                    include_domains=self._trusted_domains,
                    include_answer=False,
                    include_raw_content=False,
                )
            )
        except Exception as error:
            raise WebSearchError(f"Failed to collect evidence: {error}") from error

        raw_results = response.get("results", [])

        if not isinstance(raw_results, list):
            raise WebSearchError("Unexpected response format from Tavily")

        evidence: list[Evidence] = []

        for result in raw_results:
            item = self._to_evidence(result=result, position=len(evidence) + 1)

            if item is not None:
                evidence.append(item)

            if len(evidence) >= self._max_results:
                break

        if not evidence:
            raise WebSearchError("No relevant evidence found in the search results")

        return EvidencePackage(
            query=normalized_query,
            items=evidence,
            sufficient=True,
        )

    def _to_evidence(
        self,
        result: object,
        position: int,
    ) -> Evidence | None:
        if not isinstance(result, dict):
            return None

        url = result.get("url")
        title = result.get("title")
        content = result.get("content")
        score = result.get("score", 0)

        if not isinstance(url, str):
            return None

        if not self._is_trusted_url(url):
            return None

        if not isinstance(content, str):
            return None

        normalized_content = content.strip()

        if len(normalized_content) < 80:
            return None

        normalized_title = (
            title.strip() if isinstance(title, str) and title.strip() else url
        )

        try:
            normalized_score = float(score)
        except (TypeError, ValueError):
            normalized_score = 0.0

        source_hash = sha256(url.encode("utf-8")).hexdigest()[:20]

        return Evidence(
            citation_id=f"C{position}",
            chunk_id=f"web-{source_hash}",
            text=normalized_content[:MAX_CONTENT_LENGTH],
            document_id="web",
            page_number=0,
            source=url,
            score=normalized_score,
            title=normalized_title,
        )

    def _is_trusted_url(
        self,
        url: str,
    ) -> bool:
        parsed_url = urlparse(url)

        if parsed_url.scheme != "https":
            return False

        hostname = parsed_url.hostname

        if hostname is None:
            return False

        normalized_hostname = hostname.lower().rstrip(".")

        return any(
            normalized_hostname == domain or normalized_hostname.endswith(f".{domain}")
            for domain in self._trusted_domains
        )

    @staticmethod
    def _normalize_domains(
        domains: list[str],
    ) -> list[str]:
        normalized: list[str] = []

        for domain in domains:
            clean_domain = domain.strip().lower().rstrip(".")

            if not clean_domain:
                continue

            # Configuration should contain hostnames, not complete URLs.
            if "://" in clean_domain or "/" in clean_domain:
                raise ValueError(f"Invalid trusted domain: {domain}")

            if clean_domain not in normalized:
                normalized.append(clean_domain)

        if not normalized:
            raise ValueError("At least one valid trusted domain is required.")

        return normalized
