from dataclasses import dataclass
from typing import Any
from uuid import UUID

from anyio import to_thread
from langchain_core.embeddings import Embeddings
from pinecone import Index


class RetrievalServiceError(Exception):
    pass


class InvalidRetrievalQuery(RetrievalServiceError):
    pass


@dataclass(frozen=True)
class RetrievalMatch:
    chunk_id: str
    score: float
    text: str
    document_id: str
    page_number: int
    chunk_number: int
    source: str
    title: str | None = None
    content_hash: str | None = None
    section: str | None = None


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    matches: list[RetrievalMatch]


class RetrievalService:
    def __init__(self, embeddings: Embeddings, index: Index, namespace: str):
        self.embeddings = embeddings
        self.index = index
        self.namespace = namespace

    async def retrieve(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 5,
        document_id: UUID | None = None,
    ) -> RetrievalResult:
        normalized_query = query.strip()
        if not normalized_query:
            raise InvalidRetrievalQuery("Query cannot be empty.")

        if top_k < 1 or top_k > 20:
            raise InvalidRetrievalQuery("top_k must be between 1 and 20.")

        search_filter: dict[str, Any] = {
            "tenant_id": {"$eq": tenant_id},
        }

        if document_id:
            search_filter["document_id"] = {"$eq": document_id}

        try:
            embedding = await to_thread.run_sync(
                self.embeddings.embed_query,
                normalized_query,
            )

            query_result = await to_thread.run_sync(
                lambda: self.index.query(
                    vector=embedding,
                    top_k=top_k,
                    include_metadata=True,
                    namespace=self.namespace,
                    filter=search_filter,
                )
            )

            matches = []
            for match in query_result.matches:
                metadata = match.metadata or {}

                matches.append(
                    RetrievalMatch(
                        chunk_id=match.id,
                        score=match.score,
                        text=metadata.get("text", ""),
                        document_id=metadata.get("document_id", ""),
                        page_number=int(metadata.get("page_number", 0)),
                        chunk_number=int(metadata.get("chunk_number", 0)),
                        source=metadata.get("source", ""),
                        title=metadata.get("title"),
                        content_hash=metadata.get("content_hash"),
                        section=metadata.get("section"),
                    )
                )
        except Exception as error:
            raise RetrievalServiceError("Retrieval failed.") from error

        return RetrievalResult(query=normalized_query, matches=matches)
