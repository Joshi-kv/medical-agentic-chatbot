import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prompts.answering import SYSTEM_PROMPT, USER_PROMPT
from services.conversation_context import ContextMessage
from services.evidence import Evidence, EvidencePackage, EvidenceService
from services.web_search_workflow import WebSearchWorkflow, WebSearchWorkflowError

CITATION_PATTERN = re.compile(r"\[(C\d+)\]")

SAFETY_NOTICE = (
    "This information is educational and is not a diagnosis or "
    "a substitute for professional medical care."
)

INSUFFICIENT_MESSAGE = (
    "I couldn't find enough reliable information in the available "
    "medical sources to answer this safely."
)

ANSWER_TIMEOUT_SECONDS = 35


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: list[Evidence]
    sufficient_evidence: bool
    safety_notice: str
    route: str = "curated_documents"
    review_case_id: str | None = None


@dataclass(frozen=True)
class AnswerProgress:
    stage: str
    message: str


@dataclass(frozen=True)
class AnswerDelta:
    text: str


@dataclass(frozen=True)
class AnswerCompleted:
    result: AnswerResult


AnswerStreamItem = AnswerProgress | AnswerDelta | AnswerCompleted


class AnswerProvider(Protocol):
    async def answer(
        self,
        query: str,
        tenant_id: str,
        document_id: UUID | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        conversation_history: list[ContextMessage] | None = None,
    ) -> AnswerResult: ...

    def stream_answer(
        self,
        query: str,
        tenant_id: str,
        document_id: UUID | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
        conversation_history: list[ContextMessage] | None = None,
    ) -> AsyncIterator[AnswerStreamItem]: ...


class AnsweringError(Exception):
    pass


class AnsweringService:
    def __init__(
        self,
        evidence_service: EvidenceService,
        chat_model: BaseChatModel,
        web_search_workflow: WebSearchWorkflow | None = None,
    ) -> None:
        self._evidence_service = evidence_service
        self._chat_model = chat_model
        self._web_search_workflow = web_search_workflow

    async def answer(
        self,
        query: str,
        tenant_id: str,
        document_id: UUID | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> AnswerResult:
        """
        Answer the query based on the evidence.
        """
        evidence_package = await self._evidence_service.collect_evidence(
            query=query,
            tenant_id=tenant_id,
            document_id=document_id,
        )

        if not evidence_package.sufficient:
            web_evidence = await self._search_web_if_allowed(query, document_id)
            if web_evidence is not None and web_evidence.sufficient:
                evidence_package = web_evidence

        if not evidence_package.sufficient:
            return AnswerResult(
                answer=INSUFFICIENT_MESSAGE,
                citations=[],
                sufficient_evidence=False,
                safety_notice=SAFETY_NOTICE,
            )

        evidence_items = evidence_package.items
        route = self._evidence_route(evidence_items)

        evidence_text = "\n\n".join(
            self._format_evidence(item) for item in evidence_items
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=USER_PROMPT.format(
                    query=query,
                    evidence_text=evidence_text,
                )
            ),
        ]

        try:
            async with asyncio.timeout(ANSWER_TIMEOUT_SECONDS):
                response = await self._chat_model.ainvoke(messages)
        except TimeoutError as error:
            raise AnsweringError("Answer generation timed out.") from error
        except Exception as error:
            raise AnsweringError("Answer provider failed.") from error

        if not isinstance(response.content, str):
            raise AnsweringError("Answer provider returned unsupported content.")

        answer_text = response.content.strip()

        if not answer_text:
            raise AnsweringError("Answer provider returned empty response.")

        citations = self._validate_citations(
            answer=answer_text,
            evidence=evidence_items,
        )

        return AnswerResult(
            answer=answer_text,
            citations=citations,
            sufficient_evidence=True,
            safety_notice=SAFETY_NOTICE,
            route=route,
        )

    async def stream_answer(
        self,
        query: str,
        tenant_id: str,
        document_id: UUID | None = None,
        conversation_id: str | None = None,
        run_id: str | None = None,
    ) -> AsyncIterator[AnswerStreamItem]:
        yield AnswerProgress(
            stage="retrieval",
            message="Searching your medical documents...",
        )

        try:
            evidence_package = await self._evidence_service.collect_evidence(
                query=query,
                tenant_id=tenant_id,
                document_id=document_id,
            )
        except Exception as error:
            raise AnsweringError("Evidence retrieval failed.") from error

        if not evidence_package.sufficient and self._web_search_allowed(document_id):
            yield AnswerProgress(
                stage="web_search",
                message="Searching trusted medical websites...",
            )
            web_evidence = await self._search_web_if_allowed(query, document_id)
            if web_evidence is not None and web_evidence.sufficient:
                evidence_package = web_evidence

        if not evidence_package.sufficient:
            result = AnswerResult(
                answer=INSUFFICIENT_MESSAGE,
                citations=[],
                sufficient_evidence=False,
                safety_notice=SAFETY_NOTICE,
            )
            yield AnswerDelta(text=INSUFFICIENT_MESSAGE)
            yield AnswerCompleted(result=result)
            return

        evidence_items = evidence_package.items
        route = self._evidence_route(evidence_items)

        yield AnswerProgress(
            stage=route,
            message=f"Found {len(evidence_items)} relevant medical passages.",
        )

        evidence_text = "\n\n".join(
            self._format_evidence(item) for item in evidence_items
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=USER_PROMPT.format(
                    query=query,
                    evidence_text=evidence_text,
                )
            ),
        ]

        yield AnswerProgress(
            stage="answering",
            message="Generating an evidence-based answer...",
        )

        answer_parts: list[str] = []

        try:
            async with asyncio.timeout(ANSWER_TIMEOUT_SECONDS):
                async for chunk in self._chat_model.astream(messages):
                    if not isinstance(chunk.content, str):
                        continue

                    text = chunk.content
                    if not text:
                        continue

                    answer_parts.append(text)
                    yield AnswerDelta(text=text)

        except TimeoutError as error:
            raise AnsweringError("Answer generation timed out.") from error
        except Exception as error:
            raise AnsweringError("Answer provider failed.") from error

        answer_text = "".join(answer_parts).strip()

        if not answer_text:
            raise AnsweringError("Answer provider returned an empty response.")

        yield AnswerProgress(
            stage="validation",
            message="Validating medical citations...",
        )

        citations = self._validate_citations(
            answer=answer_text,
            evidence=evidence_items,
        )

        result = AnswerResult(
            answer=answer_text,
            citations=citations,
            sufficient_evidence=True,
            safety_notice=SAFETY_NOTICE,
            route=route,
        )

        yield AnswerCompleted(result=result)

    def _format_evidence(self, evidence: Evidence) -> str:
        source_name = evidence.title or evidence.source
        location = (
            f"URL: {evidence.source}"
            if evidence.document_id == "web"
            else f"Page: {evidence.page_number}"
        )

        return (
            f"[{evidence.citation_id}]\n"
            f"Source: {source_name}\n"
            f"{location}\n"
            f"Content: {evidence.text}"
        )

    def _validate_citations(
        self,
        answer: str,
        evidence: list[Evidence],
    ) -> list[Evidence]:
        cited_ids = set(CITATION_PATTERN.findall(answer))

        if not cited_ids:
            raise AnsweringError("Generated answer has no citations.")

        valid_ids = {item.citation_id for item in evidence}
        invalid_ids = cited_ids - valid_ids

        if invalid_ids:
            raise AnsweringError("Generated answer contains invalid citations.")

        return [item for item in evidence if item.citation_id in cited_ids]

    async def _search_web_if_allowed(
        self,
        query: str,
        document_id: UUID | None,
    ) -> EvidencePackage | None:
        if document_id is not None:
            return None

        if self._web_search_workflow is None:
            return None

        try:
            return await self._web_search_workflow.search(query)
        except WebSearchWorkflowError:
            return None

    def _web_search_allowed(self, document_id: UUID | None) -> bool:
        return document_id is None and self._web_search_workflow is not None

    @staticmethod
    def _evidence_route(evidence: list[Evidence]) -> str:
        if evidence and all(item.document_id == "web" for item in evidence):
            return "trusted_web"

        return "curated_documents"
