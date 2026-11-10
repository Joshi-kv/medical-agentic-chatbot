import asyncio
import re
from dataclasses import dataclass
from uuid import UUID

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from prompts.answering import SYSTEM_PROMPT, USER_PROMPT
from services.evidence import Evidence, EvidenceService

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


class AnsweringError(Exception):
    pass


class AnsweringService:
    def __init__(
        self,
        evidence_service: EvidenceService,
        chat_model: BaseChatModel,
    ) -> None:
        self._evidence_service = evidence_service
        self._chat_model = chat_model

    async def answer(
        self,
        query: str,
        tenant_id: str,
        document_id: UUID | None = None,
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
            return AnswerResult(
                answer=INSUFFICIENT_MESSAGE,
                citations=[],
                sufficient_evidence=False,
                safety_notice=SAFETY_NOTICE,
            )

        evidence_items = evidence_package.items

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
        )

    def _format_evidence(self, evidence: Evidence) -> str:
        source_name = evidence.title or evidence.source

        return (
            f"[{evidence.citation_id}]\n"
            f"Source: {source_name}\n"
            f"Page: {evidence.page_number}\n"
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
