from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage

from services.answering import (
    INSUFFICIENT_MESSAGE,
    SAFETY_NOTICE,
    AnswerCompleted,
    AnswerDelta,
    AnsweringError,
    AnsweringService,
    AnswerProgress,
)
from services.evidence import Evidence, EvidencePackage


def make_evidence(
    citation_id: str = "C1",
    chunk_id: str = "chunk-1",
    text: str = "Hypertension increases stroke risk.",
    title: str | None = "Cardiology Guide",
    source: str = "cardiology.pdf",
    page_number: int = 10,
    score: float = 0.92,
) -> Evidence:
    return Evidence(
        citation_id=citation_id,
        chunk_id=chunk_id,
        text=text,
        document_id="doc-123",
        page_number=page_number,
        source=source,
        score=score,
        title=title,
    )


def make_package(
    items: list[Evidence] | None = None,
    sufficient: bool = True,
    query: str = "hypertension risks",
) -> EvidencePackage:
    if items is None:
        items = [make_evidence()]
    return EvidencePackage(query=query, items=items, sufficient=sufficient)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "unsupported_content",
    [
        None,
        12345,
        ["chunk-1", "chunk-2"],
        {"text": "answer"},
        object(),
    ],
)
async def test_answer_unsupported_content_type_raises_answering_error(
    unsupported_content: object,
) -> None:
    evidence_service = AsyncMock()
    evidence_service.collect_evidence.return_value = make_package()

    chat_model = AsyncMock()
    response = SimpleNamespace(content=unsupported_content)
    chat_model.ainvoke.return_value = response

    service = AnsweringService(
        evidence_service=evidence_service,
        chat_model=chat_model,
    )

    with pytest.raises(
        AnsweringError, match="Answer provider returned unsupported content."
    ):
        await service.answer(query="What is hypertension?", tenant_id="tenant-1")


@pytest.mark.anyio
@pytest.mark.parametrize("empty_content", ["", "   ", "\n\t  \n"])
async def test_answer_empty_response_raises_answering_error(
    empty_content: str,
) -> None:
    evidence_service = AsyncMock()
    evidence_service.collect_evidence.return_value = make_package()

    chat_model = AsyncMock()
    chat_model.ainvoke.return_value = AIMessage(content=empty_content)

    service = AnsweringService(
        evidence_service=evidence_service,
        chat_model=chat_model,
    )

    with pytest.raises(
        AnsweringError, match="Answer provider returned empty response."
    ):
        await service.answer(query="What is hypertension?", tenant_id="tenant-1")


@pytest.mark.anyio
async def test_answer_chat_model_failure_raises_answering_error() -> None:
    evidence_service = AsyncMock()
    evidence_service.collect_evidence.return_value = make_package()

    chat_model = AsyncMock()
    chat_model.ainvoke.side_effect = RuntimeError("Connection timeout to LLM")

    service = AnsweringService(
        evidence_service=evidence_service,
        chat_model=chat_model,
    )

    with pytest.raises(AnsweringError, match="Answer provider failed.") as exc_info:
        await service.answer(query="What is hypertension?", tenant_id="tenant-1")

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.anyio
async def test_answer_chat_model_timeout_raises_answering_error() -> None:
    evidence_service = AsyncMock()
    evidence_service.collect_evidence.return_value = make_package()

    chat_model = AsyncMock()
    chat_model.ainvoke.side_effect = TimeoutError

    service = AnsweringService(
        evidence_service=evidence_service,
        chat_model=chat_model,
    )

    with pytest.raises(
        AnsweringError,
        match="Answer generation timed out.",
    ) as exc_info:
        await service.answer(query="What is hypertension?", tenant_id="tenant-1")

    assert isinstance(exc_info.value.__cause__, TimeoutError)


@pytest.mark.anyio
async def test_answer_insufficient_evidence_returns_insufficient_result() -> None:
    evidence_service = AsyncMock()
    evidence_service.collect_evidence.return_value = make_package(
        items=[], sufficient=False
    )

    chat_model = AsyncMock()
    service = AnsweringService(
        evidence_service=evidence_service,
        chat_model=chat_model,
    )

    result = await service.answer(query="Rare syndrome", tenant_id="tenant-1")

    assert result.answer == INSUFFICIENT_MESSAGE
    assert result.citations == []
    assert result.sufficient_evidence is False
    assert result.safety_notice == SAFETY_NOTICE
    chat_model.ainvoke.assert_not_called()


@pytest.mark.anyio
async def test_answer_success_with_valid_citations() -> None:
    e1 = make_evidence(citation_id="C1", text="High blood pressure causes strain.")
    e2 = make_evidence(citation_id="C2", text="It can lead to myocardial infarction.")
    evidence_service = AsyncMock()
    evidence_service.collect_evidence.return_value = make_package(items=[e1, e2])

    chat_model = AsyncMock()
    llm_text = (
        "High blood pressure strains the heart [C1], "
        "which may cause a myocardial infarction [C2]."
    )
    chat_model.ainvoke.return_value = AIMessage(content=llm_text)

    service = AnsweringService(
        evidence_service=evidence_service,
        chat_model=chat_model,
    )

    result = await service.answer(
        query="What does high blood pressure do?",
        tenant_id="tenant-1",
    )

    assert result.answer == llm_text
    assert result.citations == [e1, e2]
    assert result.sufficient_evidence is True
    assert result.safety_notice == SAFETY_NOTICE

    chat_model.ainvoke.assert_awaited_once()
    messages = chat_model.ainvoke.call_args[0][0]
    assert len(messages) == 2
    assert "You are an evidence-grounded" in messages[0].content
    assert "High blood pressure causes strain." in messages[1].content
    assert "[C1]" in messages[1].content
    assert "[C2]" in messages[1].content


@pytest.mark.anyio
async def test_answer_filters_uncited_evidence_from_result() -> None:
    e1 = make_evidence(citation_id="C1", text="Evidence 1")
    e2 = make_evidence(citation_id="C2", text="Evidence 2")
    evidence_service = AsyncMock()
    evidence_service.collect_evidence.return_value = make_package(items=[e1, e2])

    chat_model = AsyncMock()
    chat_model.ainvoke.return_value = AIMessage(
        content="Only the first point is relevant [C1]."
    )

    service = AnsweringService(
        evidence_service=evidence_service,
        chat_model=chat_model,
    )

    result = await service.answer(query="Query", tenant_id="tenant-1")

    assert result.citations == [e1]


@pytest.mark.anyio
async def test_answer_missing_citations_raises_answering_error() -> None:
    evidence_service = AsyncMock()
    evidence_service.collect_evidence.return_value = make_package()

    chat_model = AsyncMock()
    chat_model.ainvoke.return_value = AIMessage(
        content="Here is a completely uncited medical statement."
    )

    service = AnsweringService(
        evidence_service=evidence_service,
        chat_model=chat_model,
    )

    with pytest.raises(AnsweringError, match="Generated answer has no citations."):
        await service.answer(query="Query", tenant_id="tenant-1")


@pytest.mark.anyio
async def test_answer_invalid_citations_raises_answering_error() -> None:
    evidence_service = AsyncMock()
    evidence_service.collect_evidence.return_value = make_package(
        items=[make_evidence(citation_id="C1")]
    )

    chat_model = AsyncMock()
    chat_model.ainvoke.return_value = AIMessage(
        content="Statement grounded in [C1] but hallucinating [C99]."
    )

    service = AnsweringService(
        evidence_service=evidence_service,
        chat_model=chat_model,
    )

    with pytest.raises(
        AnsweringError, match="Generated answer contains invalid citations."
    ):
        await service.answer(query="Query", tenant_id="tenant-1")


def test_format_evidence_uses_title_when_present() -> None:
    service = AnsweringService(
        evidence_service=AsyncMock(),
        chat_model=AsyncMock(),
    )
    evidence = make_evidence(
        citation_id="C1",
        title="Merck Manual",
        source="merck.pdf",
        page_number=45,
        text="Aspirin inhibits COX-1.",
    )

    formatted = service._format_evidence(evidence)

    assert "[C1]" in formatted
    assert "Source: Merck Manual" in formatted
    assert "Page: 45" in formatted
    assert "Content: Aspirin inhibits COX-1." in formatted


def test_format_evidence_falls_back_to_source_when_title_is_none() -> None:
    service = AnsweringService(
        evidence_service=AsyncMock(),
        chat_model=AsyncMock(),
    )
    evidence = make_evidence(
        citation_id="C2",
        title=None,
        source="clinical_guidelines.pdf",
        page_number=12,
        text="Beta blockers reduce heart rate.",
    )

    formatted = service._format_evidence(evidence)

    assert "[C2]" in formatted
    assert "Source: clinical_guidelines.pdf" in formatted
    assert "Page: 12" in formatted


@pytest.mark.anyio
async def test_answer_forwards_tenant_id_and_document_id() -> None:
    evidence_service = AsyncMock()
    evidence_service.collect_evidence.return_value = make_package(sufficient=False)

    service = AnsweringService(
        evidence_service=evidence_service,
        chat_model=AsyncMock(),
    )
    target_uuid = UUID("a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11")

    await service.answer(
        query="Dosage query",
        tenant_id="tenant-hospital-9",
        document_id=target_uuid,
    )

    evidence_service.collect_evidence.assert_awaited_once_with(
        query="Dosage query",
        tenant_id="tenant-hospital-9",
        document_id=target_uuid,
    )


@pytest.mark.anyio
async def test_stream_answer_full_lifecycle() -> None:
    evidence = make_evidence(
        citation_id="C1",
        text="Beta blockers reduce heart rate.",
        title="Cardiology Guide",
        source="cardiology.pdf",
        page_number=10,
    )
    evidence_service = AsyncMock()
    evidence_service.collect_evidence.return_value = make_package(items=[evidence])

    async def fake_astream(messages: object):
        yield SimpleNamespace(content="Beta blockers ")
        yield SimpleNamespace(content="reduce heart rate [C1].")

    chat_model = AsyncMock()
    chat_model.astream = MagicMock(side_effect=fake_astream)

    service = AnsweringService(
        evidence_service=evidence_service,
        chat_model=chat_model,
    )

    items = [
        item
        async for item in service.stream_answer(
            query="What do beta blockers do?",
            tenant_id="tenant-1",
        )
    ]

    assert len(items) == 7
    assert isinstance(items[0], AnswerProgress)
    assert items[0].stage == "retrieval"
    assert isinstance(items[1], AnswerProgress)
    assert "Found 1" in items[1].message
    assert isinstance(items[2], AnswerProgress)
    assert items[2].stage == "answering"
    assert isinstance(items[3], AnswerDelta)
    assert items[3].text == "Beta blockers "
    assert isinstance(items[4], AnswerDelta)
    assert items[4].text == "reduce heart rate [C1]."
    assert isinstance(items[5], AnswerProgress)
    assert items[5].stage == "validation"
    assert isinstance(items[6], AnswerCompleted)
    assert items[6].result.answer == "Beta blockers reduce heart rate [C1]."
    assert items[6].result.citations == [evidence]
    assert items[6].result.sufficient_evidence is True


@pytest.mark.anyio
async def test_stream_answer_insufficient_evidence() -> None:
    evidence_service = AsyncMock()
    evidence_service.collect_evidence.return_value = make_package(sufficient=False)

    service = AnsweringService(
        evidence_service=evidence_service,
        chat_model=AsyncMock(),
    )

    items = [
        item
        async for item in service.stream_answer(
            query="Unknown treatment",
            tenant_id="tenant-1",
        )
    ]

    assert len(items) == 3
    assert isinstance(items[0], AnswerProgress)
    assert items[0].stage == "retrieval"
    assert isinstance(items[1], AnswerDelta)
    assert items[1].text == INSUFFICIENT_MESSAGE
    assert isinstance(items[2], AnswerCompleted)
    assert items[2].result.sufficient_evidence is False


@pytest.mark.anyio
async def test_stream_answer_missing_citations_raises_answering_error() -> None:
    evidence = make_evidence()
    evidence_service = AsyncMock()
    evidence_service.collect_evidence.return_value = make_package(items=[evidence])

    async def fake_astream(messages: object):
        yield SimpleNamespace(content="Beta blockers reduce heart rate.")

    chat_model = AsyncMock()
    chat_model.astream = MagicMock(side_effect=fake_astream)

    service = AnsweringService(
        evidence_service=evidence_service,
        chat_model=chat_model,
    )

    with pytest.raises(AnsweringError, match="has no citations"):
        async for _ in service.stream_answer(
            query="Beta blockers",
            tenant_id="tenant-1",
        ):
            pass


@pytest.mark.anyio
async def test_stream_answer_empty_response_raises_answering_error() -> None:
    evidence = make_evidence()
    evidence_service = AsyncMock()
    evidence_service.collect_evidence.return_value = make_package(items=[evidence])

    async def fake_astream(messages: object):
        yield SimpleNamespace(content="")

    chat_model = AsyncMock()
    chat_model.astream = MagicMock(side_effect=fake_astream)

    service = AnsweringService(
        evidence_service=evidence_service,
        chat_model=chat_model,
    )

    with pytest.raises(AnsweringError, match="empty response"):
        async for _ in service.stream_answer(
            query="Beta blockers",
            tenant_id="tenant-1",
        ):
            pass
