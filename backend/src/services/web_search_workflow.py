import json

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph.state import CompiledStateGraph

from services.evidence import Evidence, EvidencePackage


class WebSearchWorkflowError(Exception):
    pass


class WebSearchWorkflow:
    def __init__(
        self,
        graph: CompiledStateGraph,
    ) -> None:
        self._graph = graph

    async def search(
        self,
        query: str,
    ) -> EvidencePackage:
        normalized_query = query.strip()

        if not normalized_query:
            raise WebSearchWorkflowError("Web-search query cannot be empty.")

        try:
            result = await self._graph.ainvoke(
                {
                    "query": normalized_query,
                    "messages": [
                        HumanMessage(
                            content=normalized_query,
                        )
                    ],
                }
            )
        except Exception as error:
            raise WebSearchWorkflowError("Web-search workflow failed.") from error

        tool_message = self._find_tool_message(result.get("messages", []))

        if tool_message is None:
            raise WebSearchWorkflowError("The model did not call the web-search tool.")

        return self._parse_tool_result(
            query=normalized_query,
            message=tool_message,
        )

    def _find_tool_message(
        self,
        messages: object,
    ) -> ToolMessage | None:
        if not isinstance(messages, list):
            return None

        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                return message

        return None

    def _parse_tool_result(
        self,
        query: str,
        message: ToolMessage,
    ) -> EvidencePackage:
        if not isinstance(message.content, str):
            raise WebSearchWorkflowError(
                "Web-search tool returned unsupported content."
            )

        try:
            payload = json.loads(message.content)
        except json.JSONDecodeError as error:
            raise WebSearchWorkflowError(
                "Web-search tool returned invalid JSON."
            ) from error

        if not isinstance(payload, dict):
            raise WebSearchWorkflowError("Web-search tool returned an invalid object.")

        raw_items = payload.get("items")

        if not isinstance(raw_items, list):
            raise WebSearchWorkflowError("Web-search tool returned invalid evidence.")

        evidence: list[Evidence] = []

        for raw_item in raw_items:
            item = self._parse_evidence(raw_item)

            if item is not None:
                evidence.append(item)

        return EvidencePackage(
            query=query,
            items=evidence,
            sufficient=bool(evidence),
        )

    def _parse_evidence(
        self,
        raw_item: object,
    ) -> Evidence | None:
        if not isinstance(raw_item, dict):
            return None

        try:
            citation_id = str(raw_item["citation_id"])
            chunk_id = str(raw_item["chunk_id"])
            text = str(raw_item["text"])
            document_id = str(raw_item["document_id"])
            page_number = int(raw_item["page_number"])
            source = str(raw_item["source"])
            score = float(raw_item["score"])

            raw_title = raw_item.get("title")
            title = str(raw_title) if raw_title is not None else None
        except (KeyError, TypeError, ValueError):
            return None

        if document_id != "web":
            return None

        if not citation_id or not text or not source:
            return None

        return Evidence(
            citation_id=citation_id,
            chunk_id=chunk_id,
            text=text,
            document_id=document_id,
            page_number=page_number,
            source=source,
            score=score,
            title=title,
        )
