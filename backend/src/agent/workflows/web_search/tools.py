import json
from dataclasses import asdict

from langchain_core.tools import BaseTool, tool

from services.web_search import WebSearchService


def create_medical_web_search_tool(
    service: WebSearchService,
) -> BaseTool:
    @tool("search_trusted_medical_web")
    async def search_trusted_medical_web(
        query: str,
    ) -> str:
        """Search trusted medical websites for patient-education evidence."""

        package = await service.collect_evidence(query)

        return json.dumps(
            {
                "query": package.query,
                "sufficient": package.sufficient,
                "items": [asdict(item) for item in package.items],
            }
        )

    return search_trusted_medical_web
