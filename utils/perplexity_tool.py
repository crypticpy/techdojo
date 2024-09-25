# utils/perplexity_tool.py

import logging
from typing import Type
from utils.perplexity_utils import single_query_perplexity, extract_content_from_perplexity_response
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class PerplexitySearchInput(BaseModel):
    query: str = Field(..., description="The search query to send to Perplexity")

class PerplexitySearchTool(BaseTool):
    name = "Perplexity Search"
    description = "Use this tool to search for information on the web using Perplexity AI."
    args_schema: Type[BaseModel] = PerplexitySearchInput

    def _run(self, query: str) -> str:
        logger.info(f"🔍 Searching Perplexity for: {query}")
        try:
            result = single_query_perplexity(
                query,
                model="llama-3.1-sonar-small-128k-online",
                max_tokens=2048,
                temperature=0.2
            )
            content = extract_content_from_perplexity_response(result)
            logger.info("✅ Perplexity search completed")
            return f"Perplexity Search Results:\n\n{content}\n\nPlease analyze these results and incorporate relevant information into your response."
        except Exception as e:
            error_message = f"Error in Perplexity search: {str(e)}"
            logger.error(error_message, exc_info=True)
            return f"An error occurred during the Perplexity search: {str(e)}"

    async def _arun(self, query: str) -> str:
        return self._run(query)
