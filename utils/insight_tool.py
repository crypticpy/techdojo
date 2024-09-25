# utils/insight_tool.py

import requests
import logging
import json
from typing import List, Dict, Any
from langchain.tools import BaseTool
from langchain_core.documents import Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query")
    num_results: int = Field(4, description="Number of results to return", ge=1, le=20)

class InSightKB(BaseTool):
    name = "InSightKB"
    description = "Searches the InSight knowledge base for relevant documents. Input should be a JSON string with 'query' and optionally 'num_results' fields."
    base_url: str = Field(..., description="Base URL for the InSight API")

    def __init__(self, base_url: str, **kwargs):
        super().__init__(base_url=base_url, **kwargs)
        logger.info(f"InSightKB initialized with base_url: {self.base_url}")

    def _run(self, input_str: str) -> str:
        """
        Searches the InSight knowledge base using the provided input.

        Args:
            input_str (str): A JSON string containing the search query and optionally the number of results to return.

        Returns:
            str: A formatted string containing the retrieved documents.
        """
        try:
            input_data = json.loads(input_str)
            query = input_data.get("query")
            num_results = input_data.get("num_results", 4)
        except json.JSONDecodeError:
            # If input is not valid JSON, assume it's just the query
            query = input_str
            num_results = 4

        logger.info(f"Searching InSight KB for: {query}")
        logger.info(f"Using base URL: {self.base_url}")

        try:
            search_request = SearchRequest(query=query, num_results=num_results)
            url = f"{self.base_url}/search_get"  # Updated to use the new endpoint
            logger.info(f"Sending request to: {url}")
            response = requests.get(
                url,
                params=search_request.dict(),
                timeout=10  # Added timeout for better error handling
            )
            response.raise_for_status()
            data = response.json()

            documents = [
                Document(
                    page_content=doc["content"],
                    metadata=doc["metadata"],
                )
                for doc in data["documents"]
            ]

            if documents:
                formatted_docs = "\n\n".join(
                    [
                        f"**Document {i + 1}:**\n"
                        f"Content: {doc.page_content}\n"
                        f"Source: {doc.metadata.get('source', 'Unknown')}\n"
                        f"Relevance: {doc.metadata.get('relevance_score', 'Unknown')}\n"
                        for i, doc in enumerate(documents)
                    ]
                )
                return f"Found {len(documents)} documents in InSight KB:\n\n{formatted_docs}"
            else:
                return "No relevant documents found in InSight KB."
        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching InSight KB: {str(e)}")
            return f"An error occurred while searching InSight KB: {str(e)}"

    async def _arun(self, input_str: str) -> str:
        """Asynchronous version of _run."""
        # For now, we'll just call the synchronous version
        # In the future, you might want to implement an async version using aiohttp
        return self._run(input_str)
