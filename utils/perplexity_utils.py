# utils/perplexity_utils.py

import requests
import json
from typing import List, Dict
import os
from dotenv import load_dotenv
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")

if not PERPLEXITY_API_KEY:
    logger.warning("PERPLEXITY_API_KEY is not set in the environment variables.")

def single_query_perplexity(query: str, model: str = "llama-3.1-sonar-small-128k-online", max_tokens: int = 1024,
                            temperature: float = 0.2) -> Dict:
    """
    Send a single query to the Perplexity API and return the result.

    Args:
        query (str): The query to send to Perplexity.
        model (str): The model to use for the query.
        max_tokens (int): The maximum number of tokens to generate.
        temperature (float): The temperature for the generation.

    Returns:
        Dict: The JSON response from the Perplexity API.
    """
    url = "https://api.perplexity.ai/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant providing accurate and up-to-date information."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "return_citations": True,
        "search_domain_filter": ["perplexity.ai"],
        "return_images": False,
        "return_related_questions": False,
        "search_recency_filter": "month",
        "top_k": 0,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 1
    }

    headers = {
        "Authorization": f"Bearer {PERPLEXITY_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raises a HTTPError if the status is 4xx, 5xx
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error querying Perplexity API: {str(e)}")
        return {"error": str(e)}

def extract_content_from_perplexity_response(response: Dict) -> str:
    """
    Extract the content from a Perplexity API response.

    Args:
        response (Dict): The JSON response from the Perplexity API.

    Returns:
        str: The extracted content, or an error message if extraction fails.
    """
    try:
        return response['choices'][0]['message']['content']
    except KeyError:
        logger.error("Failed to extract content from Perplexity response")
        return "Error: Unable to extract content from Perplexity response"

def research_topic(topic: str, num_queries: int = 3) -> List[Dict]:
    """
    Perform multiple queries on a topic using the Perplexity API.

    Args:
        topic (str): The topic to research.
        num_queries (int): The number of queries to perform.

    Returns:
        List[Dict]: A list of Perplexity API responses.
    """
    results = []
    for i in range(num_queries):
        query = f"Provide detailed and up-to-date information about {topic}. Focus on aspect {i + 1}."
        result = single_query_perplexity(query)
        results.append(result)
    return results

def summarize_research(research_results: List[Dict]) -> str:
    """
    Summarize the results of multiple Perplexity queries.

    Args:
        research_results (List[Dict]): A list of Perplexity API responses.

    Returns:
        str: A summary of the research results.
    """
    summary = "Research Summary:\n\n"
    for i, result in enumerate(research_results, 1):
        content = extract_content_from_perplexity_response(result)
        summary += f"Query {i} Results:\n{content}\n\n"
    return summary

# Example usage
if __name__ == "__main__":
    topic = "artificial intelligence in healthcare"
    results = research_topic(topic)
    print(summarize_research(results))
