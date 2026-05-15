"""
Web Search Tool

WHY TOOLS MUST HAVE STRICT SCHEMAS:
- Strict schemas ensure AI agents understand exactly what inputs are required
  and what outputs to expect, preventing runtime errors and ambiguous behavior.
- Type validation catches errors early before tool execution, improving reliability.
- Well-defined schemas enable automatic tool documentation and introspection.
- Schemas serve as contracts between the AI and the tool, making integrations
  predictable and maintainable as the system evolves.

WHY READ-ONLY TOOLS ARE SAFEST TO START WITH:
- Read-only tools (like search) cannot modify system state or data, minimizing
  risk of unintended consequences during development and testing.
- They're ideal for building confidence in agent behavior before introducing
  tools that can make changes (write files, execute commands, etc.).
- Easier to debug and test since they have no side effects.
- Lower security risk - even if misused, they won't damage systems or data.
- Best practice: Start with observation/query tools before adding action tools.
"""

from pydantic import BaseModel, Field
from typing import List, Dict
import requests
import logging

logger = logging.getLogger(__name__)


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    
    query: str = Field(
        ...,
        description="The search query string to look up on the web"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results to return (1-20)"
    )


class WebSearchResult(BaseModel):
    """Output schema for web search tool results."""
    
    results: List[Dict[str, str]] = Field(
        ...,
        description="List of search results, each containing 'title', 'snippet', and 'source'"
    )
    notes: str = Field(
        default="",
        description="Additional notes or metadata about the search operation"
    )


"""
WHY TOOLS MUST FAIL SAFELY:
- Tools should never crash the entire agent system due to external failures
  (network issues, API limits, malformed data, etc.).
- Safe failure means catching exceptions, logging errors, and returning
  valid (empty) responses with clear error notes instead of propagating exceptions.
- This ensures the agent can continue reasoning and potentially try alternative
  approaches rather than halting completely.
- Graceful degradation maintains system reliability even when individual tools fail.

WHY PARTIAL RESULTS ARE BETTER THAN HALLUCINATIONS:
- Returning empty or partial results with honest notes preserves trust and accuracy.
- An agent that receives "no results found" can acknowledge limitations or try
  a different approach, whereas hallucinated data leads to confident but wrong answers.
- It's better to say "I couldn't find that" than to make up plausible-sounding
  but incorrect information.
- Partial results (e.g., 3 results when 10 were requested) still provide value
  while being transparent about limitations.
- This aligns with the principle: "I don't know" is always better than a confident lie.
"""


def web_search(input_data: WebSearchInput) -> WebSearchResult:
    """
    Perform a web search and return structured results.
    
    Uses DuckDuckGo Instant Answer API (free, no API key required).
    Falls back to a safe empty result with notes if search fails.
    
    NOTE: For production use, consider:
    - SerpAPI (paid, requires API key, more reliable)
    - Google Custom Search API (free tier available)
    - Bing Search API (requires Azure subscription)
    - ddgs Python library (recommended for production)
    
    Args:
        input_data: WebSearchInput containing query and max_results
        
    Returns:
        WebSearchResult with search results and notes
    """
    try:
        # Try to use ddgs library if available
        try:
            from ddgs import DDGS
            
            with DDGS() as ddgs:
                search_results = list(ddgs.text(
                    input_data.query,
                    max_results=input_data.max_results
                ))
            
            results = []
            for item in search_results:
                results.append({
                    "title": item.get("title", "No title"),
                    "snippet": item.get("body", "No description available"),
                    "source": item.get("href", "")
                })
            
            notes = f"Successfully retrieved {len(results)} results using DuckDuckGo"
            if len(results) < input_data.max_results:
                notes += f" (requested {input_data.max_results}, found {len(results)})"
            
            return WebSearchResult(results=results, notes=notes)
            
        except ImportError:
            # Fallback to simple HTTP-based approach
            logger.info("ddgs library not found, using fallback API")
            
            # Use DuckDuckGo's Instant Answer API (limited but free)
            url = "https://api.duckduckgo.com/"
            params = {
                "q": input_data.query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1
            }
            
            headers = {
                "User-Agent": "AgentOps/1.0"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            
            # Extract results from RelatedTopics
            related_topics = data.get("RelatedTopics", [])
            for topic in related_topics[:input_data.max_results]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:100],
                        "snippet": topic.get("Text", "No description available"),
                        "source": topic.get("FirstURL", "")
                    })
            
            # If no related topics, try Abstract
            if not results and data.get("Abstract"):
                results.append({
                    "title": data.get("Heading", "Result"),
                    "snippet": data.get("Abstract", ""),
                    "source": data.get("AbstractURL", "")
                })
            
            notes = f"Retrieved {len(results)} results using DuckDuckGo Instant Answer API"
            if len(results) < input_data.max_results:
                notes += " (limited results; install 'ddgs' for better coverage)"
            
            return WebSearchResult(results=results, notes=notes)
    
    except requests.Timeout:
        logger.warning(f"Search timeout for query: {input_data.query}")
        return WebSearchResult(
            results=[],
            notes="Search request timed out. Please try again or rephrase your query."
        )
    
    except requests.RequestException as e:
        logger.error(f"Network error during search: {e}")
        return WebSearchResult(
            results=[],
            notes=f"Network error occurred while searching. Error: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Unexpected error during search: {e}")
        return WebSearchResult(
            results=[],
            notes="An unexpected error occurred during search. The tool failed safely with no results."
        )


# Example usage (for testing)
if __name__ == "__main__":
    # Test the search function
    search_input = WebSearchInput(query="Python programming", max_results=5)
    result = web_search(search_input)
    
    print(f"Results: {len(result.results)}")
    print(f"Notes: {result.notes}")
    for i, res in enumerate(result.results, 1):
        print(f"\n{i}. {res['title']}")
        print(f"   {res['snippet'][:100]}...")
        print(f"   {res['source']}")
