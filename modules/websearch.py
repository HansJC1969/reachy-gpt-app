"""
Web search for Reachy — gives GPT-4o access to current information.

Backend priority:
  1. Tavily   (TAVILY_API_KEY set in .env)  — designed for AI agents, best quality
  2. DuckDuckGo (duckduckgo-search)         — free, no key needed

Standalone test:
    python -m modules.websearch "Wetter in Berlin heute"
"""

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MAX_SNIPPET_LEN = 300   # truncate individual snippets to keep GPT context lean


@dataclass
class SearchResult:
    title:   str
    url:     str
    snippet: str


class WebSearcher:
    """
    Unified search interface.  Auto-selects the best available backend.

    Usage
    -----
    searcher = WebSearcher()
    results  = searcher.search("Pollen Robotics Reachy Mini specs")
    text     = searcher.format_results(results)
    """

    def __init__(self) -> None:
        self._backend = self._detect_backend()
        logger.info("WebSearcher using backend: %s", self._backend)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run a web search and return up to *max_results* results."""
        logger.info("Searching: %r", query)
        try:
            if self._backend == "tavily":
                return self._search_tavily(query, max_results)
            return self._search_ddg(query, max_results)
        except Exception:
            logger.exception("Search failed for %r", query)
            return []

    def format_results(self, results: list[SearchResult]) -> str:
        """Format results as a numbered markdown list for GPT context."""
        if not results:
            return "No results found."
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            snippet = r.snippet[:MAX_SNIPPET_LEN]
            if len(r.snippet) > MAX_SNIPPET_LEN:
                snippet += "…"
            lines.append(f"{i}. **{r.title}**\n   {snippet}\n   Source: {r.url}")
        return "\n\n".join(lines)

    def search_and_format(self, query: str, max_results: int = 5) -> str:
        """Convenience: search and return formatted string in one call."""
        return self.format_results(self.search(query, max_results))

    @property
    def backend(self) -> str:
        return self._backend

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_backend() -> str:
        if os.environ.get("TAVILY_API_KEY"):
            try:
                import tavily  # noqa: F401
                return "tavily"
            except ImportError:
                logger.warning("TAVILY_API_KEY set but tavily-python not installed; using DuckDuckGo")
        try:
            from duckduckgo_search import DDGS  # noqa: F401
            return "duckduckgo"
        except ImportError:
            logger.warning(
                "duckduckgo-search not installed. "
                "Run: pip install duckduckgo-search"
            )
            return "none"

    # ------------------------------------------------------------------
    # Tavily backend
    # ------------------------------------------------------------------

    def _search_tavily(self, query: str, max_results: int) -> list[SearchResult]:
        from tavily import TavilyClient  # type: ignore
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY", ""))
        resp = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
        )
        results: list[SearchResult] = []
        for r in resp.get("results", []):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
            ))
        return results

    # ------------------------------------------------------------------
    # DuckDuckGo backend
    # ------------------------------------------------------------------

    def _search_ddg(self, query: str, max_results: int) -> list[SearchResult]:
        if self._backend == "none":
            logger.error("No search backend available")
            return []
        from duckduckgo_search import DDGS  # type: ignore
        results: list[SearchResult] = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                ))
        return results


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    query = " ".join(sys.argv[1:]) or "Pollen Robotics Reachy Mini"
    searcher = WebSearcher()
    print(f"Backend: {searcher.backend}\n")
    results = searcher.search(query)
    print(searcher.format_results(results))
