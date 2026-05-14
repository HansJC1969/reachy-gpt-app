"""
Web search for Reachy — gives GPT-4o access to current information.

Backend priority:
  1. Tavily (TAVILY_API_KEY set in .env)  — PRIMARY; designed for AI agents
  2. DuckDuckGo (ddgs package)            — fallback only if Tavily fails

Query-aware Tavily settings:
  • Finance queries (bitcoin, price, stock…): topic="finance", time_range="day"
  • News/weather/sports queries:             topic="news",    time_range="day"
  • Everything else:                         topic="general", search_depth="advanced"
  include_answer="advanced" is always requested so GPT gets a synthesised answer.

If Tavily fails at runtime the call falls through to DuckDuckGo automatically.

DuckDuckGo uses backend="html" (avoids Bing rate limits) via the "ddgs" package.

Standalone test:
    python -m modules.websearch "Bitcoin price today"
    python -m modules.websearch "Wetter in Berlin heute"
"""

import logging
import os
import re
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MAX_SNIPPET_LEN = 400


# ── Query classification ──────────────────────────────────────────────────────

_FINANCE_RE = re.compile(
    r"\b(price|preis|kurs|wert|bitcoin|btc|eth|ethereum|crypto|kryptowährung"
    r"|aktie|stock|nasdaq|dax|dow|s&p|gold|silver|silber|forex|währung|dollar"
    r"|euro|yen|inflation|zinsen|interest.rate)\b",
    re.IGNORECASE,
)

_NEWS_RE = re.compile(
    r"\b(wetter|weather|temperatur|temperature|forecast|vorhersage"
    r"|news|nachrichten|aktuell|current|today|heute|jetzt|now|live"
    r"|sport|score|ergebnis|result|fußball|football|soccer|election|wahl)\b",
    re.IGNORECASE,
)


def _classify_query(query: str) -> tuple[str, str | None]:
    """
    Return (tavily_topic, time_range) for the query.

    topic     : "finance" | "news" | "general"
    time_range: "day" | None
    """
    if _FINANCE_RE.search(query):
        return "finance", "day"
    if _NEWS_RE.search(query):
        return "news", "day"
    return "general", None


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    title:   str
    url:     str
    snippet: str


# ── Main searcher ─────────────────────────────────────────────────────────────

class WebSearcher:
    """
    Unified search interface.  Auto-selects the best available backend.

    Usage
    -----
    searcher = WebSearcher()
    results  = searcher.search("Bitcoin price today")
    text     = searcher.format_results(results)
    """

    def __init__(self) -> None:
        self._backend = self._detect_backend()
        logger.info("WebSearcher using backend: %s", self._backend)

    # ── Public API ────────────────────────────────────────────────────────────

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run a web search and return up to *max_results* results."""
        topic, time_range = _classify_query(query)
        logger.info("Searching [%s | topic=%s]: %r", self._backend, topic, query)

        # Always try Tavily first if the API key is present, regardless of
        # which backend was detected at start-up, because Tavily is more
        # reliable for financial and real-time queries.
        if os.environ.get("TAVILY_API_KEY"):
            try:
                return self._search_tavily(query, max_results, topic=topic, time_range=time_range)
            except Exception as exc:
                logger.warning("Tavily failed (%s) — falling back to DuckDuckGo", exc)

        if self._backend != "none":
            try:
                return self._search_ddg(query, max_results, time_range=time_range)
            except Exception:
                logger.exception("DuckDuckGo search failed for %r", query)

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

    # ── Backend detection ─────────────────────────────────────────────────────

    @staticmethod
    def _detect_backend() -> str:
        if os.environ.get("TAVILY_API_KEY"):
            try:
                import tavily  # noqa: F401
                return "tavily"
            except ImportError:
                logger.warning("TAVILY_API_KEY set but tavily-python not installed")
        try:
            import ddgs  # noqa: F401
            return "duckduckgo"
        except ImportError:
            logger.warning("ddgs not installed — run: pip install ddgs")
            return "none"

    # ── Tavily ────────────────────────────────────────────────────────────────

    def _search_tavily(
        self,
        query: str,
        max_results: int,
        *,
        topic: str,
        time_range: str | None,
    ) -> list[SearchResult]:
        from tavily import TavilyClient  # type: ignore

        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

        kwargs: dict = dict(
            query=query,
            max_results=max_results,
            topic=topic,
            include_answer="advanced",   # AI-synthesised direct answer
        )
        if time_range:
            kwargs["time_range"] = time_range
        if topic == "general":
            kwargs["search_depth"] = "advanced"

        resp = client.search(**kwargs)
        results: list[SearchResult] = []

        # Prepend the synthesised answer as a top result when present
        answer = (resp.get("answer") or "").strip()
        if answer:
            results.append(SearchResult(
                title="Direct answer",
                url="https://tavily.com",
                snippet=answer,
            ))

        for r in resp.get("results", []):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("content", ""),
            ))

        return results

    # ── DuckDuckGo ────────────────────────────────────────────────────────────

    def _search_ddg(
        self,
        query: str,
        max_results: int,
        *,
        time_range: str | None,
    ) -> list[SearchResult]:
        DDGS = self._import_ddgs()
        if DDGS is None:
            logger.error("No DuckDuckGo package available")
            return []

        # timelimit: Tavily uses "day"/"week"; DDG uses "d"/"w"
        ddg_timelimit_map = {"day": "d", "week": "w", "month": "m", "year": "y"}
        timelimit = ddg_timelimit_map.get(time_range or "", None)

        results: list[SearchResult] = []
        # html backend avoids Bing (which rate-limits heavily for finance queries)
        for backend in ("html", "lite"):
            try:
                with DDGS() as ddgs:
                    for r in ddgs.text(
                        query,
                        max_results=max_results,
                        backend=backend,
                        timelimit=timelimit,
                    ):
                        results.append(SearchResult(
                            title=r.get("title", ""),
                            url=r.get("href", ""),
                            snippet=r.get("body", ""),
                        ))
                if results:
                    return results
            except Exception as exc:
                logger.warning("DDG backend=%s failed (%s)", backend, exc)

        return results

    @staticmethod
    def _import_ddgs():
        """Return the DDGS class from the ddgs package, or None if not installed."""
        try:
            from ddgs import DDGS  # type: ignore
            return DDGS
        except ImportError:
            return None


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(name)s: %(message)s")

    query = " ".join(sys.argv[1:]) or "Pollen Robotics Reachy Mini"
    topic, time_range = _classify_query(query)
    searcher = WebSearcher()
    print(f"Backend    : {searcher.backend}")
    print(f"Topic      : {topic}  |  time_range: {time_range}\n")
    results = searcher.search(query)
    print(searcher.format_results(results))
