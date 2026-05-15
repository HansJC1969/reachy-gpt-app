"""
Web search for Reachy — gives GPT-4o access to current information.

Backend priority
----------------
  1. Tavily (TAVILY_API_KEY in .env)  — PRIMARY; AI-native, real-time results
  2. DuckDuckGo (ddgs package)         — last-resort fallback only

Tavily is always tried first when TAVILY_API_KEY is present.  DuckDuckGo is
only reached if Tavily raises an exception.

Query-aware Tavily settings
---------------------------
  Finance (bitcoin, kurs, aktie…): topic="finance", time_range="day"
  News/weather/sports (heute, wetter, nachrichten…): topic="news", time_range="day"
  Everything else:  search_depth="advanced"
  include_answer="advanced" is always set → Tavily prepends a synthesised answer.

Standalone test
---------------
    python -m modules.websearch "Bitcoin Kurs heute"
    python -m modules.websearch "Wetter Wien"
    python -m modules.websearch "aktuelle Nachrichten"
    python -m modules.websearch --debug "Euro Kurs"
"""

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Always load from the project-root .env regardless of the working directory.
# This ensures the key is present whether the module is imported from main.py,
# from a test script, or from any other working directory.
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH)

logger = logging.getLogger(__name__)

MAX_SNIPPET_LEN = 500


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
    Return (tavily_topic, time_range).

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
    Unified search interface.

    Tavily is always the primary backend.  DuckDuckGo is a last-resort fallback.

    Usage
    -----
    searcher = WebSearcher()
    text     = searcher.search_and_format("Bitcoin Kurs heute")
    """

    def __init__(self) -> None:
        # Re-read the key here so we pick up any late .env loading
        self._tavily_key: str = os.environ.get("TAVILY_API_KEY", "").strip()
        self._has_ddg: bool   = self._check_ddg()
        self._log_startup()

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def backend(self) -> str:
        """Primary backend that will be attempted first."""
        if self._tavily_key:
            return "tavily"
        if self._has_ddg:
            return "duckduckgo"
        return "none"

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Run a web search and return up to *max_results* results."""
        topic, time_range = _classify_query(query)

        # ── Tavily (primary) ─────────────────────────────────────────────────
        # Re-check key at call time so it works even if the env var is set
        # after WebSearcher was constructed (e.g. late dotenv loading).
        api_key = os.environ.get("TAVILY_API_KEY", "").strip() or self._tavily_key
        if api_key:
            logger.info(
                "Tavily search [topic=%s time_range=%s]: %r", topic, time_range, query
            )
            try:
                return self._search_tavily(
                    query, max_results, api_key=api_key,
                    topic=topic, time_range=time_range,
                )
            except Exception as exc:
                logger.error(
                    "Tavily search FAILED (%s: %s) — falling back to DuckDuckGo",
                    type(exc).__name__, exc,
                )

        # ── DuckDuckGo (last-resort fallback) ────────────────────────────────
        if self._has_ddg:
            logger.info("DuckDuckGo fallback search: %r", query)
            try:
                return self._search_ddg(query, max_results, time_range=time_range)
            except Exception as exc:
                logger.error("DuckDuckGo search FAILED (%s: %s)", type(exc).__name__, exc)

        logger.error(
            "No search backend available. "
            "Set TAVILY_API_KEY in .env or install ddgs."
        )
        return []

    def format_results(self, results: list[SearchResult]) -> str:
        """Format results as a numbered list for GPT context."""
        if not results:
            return "No search results found."
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            snippet = r.snippet[:MAX_SNIPPET_LEN]
            if len(r.snippet) > MAX_SNIPPET_LEN:
                snippet += "…"
            lines.append(f"{i}. **{r.title}**\n   {snippet}\n   Source: {r.url}")
        return "\n\n".join(lines)

    def search_and_format(self, query: str, max_results: int = 5) -> str:
        """Convenience: search + format in one call."""
        return self.format_results(self.search(query, max_results))

    # ── Startup diagnostics ───────────────────────────────────────────────────

    def _log_startup(self) -> None:
        if self._tavily_key:
            logger.info(
                "WebSearcher ready — PRIMARY: Tavily (key …%s)%s",
                self._tavily_key[-4:],
                "  fallback: DuckDuckGo" if self._has_ddg else "",
            )
        elif self._has_ddg:
            logger.warning(
                "WebSearcher ready — TAVILY_API_KEY not set; using DuckDuckGo only. "
                "Set TAVILY_API_KEY in .env for reliable real-time results."
            )
        else:
            logger.error(
                "WebSearcher: NO search backend available. "
                "Set TAVILY_API_KEY in .env and/or run: pip install ddgs"
            )

    # ── Tavily ────────────────────────────────────────────────────────────────

    def _search_tavily(
        self,
        query: str,
        max_results: int,
        *,
        api_key: str,
        topic: str,
        time_range: str | None,
    ) -> list[SearchResult]:
        from tavily import TavilyClient  # type: ignore

        client = TavilyClient(api_key=api_key)

        kwargs: dict = {
            "query":          query,
            "max_results":    max_results,
            "include_answer": "advanced",   # prepend AI-synthesised direct answer
        }
        # topic defaults to "general" in Tavily — only pass non-default values
        if topic in ("news", "finance"):
            kwargs["topic"] = topic
        if time_range:
            kwargs["time_range"] = time_range
        # For general queries, request deeper crawl
        if topic == "general":
            kwargs["search_depth"] = "advanced"

        resp = client.search(**kwargs)
        results: list[SearchResult] = []

        # Prepend the synthesised answer as result #0 when present
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

        logger.info(
            "Tavily returned %d results (direct_answer=%s)",
            len(results), bool(answer),
        )
        return results

    # ── DuckDuckGo ────────────────────────────────────────────────────────────

    def _search_ddg(
        self,
        query: str,
        max_results: int,
        *,
        time_range: str | None,
    ) -> list[SearchResult]:
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError:
            logger.error("ddgs not installed — run: pip install ddgs")
            return []

        ddg_timelimit = {"day": "d", "week": "w", "month": "m", "year": "y"}
        timelimit = ddg_timelimit.get(time_range or "")

        results: list[SearchResult] = []
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
                logger.warning("DDG backend=%s failed (%s: %s)", backend, type(exc).__name__, exc)

        return results

    @staticmethod
    def _check_ddg() -> bool:
        try:
            from ddgs import DDGS  # noqa: F401
            return True
        except ImportError:
            return False


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Reachy web search standalone test")
    parser.add_argument("query", nargs="*", help="Search query words")
    parser.add_argument("--debug", action="store_true", help="Show DEBUG log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    query = " ".join(args.query) if args.query else "Reachy Mini Pollen Robotics"
    topic, time_range = _classify_query(query)

    print(f"Query      : {query!r}")
    print(f"Topic      : {topic}  |  time_range: {time_range}")
    print(f"TAVILY_KEY : {'set (' + os.environ.get('TAVILY_API_KEY','')[-4:] + ')' if os.environ.get('TAVILY_API_KEY') else 'NOT SET — check .env'}")
    print()

    searcher = WebSearcher()
    print(f"Backend    : {searcher.backend}")
    print()

    results = searcher.search(query)
    if not results:
        print("No results returned.")
        sys.exit(1)

    print(searcher.format_results(results))
