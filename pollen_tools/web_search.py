"""
Tavily web search tool for reachy_mini_conversation_app.

Deployment (on the Reachy Mini robot):
---------------------------------------
1. Copy this file to:
       <app_root>/external_content/external_tools/web_search.py

2. Add to <app_root>/.env:
       TAVILY_API_KEY=<your key>
       REACHY_MINI_EXTERNAL_TOOLS_DIRECTORY=external_content/external_tools
       AUTOLOAD_EXTERNAL_TOOLS=1

3. Optionally add "web_search" to the active profile's tools.txt (not needed
   when AUTOLOAD_EXTERNAL_TOOLS=1 is set).

4. Install tavily-python on the robot if not already present:
       pip install tavily-python

The tool triggers automatically whenever the LLM decides a web search is needed
(current events, prices, weather, sports, etc.).  No other files are changed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------

_FINANCE_RE = re.compile(
    r"\b(price|preis|kurs|wert|bitcoin|btc|eth|ethereum|crypto|kryptow[aä]hrung"
    r"|aktie|stock|nasdaq|dax|dow|s&p|gold|silver|silber|forex|dollar"
    r"|euro|yen|inflation|zinsen|interest.rate)\b",
    re.IGNORECASE,
)

_NEWS_RE = re.compile(
    r"\b(wetter|weather|temperatur|temperature|forecast|vorhersage"
    r"|news|nachrichten|aktuell|current|today|heute|jetzt|now|live"
    r"|sport|score|ergebnis|result|fu[sß]ball|football|soccer|election|wahl)\b",
    re.IGNORECASE,
)


def _classify(query: str) -> tuple[str, str | None]:
    """Return (tavily_topic, time_range)."""
    if _FINANCE_RE.search(query):
        return "finance", "day"
    if _NEWS_RE.search(query):
        return "news", "day"
    return "general", None


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------

class WebSearch(Tool):
    """Search the internet for current, real-time information."""

    name = "web_search"
    description = (
        "Search the internet for current, real-time information. "
        "ALWAYS use this tool for: current events, cryptocurrency prices (Bitcoin, "
        "Ethereum, etc.), stock prices, currency exchange rates, weather forecasts, "
        "sports scores, or any fact that may have changed since training. "
        "Do NOT answer these topics from memory — call this tool first."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    async def __call__(
        self, deps: ToolDependencies, *, query: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        if not query or not query.strip():
            return {"error": "query must not be empty"}

        api_key = os.environ.get("TAVILY_API_KEY", "").strip()
        if not api_key:
            logger.warning("web_search: TAVILY_API_KEY not set")
            return {"error": "TAVILY_API_KEY is not configured. Add it to .env on the robot."}

        # Run the synchronous Tavily client off the event-loop thread so we
        # don't block the async conversation pipeline.
        return await asyncio.to_thread(self._search_sync, query, api_key)

    # ------------------------------------------------------------------

    def _search_sync(self, query: str, api_key: str) -> Dict[str, Any]:
        """Blocking Tavily search — called via asyncio.to_thread()."""
        try:
            from tavily import TavilyClient  # type: ignore
        except ImportError:
            return {
                "error": (
                    "tavily-python is not installed on the robot. "
                    "Run: pip install tavily-python"
                )
            }

        topic, time_range = _classify(query)
        logger.info("web_search: query=%r  topic=%s  time_range=%s", query, topic, time_range)

        search_kwargs: Dict[str, Any] = {
            "query": query,
            "max_results": 5,
            "include_answer": "advanced",
        }
        if topic in ("finance", "news"):
            search_kwargs["topic"] = topic
        if time_range:
            search_kwargs["time_range"] = time_range
        if topic == "general":
            search_kwargs["search_depth"] = "advanced"

        try:
            client = TavilyClient(api_key=api_key)
            resp = client.search(**search_kwargs)
        except Exception as exc:
            logger.error("web_search: Tavily error: %s", exc)
            return {"error": f"Search failed: {exc}"}

        parts: list[str] = []

        # Prepend AI-synthesised direct answer when present
        answer = (resp.get("answer") or "").strip()
        if answer:
            parts.append(f"Summary: {answer}")

        for i, r in enumerate(resp.get("results", [])[:4], 1):
            title   = r.get("title", "").strip()
            snippet = (r.get("content") or "").strip()[:400]
            url     = r.get("url", "")
            if snippet:
                parts.append(f"{i}. {title}\n   {snippet}\n   Source: {url}")

        if not parts:
            return {"result": "No results found for this query."}

        result_text = "\n\n".join(parts)
        logger.info("web_search: returned %d results", len(resp.get("results", [])))
        return {"result": result_text}
