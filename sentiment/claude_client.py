"""Claude subscription client for sentiment queries.

Wraps claude_agent_sdk.query() which uses the user's Claude subscription
without requiring ANTHROPIC_API_KEY.  Falls back transparently to None
(deterministic-only mode) on RateLimitEvent or ImportError.

Usage:
    client = ClaudeClient()
    text = client.query("Summarise the market mood for BTC/USDT: ...")
    if text is None:
        # rate-limited or SDK absent — caller uses deterministic signals only
        ...
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional SDK import — absent in environments without a Claude subscription
# ---------------------------------------------------------------------------

try:
    from claude_agent_sdk import (  # type: ignore[import-not-found]
        ClaudeAgentOptions,
        query as _sdk_query,
    )
    from claude_agent_sdk.types import (  # type: ignore[import-not-found]
        AssistantMessage,
        RateLimitEvent,
    )
    _SDK_AVAILABLE = True
except ImportError:
    _sdk_query = None  # type: ignore[assignment]
    ClaudeAgentOptions = None  # type: ignore[assignment]
    AssistantMessage = None  # type: ignore[assignment]
    RateLimitEvent = None  # type: ignore[assignment]
    _SDK_AVAILABLE = False


class RateLimitedError(Exception):
    """Raised internally when the SDK emits a RateLimitEvent."""


def _suppress_asyncgen_aclose(
    loop: asyncio.AbstractEventLoop, context: dict[str, Any]
) -> None:
    """Loop exception handler that swallows the benign async-generator cleanup
    error.

    Breaking out of the SDK stream on a RateLimitEvent forces aclose() on a
    generator whose internal task is still running; asyncio's finalizer reports
    "aclose(): asynchronous generator is already running" through the loop
    exception handler (NOT a catchable `except` at the call site). Swallow
    exactly that; everything else falls through to the default handler.
    """
    exc = context.get("exception")
    if isinstance(exc, RuntimeError) and "asynchronous generator is already running" in str(
        exc
    ):
        return
    loop.default_exception_handler(context)


class ClaudeClient:
    """Sync wrapper around the async claude_code_sdk for sentiment queries.

    Instantiate once per process; each call to ``query()`` opens a new
    SDK session (max_turns=1) and extracts the assistant text.
    """

    def __init__(self, max_turns: int = 1) -> None:
        self._max_turns = max_turns

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def query(self, prompt: str) -> str | None:
        """Query Claude and return the full assistant text, or None on failure.

        Returns None (not an exception) in two situations:
        - SDK is not installed / no subscription found
        - Claude emits a RateLimitEvent (caller falls back to deterministic)
        """
        if not _SDK_AVAILABLE:
            _log.warning("claude_code_sdk not installed; sentiment unavailable")
            return None
        try:
            return asyncio.run(self._async_query(prompt))
        except RateLimitedError:
            _log.warning("Claude rate limit reached; falling back to deterministic-only")
            return None
        except Exception as exc:  # noqa: BLE001
            _log.error("claude_client.query failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _async_query(self, prompt: str) -> str | None:
        # The aclose-on-rate-limit cleanup error surfaces via the loop exception
        # handler (asyncio's async-gen finalizer), not as a catchable except at
        # the break site, so install a handler that swallows exactly that.
        asyncio.get_running_loop().set_exception_handler(_suppress_asyncgen_aclose)
        parts: list[str] = []
        rate_limited = False
        async for event in _sdk_query(  # type: ignore[misc]
            prompt=prompt,
            options=ClaudeAgentOptions(max_turns=self._max_turns),
        ):
            if RateLimitEvent is not None and isinstance(event, RateLimitEvent):
                rate_limited = True
                break
            if AssistantMessage is not None and isinstance(event, AssistantMessage):
                for block in event.content:
                    text = getattr(block, "text", None)
                    if text:
                        parts.append(text)
        if rate_limited:
            raise RateLimitedError
        return "\n".join(parts) if parts else None
