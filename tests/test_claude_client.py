"""Tests for sentiment/claude_client.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentiment.claude_client import ClaudeClient, RateLimitedError


# ---------------------------------------------------------------------------
# Helpers — fake SDK event types
# ---------------------------------------------------------------------------

class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeAssistantMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]


class _FakeRateLimitEvent:
    pass


async def _events(*items):
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestClaudeClientSdkMissing:
    """SDK import fails → query() returns None without raising."""

    def test_returns_none_when_sdk_absent(self):
        with patch("sentiment.claude_client._SDK_AVAILABLE", False):
            client = ClaudeClient()
            assert client.query("test prompt") is None


class TestClaudeClientQuery:
    """SDK present — exercise the async path via mocks."""

    def _patch_sdk(self, events_iter):
        """Return a context manager that patches the SDK internals."""
        return patch.multiple(
            "sentiment.claude_client",
            _SDK_AVAILABLE=True,
            _sdk_query=MagicMock(return_value=events_iter),
            AssistantMessage=_FakeAssistantMessage,
            RateLimitEvent=_FakeRateLimitEvent,
            ClaudeCodeOptions=MagicMock(return_value=MagicMock()),
        )

    def test_returns_assistant_text(self):
        msg = _FakeAssistantMessage("bullish sentiment detected")
        with self._patch_sdk(_events(msg)):
            client = ClaudeClient()
            result = client.query("prompt")
        assert result == "bullish sentiment detected"

    def test_concatenates_multiple_blocks(self):
        msg = _FakeAssistantMessage.__new__(_FakeAssistantMessage)
        msg.content = [_FakeTextBlock("part one"), _FakeTextBlock("part two")]
        with self._patch_sdk(_events(msg)):
            client = ClaudeClient()
            result = client.query("prompt")
        assert result == "part one\npart two"

    def test_returns_none_on_rate_limit(self):
        rate_limit = _FakeRateLimitEvent()
        with self._patch_sdk(_events(rate_limit)):
            client = ClaudeClient()
            result = client.query("prompt")
        assert result is None

    def test_returns_none_when_no_assistant_message(self):
        with self._patch_sdk(_events()):
            client = ClaudeClient()
            result = client.query("prompt")
        assert result is None

    def test_returns_none_on_unexpected_exception(self):
        async def _bad_query(*args, **kwargs):
            raise RuntimeError("connection error")
            yield  # make it an async generator

        with patch.multiple(
            "sentiment.claude_client",
            _SDK_AVAILABLE=True,
            _sdk_query=_bad_query,
            ClaudeCodeOptions=MagicMock(return_value=MagicMock()),
        ):
            client = ClaudeClient()
            result = client.query("prompt")
        assert result is None

    def test_rate_limit_event_before_message(self):
        """RateLimitEvent anywhere in the stream causes fallback (None)."""
        msg = _FakeAssistantMessage("never returned")
        rate_limit = _FakeRateLimitEvent()
        with self._patch_sdk(_events(rate_limit, msg)):
            client = ClaudeClient()
            result = client.query("prompt")
        assert result is None
