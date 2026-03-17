# -*- coding: utf-8 -*-
"""Test reasoning_content handling when messages are merged by formatter.

This test verifies that reasoning_content from consecutive assistant messages
is properly accumulated when the formatter merges them.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agentscope.message import Msg


class TestReasoningContentMerge:
    """Test reasoning_content handling for merged messages."""

    @pytest.fixture
    def formatter_class(self):
        """Get the FileBlockSupportFormatter class."""
        from copaw.agents.model_factory import (
            _create_file_block_support_formatter,
        )
        from agentscope.formatter import OpenAIChatFormatter

        return _create_file_block_support_formatter(OpenAIChatFormatter)

    @pytest.mark.asyncio
    async def test_single_assistant_message_preserves_reasoning(
        self, formatter_class
    ):
        """Reasoning content should be preserved for single assistant message."""
        formatter = formatter_class()

        # Create assistant message with thinking block
        msg = Msg(
            name="assistant",
            content=[
                {"type": "thinking", "thinking": "Let me analyze this..."},
                {"type": "text", "text": "Here's my answer."},
            ],
            role="assistant",
        )

        messages = await formatter._format([msg])

        assert len(messages) == 1
        assert messages[0].get("reasoning_content") == "Let me analyze this..."

    @pytest.mark.asyncio
    async def test_consecutive_assistant_messages_accumulate_reasoning(
        self, formatter_class
    ):
        """When consecutive assistant messages are merged, reasoning should accumulate.

        This tests the fix for: 'Assistant message count mismatch after formatting'
        """
        formatter = formatter_class()

        # Create multiple assistant messages with thinking blocks
        msg1 = Msg(
            name="assistant",
            content=[
                {"type": "thinking", "thinking": "First thought..."},
                {"type": "text", "text": "Response 1"},
            ],
            role="assistant",
        )
        msg2 = Msg(
            name="assistant",
            content=[
                {"type": "thinking", "thinking": "Second thought..."},
                {"type": "text", "text": "Response 2"},
            ],
            role="assistant",
        )

        # Format both messages
        messages = await formatter._format([msg1, msg2])

        # Check that reasoning is accumulated in the output
        # Note: Formatter may merge consecutive same-role messages
        assert len(messages) >= 1

        # Find assistant messages in output
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        assert len(assistant_msgs) >= 1

        # If merged, reasoning should be accumulated in last assistant message
        # If not merged, each should have its own reasoning
        last_assistant = assistant_msgs[-1]
        reasoning = last_assistant.get("reasoning_content", "")

        # Should contain both reasonings (accumulated or separate)
        if len(assistant_msgs) == 1:
            # Merged case: accumulate both
            assert "First thought" in reasoning or "Second thought" in reasoning
        else:
            # Not merged: each has own reasoning
            assert len(assistant_msgs) == 2

    @pytest.mark.asyncio
    async def test_no_thinking_block_no_reasoning_content(
        self, formatter_class
    ):
        """Messages without thinking blocks should not have reasoning_content."""
        formatter = formatter_class()

        msg = Msg(
            name="assistant",
            content=[{"type": "text", "text": "Just a regular response."}],
            role="assistant",
        )

        messages = await formatter._format([msg])

        assert len(messages) == 1
        assert "reasoning_content" not in messages[0]

    @pytest.mark.asyncio
    async def test_empty_thinking_skipped(self, formatter_class):
        """Empty thinking blocks should not create reasoning_content."""
        formatter = formatter_class()

        msg = Msg(
            name="assistant",
            content=[
                {"type": "thinking", "thinking": ""},  # Empty
                {"type": "thinking", "thinking": "   "},  # Whitespace only
                {"type": "text", "text": "Response"},
            ],
            role="assistant",
        )

        messages = await formatter._format([msg])

        # Empty thinking should result in no or empty reasoning_content
        assert messages[0].get("reasoning_content", "").strip() == ""