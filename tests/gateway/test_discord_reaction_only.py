"""Regression tests for Discord native reaction-only responses."""

from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, ProcessingOutcome
from gateway.session import SessionSource
from gateway.session_context import clear_session_vars, set_session_vars


def _ensure_discord_mock():
    if "discord" in sys.modules and hasattr(sys.modules["discord"], "__file__"):
        return

    discord_mod = MagicMock()
    discord_mod.Intents.default.return_value = MagicMock()
    discord_mod.DMChannel = type("DMChannel", (), {})
    discord_mod.Thread = type("Thread", (), {})
    discord_mod.ForumChannel = type("ForumChannel", (), {})
    discord_mod.Interaction = object
    discord_mod.app_commands = SimpleNamespace(
        describe=lambda **kwargs: (lambda fn: fn),
        choices=lambda **kwargs: (lambda fn: fn),
        Choice=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    ext_mod = MagicMock()
    commands_mod = MagicMock()
    commands_mod.Bot = MagicMock
    ext_mod.commands = commands_mod

    sys.modules.setdefault("discord", discord_mod)
    sys.modules.setdefault("discord.ext", ext_mod)
    sys.modules.setdefault("discord.ext.commands", commands_mod)


_ensure_discord_mock()

from plugins.platforms.discord.adapter import DiscordAdapter  # noqa: E402


def _event(raw_message) -> MessageEvent:
    return MessageEvent(
        text="react only",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="chan-1",
            chat_type="group",
            user_id="user-1",
            user_name="User",
        ),
        raw_message=raw_message,
        message_id="msg-1",
    )


@pytest.mark.asyncio
async def test_respond_with_reaction_adds_native_discord_reaction_only():
    from gateway.reaction_only import (
        begin_reaction_scope,
        reset_reaction_scope,
        respond_with_reaction_tool,
    )

    raw_message = SimpleNamespace(add_reaction=AsyncMock())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    event = _event(raw_message)
    session_tokens = set_session_vars(platform="discord", chat_id="chan-1", message_id="msg-1")
    scope_token = begin_reaction_scope(event=event, adapter=adapter, loop=asyncio.get_running_loop())
    try:
        result_text = await asyncio.to_thread(
            respond_with_reaction_tool,
            {"emoji": "👍", "reason": "acknowledge"},
        )
    finally:
        reset_reaction_scope(scope_token)
        clear_session_vars(session_tokens)

    result = json.loads(result_text)
    assert result == {
        "success": True,
        "mode": "reaction_only",
        "emoji": "👍",
        "message": "Native Discord reaction added; do not send any text response for this turn.",
    }
    raw_message.add_reaction.assert_awaited_once_with("👍")
    assert getattr(event, "_hermes_reaction_only") is True


@pytest.mark.asyncio
async def test_discord_completion_reaction_suppressed_after_reaction_only():
    raw_message = SimpleNamespace(add_reaction=AsyncMock(), remove_reaction=AsyncMock())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(user=SimpleNamespace(id=999))
    event = _event(raw_message)
    event._hermes_reaction_only = True

    await adapter.on_processing_complete(event, ProcessingOutcome.SUCCESS)

    raw_message.remove_reaction.assert_awaited_once_with("👀", adapter._client.user)
    raw_message.add_reaction.assert_not_awaited()


def test_reaction_tool_schema_only_appears_for_live_discord_session(monkeypatch):
    from gateway.reaction_only import begin_reaction_scope, reset_reaction_scope
    from model_tools import _clear_tool_defs_cache, get_tool_definitions

    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    raw_message = SimpleNamespace(add_reaction=AsyncMock())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    event = _event(raw_message)

    _clear_tool_defs_cache()
    no_session_tools = get_tool_definitions(["messaging"], quiet_mode=True)
    assert "respond_with_reaction" not in {t["function"]["name"] for t in no_session_tools}

    session_tokens = set_session_vars(platform="discord", chat_id="chan-1", message_id="msg-1")
    scope_token = begin_reaction_scope(event=event, adapter=adapter, loop=None)
    try:
        _clear_tool_defs_cache()
        tools = get_tool_definitions(["messaging"], quiet_mode=True)
    finally:
        reset_reaction_scope(scope_token)
        clear_session_vars(session_tokens)
        _clear_tool_defs_cache()

    assert "respond_with_reaction" in {t["function"]["name"] for t in tools}
    assert "discord" not in {t["function"]["name"] for t in tools}
    assert "discord_admin" not in {t["function"]["name"] for t in tools}

    _clear_tool_defs_cache()
    session_tokens = set_session_vars(platform="discord", chat_id="chan-1", message_id="msg-1")
    scope_token = begin_reaction_scope(event=event, adapter=adapter, loop=None)
    try:
        live_tools = get_tool_definitions(["messaging"], quiet_mode=True)
        assert "respond_with_reaction" in {t["function"]["name"] for t in live_tools}
    finally:
        reset_reaction_scope(scope_token)
        clear_session_vars(session_tokens)

    # Regression guard: quiet-mode tool-definition caching must not leak the
    # live Discord-only reaction tool into later non-Discord/non-root contexts.
    cached_after_scope = get_tool_definitions(["messaging"], quiet_mode=True)
    assert "respond_with_reaction" not in {t["function"]["name"] for t in cached_after_scope}
    _clear_tool_defs_cache()


def test_mixed_reaction_tool_batch_executes_no_tools():
    from agent.tool_executor import execute_tool_calls_sequential

    def tool_call(name: str, args: dict, call_id: str):
        return SimpleNamespace(
            id=call_id,
            function=SimpleNamespace(name=name, arguments=json.dumps(args)),
        )

    assistant_message = SimpleNamespace(
        tool_calls=[
            tool_call("respond_with_reaction", {"emoji": "👍"}, "call-react"),
            tool_call("terminal", {"command": "echo should-not-run"}, "call-terminal"),
        ]
    )
    agent = SimpleNamespace(_interrupt_requested=False)
    messages: list[dict] = []

    execute_tool_calls_sequential(agent, assistant_message, messages, "task-1")

    assert len(messages) == 2
    assert {m["tool_call_id"] for m in messages} == {"call-react", "call-terminal"}
    assert all("mixed tool batches execute no tools" in m["content"] for m in messages)


def test_reaction_only_completed_flag_resets_at_turn_start(monkeypatch):
    from agent.conversation_loop import run_conversation

    class Guardrails:
        def reset_for_turn(self):
            raise RuntimeError("stop after turn-reset block")

    agent = SimpleNamespace(
        _reaction_only_completed=True,
        session_id="sid",
        provider="",
        model="",
        _memory_write_origin="assistant_tool",
        _ensure_db_session=lambda: None,
        _restore_primary_runtime=lambda: None,
        _tool_guardrails=Guardrails(),
    )

    with pytest.raises(RuntimeError, match="stop after turn-reset block"):
        run_conversation(agent, "hello")

    assert agent._reaction_only_completed is False


def test_reaction_tool_fails_after_visible_output_committed():
    from gateway.reaction_only import (
        begin_reaction_scope,
        mark_visible_delivery,
        reset_reaction_scope,
        respond_with_reaction_tool,
    )

    raw_message = SimpleNamespace(add_reaction=AsyncMock())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    event = _event(raw_message)
    session_tokens = set_session_vars(platform="discord", chat_id="chan-1", message_id="msg-1")
    scope_token = begin_reaction_scope(event=event, adapter=adapter, loop=None)
    try:
        mark_visible_delivery("status:thinking")
        result = json.loads(respond_with_reaction_tool({"emoji": "👍"}))
    finally:
        reset_reaction_scope(scope_token)
        clear_session_vars(session_tokens)

    assert result["success"] is False
    assert "visible output" in result["error"]
    assert result["visible_delivery_reasons"] == ["status:thinking"]
    raw_message.add_reaction.assert_not_called()


def test_visible_delivery_tool_marks_reaction_state_fail_closed():
    from gateway.reaction_only import (
        begin_reaction_scope,
        mark_visible_tool_delivery,
        reaction_delivery_state_dict,
        reset_reaction_scope,
    )

    raw_message = SimpleNamespace(add_reaction=AsyncMock())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    event = _event(raw_message)
    scope_token = begin_reaction_scope(event=event, adapter=adapter, loop=None)
    try:
        mark_visible_tool_delivery("discord", {"action": "send_message", "content": "hello"})
        state = reaction_delivery_state_dict()
    finally:
        reset_reaction_scope(scope_token)

    assert state["mode"] == "visible"
    assert state["visible_delivery_committed"] is True
    assert state["visible_delivery_reasons"] == ["visible_tool:discord.send_message"]


@pytest.mark.asyncio
async def test_auto_thread_seed_fallback_suppressed_for_reaction_capable_message():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    message = SimpleNamespace(
        content="react-capable",
        add_reaction=AsyncMock(),
        create_thread=AsyncMock(side_effect=RuntimeError("direct create failed")),
        channel=SimpleNamespace(send=AsyncMock()),
        author=SimpleNamespace(display_name="Jezza"),
    )

    result = await adapter._auto_create_thread(message)

    assert result is None
    message.create_thread.assert_awaited_once()
    message.channel.send.assert_not_awaited()


@pytest.mark.asyncio
async def test_reaction_only_state_and_tool_result_are_transcript_safe_json():
    from agent.tool_dispatch_helpers import make_tool_result_message
    from gateway.reaction_only import (
        begin_reaction_scope,
        reaction_delivery_state_dict,
        reset_reaction_scope,
        respond_with_reaction_tool,
    )

    raw_message = SimpleNamespace(add_reaction=AsyncMock())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    event = _event(raw_message)
    session_tokens = set_session_vars(platform="discord", chat_id="chan-1", message_id="msg-1")
    scope_token = begin_reaction_scope(event=event, adapter=adapter, loop=asyncio.get_running_loop())
    try:
        result_text = await asyncio.to_thread(
            respond_with_reaction_tool,
            {"emoji": "✅", "reason": "complete"},
        )
        state = reaction_delivery_state_dict()
        tool_message = make_tool_result_message(
            "respond_with_reaction",
            result_text,
            "call-reaction",
        )
    finally:
        reset_reaction_scope(scope_token)
        clear_session_vars(session_tokens)

    # These are the artifacts persisted in the normal transcript/tool trace
    # path.  They must stay JSON-only and contain no raw Discord event/adapter
    # objects, so no DB migration is needed for reaction-only turns.
    json.dumps(state, ensure_ascii=False)
    json.dumps(tool_message, ensure_ascii=False)
    assert state == {
        "mode": "reaction_only",
        "visible_delivery_committed": False,
        "visible_delivery_reasons": [],
        "reaction_emoji": "✅",
        "reaction_reason": "complete",
        "reaction_success": True,
    }
    assert tool_message["role"] == "tool"
    assert tool_message["name"] == "respond_with_reaction"
    assert json.loads(tool_message["content"])["mode"] == "reaction_only"


@pytest.mark.asyncio
async def test_reaction_tool_fails_fast_on_gateway_event_loop_thread():
    from gateway.reaction_only import (
        begin_reaction_scope,
        reset_reaction_scope,
        respond_with_reaction_tool,
    )

    raw_message = SimpleNamespace(add_reaction=AsyncMock())
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    event = _event(raw_message)
    session_tokens = set_session_vars(platform="discord", chat_id="chan-1", message_id="msg-1")
    scope_token = begin_reaction_scope(event=event, adapter=adapter, loop=asyncio.get_running_loop())
    try:
        result = json.loads(respond_with_reaction_tool({"emoji": "👍"}))
    finally:
        reset_reaction_scope(scope_token)
        clear_session_vars(session_tokens)

    assert result["success"] is False
    assert "event-loop thread" in result["error"]
    raw_message.add_reaction.assert_not_awaited()
