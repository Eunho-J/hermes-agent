"""Session-scoped Discord reaction-only response state.

This module keeps the cross-layer state for turns where the agent chooses to
answer by adding a native reaction to the triggering Discord message instead of
sending visible text/media output.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional


_REACTION_ONLY_TOOL = "respond_with_reaction"
_STATE: ContextVar["ReactionDeliveryState | None"] = ContextVar(
    "HERMES_REACTION_DELIVERY_STATE",
    default=None,
)
_REACTION_TOOL_ALLOWED: ContextVar[bool] = ContextVar(
    "HERMES_REACTION_TOOL_ALLOWED",
    default=True,
)


@dataclass
class ReactionDeliveryState:
    """Mutable per-turn delivery state shared by gateway and agent tool loop."""

    event: Any = None
    adapter: Any = None
    loop: Any = None
    mode: Optional[str] = None  # None | "reaction_only" | "visible"
    visible_delivery_committed: bool = False
    visible_delivery_reasons: list[str] = field(default_factory=list)
    reaction_emoji: Optional[str] = None
    reaction_reason: Optional[str] = None
    reaction_success: bool = False
    reaction_error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        data = {
            "mode": self.mode,
            "visible_delivery_committed": self.visible_delivery_committed,
            "visible_delivery_reasons": list(self.visible_delivery_reasons),
        }
        if self.reaction_emoji is not None:
            data["reaction_emoji"] = self.reaction_emoji
        if self.reaction_reason:
            data["reaction_reason"] = self.reaction_reason
        if self.reaction_success:
            data["reaction_success"] = True
        if self.reaction_error:
            data["reaction_error"] = self.reaction_error
        return data


def begin_reaction_scope(*, event: Any, adapter: Any, loop: Any = None):
    """Install a reaction delivery state for the current gateway turn."""
    state = ReactionDeliveryState(event=event, adapter=adapter, loop=loop)
    return _STATE.set(state)


def reset_reaction_scope(token) -> None:
    _STATE.reset(token)


def get_reaction_state() -> ReactionDeliveryState | None:
    return _STATE.get()


def _platform_name(event: Any = None) -> str:
    try:
        source = getattr(event, "source", None)
        platform = getattr(source, "platform", None)
        return str(getattr(platform, "value", platform) or "").lower()
    except Exception:
        return ""


def reaction_capable() -> bool:
    if not _REACTION_TOOL_ALLOWED.get():
        return False
    try:
        from gateway.session_context import get_session_env
        if get_session_env("HERMES_SESSION_PLATFORM", "").lower() != "discord":
            return False
        if not get_session_env("HERMES_SESSION_MESSAGE_ID", ""):
            return False
    except Exception:
        return False
    state = get_reaction_state()
    if state is None:
        return False
    event = state.event
    if _platform_name(event) != "discord":
        return False
    raw = getattr(event, "raw_message", None)
    if raw is not None and hasattr(raw, "add_reaction"):
        return True
    return bool(getattr(event, "message_id", None))


def mark_visible_delivery(reason: str) -> None:
    """Record that this turn already produced user-visible output."""
    state = get_reaction_state()
    if state is None:
        return
    state.visible_delivery_committed = True
    if reason and reason not in state.visible_delivery_reasons:
        state.visible_delivery_reasons.append(reason)
    if state.mode is None:
        state.mode = "visible"


def mark_visible_tool_delivery(function_name: str, args: dict[str, Any] | None = None) -> None:
    """Record model-requested tools that can visibly deliver in the gateway turn.

    Tool-progress bubbles are optional, so the reaction-only fail-closed
    contract cannot rely on progress callbacks alone.  This helper marks
    user-visible delivery intent for tools that send/create platform-visible
    artifacts directly.
    """
    args = args or {}
    name = str(function_name or "")
    if name == "send_message":
        if str(args.get("action", "send") or "send").lower() != "list":
            mark_visible_delivery("visible_tool:send_message")
        return
    if name in {"tts", "speak"}:
        mark_visible_delivery(f"visible_tool:{name}")
        return
    if name in {"discord", "discord_admin"}:
        action = str(args.get("action") or "").lower()
        if action in {
            "send_message",
            "create_thread",
            "create_forum_post",
            "pin_message",
            "unpin_message",
            "delete_message",
            "add_role",
            "remove_role",
        }:
            mark_visible_delivery(f"visible_tool:{name}.{action}")


def has_visible_delivery() -> bool:
    state = get_reaction_state()
    return bool(state and state.visible_delivery_committed)


def is_reaction_only() -> bool:
    state = get_reaction_state()
    return bool(state and state.mode == "reaction_only" and state.reaction_success)


def reaction_delivery_state_dict() -> dict[str, Any] | None:
    state = get_reaction_state()
    return state.as_dict() if state is not None else None


def set_reaction_tool_allowed(value: bool):
    return _REACTION_TOOL_ALLOWED.set(bool(value))


def reset_reaction_tool_allowed(token) -> None:
    _REACTION_TOOL_ALLOWED.reset(token)


_CUSTOM_DISCORD_EMOJI_RE = re.compile(r"^<a?:[A-Za-z0-9_]{2,32}:\d{2,25}>$")


def _normalize_emoji(value: Any) -> str:
    emoji = str(value or "").strip()
    if not emoji:
        raise ValueError("emoji is required")
    # Discord accepts either unicode emoji or a custom emoji mention. Avoid
    # multi-token text masquerading as a reaction payload.
    if _CUSTOM_DISCORD_EMOJI_RE.match(emoji):
        return emoji
    if any(ch.isspace() for ch in emoji):
        raise ValueError("emoji must be a single Discord reaction emoji")
    if len(emoji) > 64:
        raise ValueError("emoji is too long for a Discord reaction")
    return emoji


def respond_with_reaction_tool(args: dict[str, Any] | None = None) -> str:
    """Synchronous agent-loop tool implementation."""
    args = args or {}
    state = get_reaction_state()
    if state is None or not reaction_capable():
        return json.dumps(
            {
                "success": False,
                "error": "respond_with_reaction is only available for live Discord gateway message turns.",
            },
            ensure_ascii=False,
        )
    if state.visible_delivery_committed:
        return json.dumps(
            {
                "success": False,
                "error": "Cannot use reaction-only response after visible output was already delivered.",
                "visible_delivery_reasons": state.visible_delivery_reasons,
            },
            ensure_ascii=False,
        )

    try:
        emoji = _normalize_emoji(args.get("emoji"))
    except ValueError as exc:
        state.reaction_error = str(exc)
        return json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False)

    adapter = state.adapter
    if adapter is None or not hasattr(adapter, "send_reaction"):
        state.reaction_error = "Discord adapter does not support send_reaction"
        return json.dumps({"success": False, "error": state.reaction_error}, ensure_ascii=False)

    try:
        loop = state.loop
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
        if loop is not None and loop.is_running():
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop is loop:
                state.reaction_error = (
                    "Cannot synchronously add a Discord reaction from the gateway event-loop thread"
                )
                return json.dumps(
                    {
                        "success": False,
                        "error": state.reaction_error,
                    },
                    ensure_ascii=False,
                )

        coro = adapter.send_reaction(state.event, emoji)
        if asyncio.iscoroutine(coro) or hasattr(coro, "__await__"):
            if loop is not None and loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
                ok = bool(fut.result(timeout=15))
            else:
                ok = bool(asyncio.run(coro))
        else:
            ok = bool(coro)
    except Exception as exc:  # pragma: no cover - defensive path
        state.reaction_error = str(exc)
        return json.dumps({"success": False, "error": f"Failed to add Discord reaction: {exc}"}, ensure_ascii=False)

    if not ok:
        state.reaction_error = "Discord rejected the reaction request"
        return json.dumps({"success": False, "error": state.reaction_error}, ensure_ascii=False)

    state.mode = "reaction_only"
    state.reaction_emoji = emoji
    reason = str(args.get("reason") or "").strip()
    state.reaction_reason = reason or None
    state.reaction_success = True
    try:
        setattr(state.event, "_hermes_reaction_only", True)
        setattr(state.event, "_hermes_reaction_emoji", emoji)
    except Exception:
        pass
    return json.dumps(
        {
            "success": True,
            "mode": "reaction_only",
            "emoji": emoji,
            "message": "Native Discord reaction added; do not send any text response for this turn.",
        },
        ensure_ascii=False,
    )


RESPOND_WITH_REACTION_SCHEMA = {
    "type": "function",
    "function": {
        "name": _REACTION_ONLY_TOOL,
        "description": (
            "End the current live Discord gateway turn by adding a native reaction "
            "to the triggering message instead of sending a text reply. Use only "
            "when an emoji reaction is sufficient and no visible response is needed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "emoji": {
                    "type": "string",
                    "description": "A single Unicode emoji or Discord custom emoji mention to add as a reaction.",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional private rationale for the transcript/tool trace.",
                },
                "target": {
                    "type": "string",
                    "enum": ["trigger"],
                    "description": "Reaction target. Only the triggering message is supported.",
                },
            },
            "required": ["emoji"],
            "additionalProperties": False,
        },
    },
}

__all__ = [
    "RESPOND_WITH_REACTION_SCHEMA",
    "begin_reaction_scope",
    "get_reaction_state",
    "has_visible_delivery",
    "is_reaction_only",
    "mark_visible_tool_delivery",
    "mark_visible_delivery",
    "reaction_capable",
    "reaction_delivery_state_dict",
    "reset_reaction_scope",
    "reset_reaction_tool_allowed",
    "respond_with_reaction_tool",
    "set_reaction_tool_allowed",
]
