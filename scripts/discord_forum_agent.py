#!/usr/bin/env python3
"""Monitor a Discord forum channel and run Hermes on unclaimed posts.

The monitor is intentionally small and stateful:
- list active/recently archived forum posts;
- skip posts already recorded in state;
- skip posts that already contain a bot claim for this agent;
- post a claim, run Hermes once with the post transcript, then post the result;
- persist completion/failure state so cron reruns do not duplicate work.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home

DISCORD_API_BASE = "https://discord.com/api/v10"
DEFAULT_STATE_PATH = get_hermes_home() / "discord-forum-agent-state.json"


def _read_dotenv_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key and value.strip():
            return value.strip().strip('"').strip("'")
    return ""


def get_bot_token() -> str:
    token = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    if not token:
        token = _read_dotenv_value(get_hermes_home() / ".env", "DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN not configured")
    return token


def discord_request(
    method: str,
    path: str,
    token: str,
    params: dict[str, str] | None = None,
    body: dict[str, Any] | None = None,
    timeout: int = 30,
) -> Any:
    url = f"{DISCORD_API_BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bot {token}",
            "Content-Type": "application/json",
            "User-Agent": "Hermes-Agent Discord Forum Monitor",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 204:
                return None
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Discord API error {exc.code}: {raw}") from exc


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"threads": {}}
    return json.loads(path.read_text())


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    tmp.replace(path)


def list_forum_threads(token: str, forum_channel_id: str, limit: int = 100) -> list[dict[str, Any]]:
    channel = discord_request("GET", f"/channels/{forum_channel_id}", token)
    guild_id = channel.get("guild_id") if isinstance(channel, dict) else None
    if not guild_id:
        raise RuntimeError(f"Discord channel {forum_channel_id} has no guild_id")
    active = discord_request("GET", f"/guilds/{guild_id}/threads/active", token)
    archived = discord_request(
        "GET",
        f"/channels/{forum_channel_id}/threads/archived/public",
        token,
        params={"limit": str(max(1, min(limit, 100)))},
    )
    seen: set[str] = set()
    threads: list[dict[str, Any]] = []
    for payload in (active, archived):
        for thread in (payload or {}).get("threads", []):
            thread_id = str(thread.get("id") or "")
            if not thread_id or thread_id in seen or str(thread.get("parent_id") or "") != str(forum_channel_id):
                continue
            seen.add(thread_id)
            threads.append(thread)
    return threads


def forum_tag_ids_by_name(token: str, forum_channel_id: str, names: set[str]) -> set[str]:
    """Return forum tag IDs whose names match ``names`` case-insensitively."""
    channel = discord_request("GET", f"/channels/{forum_channel_id}", token)
    wanted = {name.lower() for name in names}
    matched: set[str] = set()
    for tag in (channel or {}).get("available_tags", []) or []:
        tag_name = str(tag.get("name") or "").strip().lower()
        tag_id = str(tag.get("id") or "").strip()
        if tag_name in wanted and tag_id:
            matched.add(tag_id)
    return matched


def fetch_messages(token: str, thread_id: str, limit: int = 50) -> list[dict[str, Any]]:
    messages = discord_request(
        "GET",
        f"/channels/{thread_id}/messages",
        token,
        params={"limit": str(max(1, min(limit, 100)))},
    )
    return list(reversed(messages or []))


def has_agent_claim(messages: list[dict[str, Any]], agent_name: str) -> bool:
    needle = f"claimed by {agent_name}".lower()
    for msg in messages:
        author = msg.get("author") or {}
        if not author.get("bot"):
            continue
        if needle in str(msg.get("content") or "").lower():
            return True
    return False


def select_claimable_threads(
    threads: list[dict[str, Any]],
    state: dict[str, Any],
    fetcher: Callable[[str], list[dict[str, Any]]],
    *,
    agent_name: str,
    excluded_tag_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    processed = state.setdefault("threads", {})
    excluded_tag_ids = excluded_tag_ids or set()
    claimable: list[dict[str, Any]] = []
    for thread in threads:
        thread_id = str(thread.get("id") or "")
        if not thread_id or processed.get(thread_id, {}).get("status") in {"claimed", "completed"}:
            continue
        thread_tag_ids = {str(tag_id) for tag_id in (thread.get("applied_tags") or [])}
        if thread_tag_ids & excluded_tag_ids:
            continue
        if has_agent_claim(fetcher(thread_id), agent_name):
            continue
        claimable.append(thread)
    return claimable


def send_message(token: str, channel_id: str, content: str) -> dict[str, Any]:
    return discord_request("POST", f"/channels/{channel_id}/messages", token, body={"content": content})


def format_transcript(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for msg in messages:
        author = (msg.get("author") or {}).get("username") or "unknown"
        content = str(msg.get("content") or "").strip()
        if content:
            lines.append(f"{author}: {content}")
    return "\n".join(lines)


def build_prompt(thread: dict[str, Any], transcript: str, agent_name: str) -> str:
    return (
        f"You are Hermes profile {agent_name}. A Discord forum post has been claimed for autonomous work.\n"
        f"Thread title: {thread.get('name') or thread.get('id')}\n\n"
        "Read the transcript, perform the requested work using available tools, and return a concise final result. "
        "Do not schedule another cron job.\n\n"
        f"Transcript:\n{transcript}"
    )


def run_hermes(profile: str, prompt: str, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["hermes", "-p", profile, "chat", "-Q", "-q", prompt],
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def run_monitor(
    *,
    forum_channel_id: str,
    state_path: Path,
    profile: str,
    agent_name: str,
    max_tasks: int = 1,
    dry_run: bool = False,
    hermes_timeout: int = 1800,
) -> dict[str, Any]:
    token = get_bot_token()
    state = load_state(state_path)
    threads = list_forum_threads(token, forum_channel_id)
    excluded_tag_ids = forum_tag_ids_by_name(token, forum_channel_id, {"reject", "done"})
    messages_cache: dict[str, list[dict[str, Any]]] = {}

    def cached_fetch(thread_id: str) -> list[dict[str, Any]]:
        if thread_id not in messages_cache:
            messages_cache[thread_id] = fetch_messages(token, thread_id)
        return messages_cache[thread_id]

    claimable = select_claimable_threads(
        threads,
        state,
        cached_fetch,
        agent_name=agent_name,
        excluded_tag_ids=excluded_tag_ids,
    )
    result = {"claimable": [str(t.get("id")) for t in claimable], "processed": [], "failed": []}
    if dry_run:
        return result

    for thread in claimable[: max(1, max_tasks)]:
        thread_id = str(thread["id"])
        now = int(time.time())
        state.setdefault("threads", {})[thread_id] = {"status": "claimed", "claimed_at": now, "name": thread.get("name")}
        save_state(state_path, state)
        send_message(token, thread_id, f"Claimed by {agent_name}; running autonomous work now.")
        transcript = format_transcript(cached_fetch(thread_id))
        prompt = build_prompt(thread, transcript, agent_name)
        completed = run_hermes(profile, prompt, hermes_timeout)
        output = (completed.stdout or completed.stderr or "").strip()
        if completed.returncode == 0:
            state["threads"][thread_id].update({"status": "completed", "completed_at": int(time.time())})
            send_message(token, thread_id, output or f"{agent_name} completed this task.")
            result["processed"].append(thread_id)
        else:
            state["threads"][thread_id].update(
                {"status": "failed", "failed_at": int(time.time()), "returncode": completed.returncode}
            )
            send_message(token, thread_id, f"{agent_name} failed to complete this task.\n\n{output}")
            result["failed"].append(thread_id)
        save_state(state_path, state)
    return result


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor a Discord forum and run Hermes on unclaimed posts.")
    parser.add_argument("--forum-channel-id", default=os.getenv("DISCORD_FORUM_CHANNEL_ID", ""))
    parser.add_argument("--state-path", type=Path, default=Path(os.getenv("DISCORD_FORUM_AGENT_STATE", DEFAULT_STATE_PATH)))
    parser.add_argument("--profile", default=os.getenv("HERMES_FORUM_AGENT_PROFILE", "warp"))
    parser.add_argument("--agent-name", default=os.getenv("HERMES_FORUM_AGENT_NAME", "warp"))
    parser.add_argument("--max-tasks", type=int, default=int(os.getenv("HERMES_FORUM_AGENT_MAX_TASKS", "1")))
    parser.add_argument("--hermes-timeout", type=int, default=int(os.getenv("HERMES_FORUM_AGENT_TIMEOUT", "1800")))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet-if-empty", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.forum_channel_id:
        raise SystemExit("--forum-channel-id or DISCORD_FORUM_CHANNEL_ID is required")
    result = run_monitor(
        forum_channel_id=args.forum_channel_id,
        state_path=args.state_path,
        profile=args.profile,
        agent_name=args.agent_name,
        max_tasks=args.max_tasks,
        dry_run=args.dry_run,
        hermes_timeout=args.hermes_timeout,
    )
    if not (args.quiet_if_empty and not result["claimable"] and not result["processed"] and not result["failed"]):
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
