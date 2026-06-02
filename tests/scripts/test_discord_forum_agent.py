"""Tests for Discord forum autonomous work monitor."""

import json
import subprocess

from scripts import discord_forum_agent as agent


def test_selects_unprocessed_threads_without_agent_claims():
    state = {"threads": {"done": {"status": "completed"}}}
    threads = [
        {"id": "done", "name": "Done task"},
        {"id": "claimed", "name": "Claimed task"},
        {"id": "new", "name": "New task"},
    ]
    messages_by_thread = {
        "claimed": [
            {"id": "1", "content": "Claimed by warp", "author": {"bot": True, "username": "Hermes"}},
        ],
        "new": [
            {"id": "2", "content": "Please do this", "author": {"bot": False, "username": "Cayde"}},
        ],
    }

    selected = agent.select_claimable_threads(
        threads,
        state,
        lambda thread_id: messages_by_thread.get(thread_id, []),
        agent_name="warp",
    )

    assert [thread["id"] for thread in selected] == ["new"]


def test_claim_run_post_and_mark_completed(monkeypatch, tmp_path):
    state_path = tmp_path / "state.json"
    calls = []

    def fake_request(method, path, token, params=None, body=None, timeout=15):
        calls.append((method, path, body))
        if path == "/channels/forum":
            return {"id": "forum", "guild_id": "guild", "type": 15}
        if path == "/guilds/guild/threads/active":
            return {"threads": [{"id": "thread1", "name": "Do thing", "parent_id": "forum"}]}
        if path == "/channels/forum/threads/archived/public":
            return {"threads": []}
        if path == "/channels/thread1/messages" and method == "GET":
            return [
                {
                    "id": "m1",
                    "content": "Build the thing",
                    "author": {"bot": False, "username": "Cayde"},
                    "timestamp": "2026-06-02T00:00:00Z",
                }
            ]
        if path == "/channels/thread1/messages" and method == "POST":
            return {"id": "posted", "channel_id": "thread1", "content": body["content"]}
        raise AssertionError((method, path, body))

    def fake_run(cmd, text, capture_output, timeout):
        assert cmd[:4] == ["hermes", "-p", "warp", "chat"]
        assert "Build the thing" in cmd[-1]
        return subprocess.CompletedProcess(cmd, 0, stdout="done", stderr="")

    monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
    monkeypatch.setattr(agent, "discord_request", fake_request)
    monkeypatch.setattr(agent.subprocess, "run", fake_run)

    result = agent.run_monitor(
        forum_channel_id="forum",
        state_path=state_path,
        profile="warp",
        agent_name="warp",
        max_tasks=1,
    )

    assert result["processed"] == ["thread1"]
    saved = json.loads(state_path.read_text())
    assert saved["threads"]["thread1"]["status"] == "completed"
    posted_bodies = [body["content"] for method, path, body in calls if method == "POST"]
    assert any("Claimed by warp" in body for body in posted_bodies)
    assert any("done" in body for body in posted_bodies)


def test_get_bot_token_reads_profile_dotenv(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    env_path.write_text("DISCORD_BOT_TOKEN=dotenv-token\n")
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.setattr(agent, "get_hermes_home", lambda: tmp_path)

    assert agent.get_bot_token() == "dotenv-token"


def test_main_quiet_if_empty_suppresses_empty_output(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
    monkeypatch.setattr(agent, "list_forum_threads", lambda token, forum_channel_id: [])

    rc = agent.main([
        "--forum-channel-id", "forum",
        "--state-path", str(tmp_path / "state.json"),
        "--quiet-if-empty",
    ])

    assert rc == 0
    assert capsys.readouterr().out == ""


def test_dry_run_does_not_post_or_spawn(monkeypatch, tmp_path):
    posted = []

    def fake_request(method, path, token, params=None, body=None, timeout=15):
        if method == "POST":
            posted.append(body)
        if path == "/channels/forum":
            return {"id": "forum", "guild_id": "guild", "type": 15}
        if path == "/guilds/guild/threads/active":
            return {"threads": [{"id": "thread1", "name": "Do thing", "parent_id": "forum"}]}
        if path == "/channels/forum/threads/archived/public":
            return {"threads": []}
        if path == "/channels/thread1/messages":
            return [{"id": "m1", "content": "Build", "author": {"bot": False, "username": "Cayde"}}]
        return {"id": "posted"}

    monkeypatch.setenv("DISCORD_BOT_TOKEN", "tok")
    monkeypatch.setattr(agent, "discord_request", fake_request)
    monkeypatch.setattr(agent.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError("spawned")))

    result = agent.run_monitor(
        forum_channel_id="forum",
        state_path=tmp_path / "state.json",
        profile="warp",
        agent_name="warp",
        dry_run=True,
    )

    assert result["claimable"] == ["thread1"]
    assert posted == []
