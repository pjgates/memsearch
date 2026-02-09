"""Tests for session log parsing."""

import json
from pathlib import Path

from memsearch.session import parse_session_file


def test_parse_session(tmp_path: Path):
    log = tmp_path / "session.jsonl"
    lines = [
        json.dumps({
            "type": "user",
            "sessionId": "s1",
            "message": {"role": "user", "content": "Hello"},
            "timestamp": "2025-01-01T00:00:00Z",
        }),
        json.dumps({
            "type": "assistant",
            "sessionId": "s1",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!"}],
            },
            "timestamp": "2025-01-01T00:00:01Z",
        }),
    ]
    log.write_text("\n".join(lines))

    sessions = parse_session_file(log)
    assert len(sessions) == 1
    s = sessions[0]
    assert s.session_id == "s1"
    assert len(s.messages) == 2
    assert s.messages[0].role == "user"
    assert s.messages[0].content == "Hello"
    assert s.messages[1].content == "Hi there!"


def test_to_markdown(tmp_path: Path):
    log = tmp_path / "session.jsonl"
    lines = [
        json.dumps({
            "type": "user",
            "sessionId": "s1",
            "message": {"role": "user", "content": "What is 2+2?"},
            "timestamp": "2025-01-01T00:00:00Z",
        }),
        json.dumps({
            "type": "assistant",
            "sessionId": "s1",
            "message": {"role": "assistant", "content": "4"},
            "timestamp": "2025-01-01T00:00:01Z",
        }),
    ]
    log.write_text("\n".join(lines))

    sessions = parse_session_file(log)
    md = sessions[0].to_markdown()
    assert "# Session s1" in md
    assert "## User" in md
    assert "What is 2+2?" in md
    assert "## Assistant" in md


def test_empty_file(tmp_path: Path):
    log = tmp_path / "empty.jsonl"
    log.write_text("")
    assert parse_session_file(log) == []


def test_missing_file(tmp_path: Path):
    assert parse_session_file(tmp_path / "nope.jsonl") == []
