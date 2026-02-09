"""Session log parser — extract conversations from Claude JSONL session logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SessionMessage:
    """A single message extracted from a session log."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: str


@dataclass
class Session:
    """A parsed conversation session."""

    session_id: str
    messages: list[SessionMessage]
    source: str  # file path

    def to_markdown(self) -> str:
        """Render session as markdown for chunking / embedding."""
        parts: list[str] = [f"# Session {self.session_id}\n"]
        for msg in self.messages:
            role_label = "User" if msg.role == "user" else "Assistant"
            parts.append(f"## {role_label}\n\n{msg.content}\n")
        return "\n".join(parts)


def parse_session_file(path: str | Path) -> list[Session]:
    """Parse a JSONL session log into ``Session`` objects.

    Expects each line to be a JSON object with at least ``"type"``
    (``"user"`` or ``"assistant"``) and ``"message"`` fields.
    """
    path = Path(path)
    if not path.exists():
        return []

    messages_by_session: dict[str, list[SessionMessage]] = {}
    session_ids_order: list[str] = []

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = obj.get("type", "")
        if msg_type not in ("user", "assistant"):
            continue

        session_id = obj.get("sessionId", "unknown")
        message = obj.get("message", {})
        role = message.get("role", msg_type)

        # Extract text content
        content_raw = message.get("content", "")
        if isinstance(content_raw, list):
            # Content can be a list of content blocks
            text_parts = []
            for block in content_raw:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            content = "\n".join(text_parts)
        else:
            content = str(content_raw)

        if not content.strip():
            continue

        timestamp = obj.get("timestamp", "")

        if session_id not in messages_by_session:
            messages_by_session[session_id] = []
            session_ids_order.append(session_id)

        messages_by_session[session_id].append(
            SessionMessage(role=role, content=content, timestamp=timestamp)
        )

    return [
        Session(
            session_id=sid,
            messages=messages_by_session[sid],
            source=str(path),
        )
        for sid in session_ids_order
        if messages_by_session[sid]
    ]
