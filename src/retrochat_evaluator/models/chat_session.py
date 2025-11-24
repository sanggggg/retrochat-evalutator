"""Chat session data models for parsing refined JSON format."""

from typing import Optional, Any, Literal
from pathlib import Path
import json

from pydantic import BaseModel, Field


# Default truncation settings
DEFAULT_MAX_CONTENT_LENGTH = 2000
DEFAULT_KEEP_LENGTH = 1000


def truncate_content(
    content: str,
    max_length: int = DEFAULT_MAX_CONTENT_LENGTH,
    keep_length: int = DEFAULT_KEEP_LENGTH,
) -> str:
    """Truncate content if it exceeds max_length.

    If content is longer than max_length, keep the first and last keep_length
    characters and insert <manual_skip/> tag in between.

    Args:
        content: The content string to potentially truncate.
        max_length: Maximum allowed length before truncation.
        keep_length: Number of characters to keep at start and end.

    Returns:
        Original content if within limit, or truncated content with <manual_skip/> tag.
    """
    if len(content) <= max_length:
        return content

    start = content[:keep_length]
    end = content[-keep_length:]
    return f"{start}\n<manual_skip/>\n{end}"


class Turn(BaseModel):
    """A single turn in the chat session."""

    turn_number: int = Field(..., description="Turn number in the conversation")
    role: Literal["user", "assistant", "system"] = Field(
        ..., description="Role: user, assistant, or system"
    )
    message_type: str = Field(
        ..., description="Message type: simple_message, thinking, tool_request(X), tool_result(X)"
    )
    content: str = Field(default="", description="Content of the turn")


class ChatSession(BaseModel):
    """A complete chat session parsed from JSON."""

    session_id: str = Field(..., description="Unique session identifier")
    total_turns: int = Field(default=0, description="Total number of turns")
    turns: list[Turn] = Field(default_factory=list, description="Turns in the session")

    def format_for_prompt(self) -> str:
        """Format entire session as JSON string for prompt inclusion."""
        return json.dumps(self.model_dump(), indent=2, ensure_ascii=False)

    @property
    def turn_count(self) -> int:
        """Get the number of conversation turns (unique turn numbers)."""
        if not self.turns:
            return 0
        return max(t.turn_number for t in self.turns)

    @classmethod
    def from_json(cls, path: Path) -> "ChatSession":
        """Parse a chat session from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            session_id=data.get("session_id", path.stem),
            total_turns=data.get("total_turns", 0),
            turns=[Turn(**turn) for turn in data.get("turns", [])],
        )

    @classmethod
    def from_jsonl(cls, path: Path) -> "ChatSession":
        """Parse a chat session from JSONL file (legacy Claude Code format).

        This method converts the old JSONL format to the new Turn-based format.
        """
        turns = []
        session_id = None
        turn_number = 0

        # Track tool calls to match with results
        pending_tool_calls: dict[str, str] = {}

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                entry_type = entry.get("type")

                # Skip file history snapshots
                if entry_type == "file-history-snapshot":
                    continue

                # Extract session metadata from first user message
                if session_id is None and entry.get("sessionId"):
                    session_id = entry.get("sessionId")

                if entry_type == "user":
                    message_data = entry.get("message", {})
                    content = message_data.get("content", "")

                    # Handle tool results
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "tool_result":
                                tool_id = item.get("tool_use_id")
                                if tool_id and tool_id in pending_tool_calls:
                                    tool_name = pending_tool_calls[tool_id]
                                    result_content = item.get("content", "")
                                    if isinstance(result_content, str):
                                        turn_number += 1
                                        turns.append(
                                            Turn(
                                                turn_number=turn_number,
                                                role="system",
                                                message_type=f"tool_result({tool_name})",
                                                content=truncate_content(result_content),
                                            )
                                        )
                        continue
                    else:
                        turn_number += 1
                        turns.append(
                            Turn(
                                turn_number=turn_number,
                                role="user",
                                message_type="simple_message",
                                content=truncate_content(content),
                            )
                        )

                elif entry_type == "assistant":
                    message_data = entry.get("message", {})
                    content_list = message_data.get("content", [])

                    for item in content_list:
                        item_type = item.get("type")

                        if item_type == "text":
                            text = item.get("text", "")
                            if text:
                                turns.append(
                                    Turn(
                                        turn_number=turn_number,
                                        role="assistant",
                                        message_type="simple_message",
                                        content=truncate_content(text),
                                    )
                                )
                        elif item_type == "thinking":
                            thinking = item.get("thinking", "")
                            if thinking:
                                turns.append(
                                    Turn(
                                        turn_number=turn_number,
                                        role="assistant",
                                        message_type="thinking",
                                        content=truncate_content(thinking),
                                    )
                                )
                        elif item_type == "tool_use":
                            tool_id = item.get("id", "")
                            tool_name = item.get("name", "")
                            tool_input = item.get("input", {})

                            pending_tool_calls[tool_id] = tool_name

                            # Format tool input as YAML-like string
                            input_lines = []
                            for key, value in tool_input.items():
                                input_lines.append(f"{key}: {value}")
                            input_str = "\n".join(input_lines)

                            turns.append(
                                Turn(
                                    turn_number=turn_number,
                                    role="assistant",
                                    message_type=f"tool_request({tool_name})",
                                    content=truncate_content(input_str),
                                )
                            )

        return cls(
            session_id=session_id or path.stem,
            total_turns=turn_number,
            turns=turns,
        )

    def to_json(self, path: Path) -> None:
        """Save session to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2, ensure_ascii=False)
