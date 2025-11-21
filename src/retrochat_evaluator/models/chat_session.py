"""Chat session data models for parsing Claude Code JSONL format."""

from typing import Optional, Any
from datetime import datetime
from pathlib import Path
import json

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """A single tool call made by the assistant."""

    tool_id: str = Field(..., description="Unique ID for this tool call")
    tool_name: str = Field(..., description="Name of the tool called")
    tool_input: dict[str, Any] = Field(
        default_factory=dict, description="Input parameters for the tool"
    )
    tool_result: Optional[str] = Field(default=None, description="Result from tool execution")

    def format_for_prompt(self, max_result_length: int = 500) -> str:
        """Format tool call for prompt inclusion."""
        # Summarize input based on tool type
        input_summary = self._summarize_input()
        result_str = ""
        if self.tool_result:
            result = self.tool_result
            if len(result) > max_result_length:
                result = result[:max_result_length] + "... (truncated)"
            result_str = f"\n  Result: {result}"
        return f"- {self.tool_name}: {input_summary}{result_str}"

    def _summarize_input(self) -> str:
        """Create a summary of the tool input."""
        if self.tool_name == "Read":
            return self.tool_input.get("file_path", "unknown file")
        elif self.tool_name == "Write":
            return self.tool_input.get("file_path", "unknown file")
        elif self.tool_name == "Edit":
            return self.tool_input.get("file_path", "unknown file")
        elif self.tool_name == "Glob":
            return self.tool_input.get("pattern", "unknown pattern")
        elif self.tool_name == "Grep":
            return f"'{self.tool_input.get('pattern', '')}'"
        elif self.tool_name == "Bash":
            cmd = self.tool_input.get("command", "")
            if len(cmd) > 50:
                cmd = cmd[:50] + "..."
            return cmd
        else:
            # Generic summary
            return json.dumps(self.tool_input)[:100]


class ChatMessage(BaseModel):
    """A single message in the chat session."""

    role: str = Field(..., description="Message role: user or assistant")
    content: str = Field(default="", description="Text content of the message")
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Tool calls made in this message"
    )
    thinking: Optional[str] = Field(
        default=None, description="Assistant's thinking (excluded from output)"
    )
    timestamp: Optional[datetime] = Field(default=None, description="When message was sent")

    def format_for_prompt(self, include_tool_results: bool = True) -> str:
        """Format message for prompt inclusion."""
        parts = []

        if self.role == "user":
            parts.append(f"[USER]\n{self.content}")
        else:
            if self.content:
                parts.append(f"[ASSISTANT]\n{self.content}")

            if self.tool_calls:
                tool_strs = [tc.format_for_prompt() for tc in self.tool_calls]
                parts.append("[TOOL_CALLS]\n" + "\n".join(tool_strs))

        return "\n\n".join(parts)


class ChatSession(BaseModel):
    """A complete chat session parsed from JSONL."""

    session_id: str = Field(..., description="Unique session identifier")
    messages: list[ChatMessage] = Field(
        default_factory=list, description="Messages in the session"
    )
    cwd: Optional[str] = Field(default=None, description="Working directory")
    git_branch: Optional[str] = Field(default=None, description="Git branch")
    start_timestamp: Optional[datetime] = Field(default=None, description="Session start time")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional session metadata"
    )

    def format_for_prompt(self) -> str:
        """Format entire session for prompt inclusion."""
        lines = ["=== Chat Session ==="]
        lines.append(f"Session ID: {self.session_id}")

        if self.start_timestamp:
            lines.append(f"Timestamp: {self.start_timestamp.isoformat()}")
        if self.git_branch:
            lines.append(f"Git Branch: {self.git_branch}")
        if self.cwd:
            lines.append(f"Working Directory: {self.cwd}")

        lines.append("")

        # Group messages into turns
        turns = self._group_into_turns()
        for i, turn in enumerate(turns, 1):
            lines.append(f"--- Turn {i} ---")
            for msg in turn:
                lines.append(msg.format_for_prompt())
            lines.append("")

        lines.append("=== End Session ===")
        lines.append(f"Total Turns: {len(turns)}")

        return "\n".join(lines)

    def _group_into_turns(self) -> list[list[ChatMessage]]:
        """Group messages into user-assistant turns."""
        turns = []
        current_turn = []

        for msg in self.messages:
            if msg.role == "user":
                if current_turn:
                    turns.append(current_turn)
                current_turn = [msg]
            else:
                current_turn.append(msg)

        if current_turn:
            turns.append(current_turn)

        return turns

    @property
    def turn_count(self) -> int:
        """Get the number of conversation turns."""
        return len(self._group_into_turns())

    @classmethod
    def from_jsonl(cls, path: Path) -> "ChatSession":
        """Parse a chat session from JSONL file."""
        messages = []
        session_id = None
        cwd = None
        git_branch = None
        start_timestamp = None
        metadata = {}

        # Track tool calls to match with results
        pending_tool_calls: dict[str, ToolCall] = {}

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
                    cwd = entry.get("cwd")
                    git_branch = entry.get("gitBranch")
                    if entry.get("timestamp"):
                        try:
                            start_timestamp = datetime.fromisoformat(
                                entry["timestamp"].replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass

                if entry_type == "user":
                    message_data = entry.get("message", {})
                    content = message_data.get("content", "")

                    # Handle tool results
                    if isinstance(content, list):
                        for item in content:
                            if item.get("type") == "tool_result":
                                tool_id = item.get("tool_use_id")
                                if tool_id and tool_id in pending_tool_calls:
                                    result_content = item.get("content", "")
                                    if isinstance(result_content, str):
                                        pending_tool_calls[tool_id].tool_result = result_content
                        # Skip tool result messages for display
                        continue
                    else:
                        timestamp = None
                        if entry.get("timestamp"):
                            try:
                                timestamp = datetime.fromisoformat(
                                    entry["timestamp"].replace("Z", "+00:00")
                                )
                            except ValueError:
                                pass

                        messages.append(
                            ChatMessage(
                                role="user",
                                content=content,
                                timestamp=timestamp,
                            )
                        )

                elif entry_type == "assistant":
                    message_data = entry.get("message", {})
                    content_list = message_data.get("content", [])

                    text_content = ""
                    thinking = None
                    tool_calls = []

                    for item in content_list:
                        item_type = item.get("type")

                        if item_type == "text":
                            text_content += item.get("text", "")
                        elif item_type == "thinking":
                            thinking = item.get("thinking", "")
                        elif item_type == "tool_use":
                            tool_call = ToolCall(
                                tool_id=item.get("id", ""),
                                tool_name=item.get("name", ""),
                                tool_input=item.get("input", {}),
                            )
                            tool_calls.append(tool_call)
                            pending_tool_calls[tool_call.tool_id] = tool_call

                    # Only add message if it has content or tool calls
                    if text_content or tool_calls:
                        timestamp = None
                        if entry.get("timestamp"):
                            try:
                                timestamp = datetime.fromisoformat(
                                    entry["timestamp"].replace("Z", "+00:00")
                                )
                            except ValueError:
                                pass

                        messages.append(
                            ChatMessage(
                                role="assistant",
                                content=text_content,
                                tool_calls=tool_calls,
                                thinking=thinking,
                                timestamp=timestamp,
                            )
                        )

        return cls(
            session_id=session_id or path.stem,
            messages=messages,
            cwd=cwd,
            git_branch=git_branch,
            start_timestamp=start_timestamp,
            metadata=metadata,
        )
