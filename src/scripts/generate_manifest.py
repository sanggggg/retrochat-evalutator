#!/usr/bin/env python3
"""Generate dataset manifest from input directory with detailed metadata extraction."""

import argparse
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def format_size(size_bytes: int) -> str:
    """Format bytes to human readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def extract_author_from_path(rel_path: str) -> str:
    """Extract author from directory path.

    Format: -Users-{author}-{path}/filename.jsonl
    """
    dir_path = os.path.dirname(rel_path)

    # Match pattern: -Users-{author}-
    match = re.match(r"^-Users-([^-]+)-", dir_path)
    if match:
        return match.group(1)

    # Fallback: extract from directory name
    if dir_path:
        parts = dir_path.split("/")
        for part in parts:
            if part.startswith("-Users-"):
                author = part.replace("-Users-", "").split("-")[0]
                if author:
                    return author

    return "unknown"


def extract_model_from_jsonl(jsonl_path: Path) -> str:
    """Extract model name from JSONL file."""
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    message = entry.get("message", {})
                    model = message.get("model")
                    if model:
                        return model
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception:
        pass
    return ""


def calculate_tokens(jsonl_path: Path) -> Dict[str, int]:
    """Calculate total input and output tokens from JSONL file.

    Note: input_tokens already includes cache tokens, so we don't double-count.
    """
    total_input_tokens = 0
    total_output_tokens = 0

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "assistant":
                        message = entry.get("message", {})
                        usage = message.get("usage", {})

                        # input_tokens already includes all input tokens (including cache)
                        input_tokens = usage.get("input_tokens", 0)
                        total_input_tokens += input_tokens

                        # Sum up output tokens
                        output_tokens = usage.get("output_tokens", 0)
                        total_output_tokens += output_tokens
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception:
        pass

    return {
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
    }


def calculate_session_duration(jsonl_path: Path) -> Optional[float]:
    """Calculate session duration in seconds from first to last timestamp."""
    timestamps = []

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    timestamp_str = entry.get("timestamp")
                    if timestamp_str:
                        # Parse ISO format: 2025-11-06T05:47:43.133Z
                        try:
                            # Remove 'Z' and parse
                            if timestamp_str.endswith("Z"):
                                timestamp_str = timestamp_str[:-1] + "+00:00"
                            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                            timestamps.append(dt)
                        except (ValueError, AttributeError):
                            continue
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception:
        pass

    if len(timestamps) < 2:
        return None

    timestamps.sort()
    duration = (timestamps[-1] - timestamps[0]).total_seconds()
    return duration


def count_lines(text: str) -> int:
    """Count lines in text."""
    if not text:
        return 0
    return len(text.splitlines())


def calculate_code_changes(jsonl_path: Path) -> Dict[str, int]:
    """Calculate LOC added, removed, and net growth from Edit, Write, and Delete tool calls."""
    lines_added = 0
    lines_removed = 0

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "assistant":
                        message = entry.get("message", {})
                        content = message.get("content", [])

                        if isinstance(content, list):
                            for item in content:
                                if item.get("type") == "tool_use":
                                    tool_name = item.get("name", "")
                                    tool_input = item.get("input", {})

                                    if tool_name == "Edit":
                                        # Edit: compare old_string vs new_string
                                        old_string = tool_input.get("old_string", "")
                                        new_string = tool_input.get("new_string", "")

                                        old_lines = count_lines(old_string)
                                        new_lines = count_lines(new_string)

                                        if old_lines > new_lines:
                                            lines_removed += old_lines - new_lines
                                        elif new_lines > old_lines:
                                            lines_added += new_lines - old_lines

                                    elif tool_name == "Write":
                                        # Write: creates new file, all content is added
                                        content_str = tool_input.get("content", "")
                                        lines_added += count_lines(content_str)

                                    elif tool_name == "Delete":
                                        # Delete: removes file, we can't know exact line count
                                        # but we can check if there's a file_path to track it
                                        # For now, we'll count it as a deletion event
                                        # Note: We can't know exact lines without reading the file
                                        # So we'll just track that a deletion happened
                                        pass  # Delete tool doesn't provide line count info
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception:
        pass

    net_growth = lines_added - lines_removed

    return {"lines_added": lines_added, "lines_removed": lines_removed, "net_growth": net_growth}


def count_tool_uses(jsonl_path: Path) -> Dict[str, int]:
    """Count tool_use calls by tool name."""
    tool_counts: Dict[str, int] = {}

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "assistant":
                        message = entry.get("message", {})
                        content = message.get("content", [])

                        if isinstance(content, list):
                            for item in content:
                                if item.get("type") == "tool_use":
                                    tool_name = item.get("name", "")
                                    if tool_name:
                                        tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception:
        pass

    return tool_counts


def count_user_turns(jsonl_path: Path) -> int:
    """Count the number of user messages (turns) in the session."""
    user_turn_count = 0

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    if entry.get("type") == "user":
                        user_turn_count += 1
                except (json.JSONDecodeError, KeyError):
                    continue
    except Exception:
        pass

    return user_turn_count


def calculate_token_efficiency_score(net_growth: int, total_tokens: int) -> float:
    """Calculate token efficiency score: net_growth / total_tokens.

    Returns 0.0 if total_tokens is 0 or negative.
    """
    if total_tokens <= 0:
        return 0.0
    return round(net_growth / total_tokens, 6)


def calculate_user_turn_efficiency_score(net_growth: int, user_turns: int) -> float:
    """Calculate user turn efficiency score: net_growth / user_turns.

    Returns 0.0 if user_turns is 0.
    """
    if user_turns == 0:
        return 0.0
    return round(net_growth / user_turns, 2)


def process_jsonl_file(
    jsonl_path: Path, input_dir: Path, min_size_kb: int = 4
) -> Optional[Dict[str, Any]]:
    """Process a single JSONL file and extract metadata.

    Returns None if file is too small or cannot be processed.
    """
    # Check file size
    file_size = jsonl_path.stat().st_size
    min_size_bytes = min_size_kb * 1024

    if file_size < min_size_bytes:
        return None

    # Get relative path
    try:
        rel_path = str(jsonl_path.relative_to(input_dir))
    except ValueError:
        rel_path = str(jsonl_path)

    # Extract metadata
    author = extract_author_from_path(rel_path)
    model = extract_model_from_jsonl(jsonl_path)
    tokens = calculate_tokens(jsonl_path)
    duration = calculate_session_duration(jsonl_path)
    code_changes = calculate_code_changes(jsonl_path)
    tool_counts = count_tool_uses(jsonl_path)
    user_turns = count_user_turns(jsonl_path)

    # Calculate scores
    net_growth = code_changes["net_growth"]
    total_tokens = tokens["total_tokens"]

    token_efficiency = calculate_token_efficiency_score(net_growth, total_tokens)
    user_turn_efficiency = calculate_user_turn_efficiency_score(net_growth, user_turns)
    excellence = 0.0  # Will be manually marked later

    # Build metadata
    metadata = {
        "author": author,
        "model": model,
        "file_size_bytes": file_size,
        "file_size_kb": round(file_size / 1024, 2),
        "input_tokens": tokens["input_tokens"],
        "output_tokens": tokens["output_tokens"],
        "total_tokens": tokens["total_tokens"],
        "lines_added": code_changes["lines_added"],
        "lines_removed": code_changes["lines_removed"],
        "net_growth": code_changes["net_growth"],
        "user_turns": user_turns,
        "tool_use_counts": tool_counts,
    }

    if duration is not None:
        metadata["session_duration_seconds"] = round(duration, 2)
        metadata["session_duration_minutes"] = round(duration / 60, 2)
    else:
        metadata["session_duration_seconds"] = None
        metadata["session_duration_minutes"] = None

    return {
        "file": rel_path,
        "scores": {
            "token_efficiency": token_efficiency,
            "user_turn_efficiency": user_turn_efficiency,
            "excellence": excellence,
        },
        "metadata": metadata,
    }


def annotate_score_percentiles(sessions: list[Dict[str, Any]]) -> None:
    """Annotate each session with percentile ranks per score."""
    if not sessions:
        return

    # Collect score values per metric
    score_entries: dict[str, list[tuple[int, float]]] = {}
    for idx, session in enumerate(sessions):
        for score_name, score_value in session.get("scores", {}).items():
            if isinstance(score_value, (int, float)):
                score_entries.setdefault(score_name, []).append((idx, float(score_value)))

    for score_name, entries in score_entries.items():
        if not entries:
            continue

        # Sort descending to compute percentile ranks
        entries.sort(key=lambda item: item[1], reverse=True)
        total = len(entries)
        for rank, (session_idx, _) in enumerate(entries):
            percentile_rank = round(100.0 * (total - rank) / total, 3)
            session = sessions[session_idx]
            percentiles = session.setdefault("score_percentiles", {})
            percentiles[score_name] = percentile_rank

def generate_manifest(
    input_dir: str,
    output_file: str,
    min_size_kb: int = 4,
    max_sessions: Optional[int] = None,
):
    """Generate manifest from input directory."""
    input_path = Path(input_dir)

    if not input_path.exists():
        print(f"Error: Input directory '{input_dir}' does not exist")
        return

    print(f"Scanning {input_dir} for .jsonl files (min size: {min_size_kb}KB)...")

    # Find all JSONL files
    jsonl_files = list(input_path.rglob("*.jsonl"))
    print(f"Found {len(jsonl_files)} total .jsonl files")

    # Process files
    sessions = []
    processed = 0
    skipped = 0

    for jsonl_file in sorted(jsonl_files):
        try:
            session_data = process_jsonl_file(jsonl_file, input_path, min_size_kb)
            if session_data:
                sessions.append(session_data)
                processed += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"Warning: Failed to process {jsonl_file}: {e}")
            skipped += 1

        # Progress indicator
        if (processed + skipped) % 100 == 0:
            print(
                f"  Processed: {processed + skipped}/{len(jsonl_files)} (kept: {processed}, skipped: {skipped})"
            )

    print(f"\nProcessed {processed + skipped} files:")
    print(f"  Kept: {processed} files (>= {min_size_kb}KB)")
    print(f"  Skipped: {skipped} files (< {min_size_kb}KB or errors)")

    # Apply session limits if specified (random sampling)
    if max_sessions is not None:
        if len(sessions) > max_sessions:
            print(f"\nRandomly sampling {max_sessions} sessions from {len(sessions)} total sessions")
            random.seed(42)  # Fixed seed for reproducibility
            sessions = random.sample(sessions, max_sessions)

    # Annotate percentile stats before applying splits
    annotate_score_percentiles(sessions)

    # Split into training and validation sets (9:1 ratio)
    if len(sessions) > 0:
        # Shuffle sessions for random split
        random.seed(42)  # Fixed seed for reproducibility
        shuffled_sessions = sessions.copy()
        random.shuffle(shuffled_sessions)

        # Calculate split point (90% training, 10% validation)
        split_point = int(len(shuffled_sessions) * 0.9)
        training_sessions = shuffled_sessions[:split_point]
        validation_sessions = shuffled_sessions[split_point:]

        # Add split field to each session
        for session in training_sessions:
            session["split"] = "training"
        for session in validation_sessions:
            session["split"] = "validation"

        print(f"\nSplit sessions:")
        print(f"  Training: {len(training_sessions)} sessions (90%)")
        print(f"  Validation: {len(validation_sessions)} sessions (10%)")
    else:
        print("\nNo sessions to split")

    # Create manifest
    manifest = {"sessions": sessions}

    # Save to file
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nGenerated manifest: {output_file}")
    print(f"Total sessions: {len(sessions)}")

    # Print summary statistics
    if sessions:
        total_tokens = sum(s["metadata"].get("total_tokens", 0) for s in sessions)
        total_duration = sum(
            s["metadata"].get("session_duration_minutes", 0) or 0 for s in sessions
        )
        total_growth = sum(s["metadata"].get("net_growth", 0) for s in sessions)

        print(f"\nSummary:")
        print(f"  Total tokens: {total_tokens:,}")
        print(
            f"  Total session duration: {total_duration:.1f} minutes ({total_duration/60:.1f} hours)"
        )
        print(f"  Net LOC growth: {total_growth:,} lines")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset manifest from JSONL files with detailed metadata"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="input",
        help="Input directory containing JSONL files (default: input)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset.json",
        help="Output manifest file (default: dataset.json)",
    )
    parser.add_argument(
        "--min-size-kb", type=int, default=4, help="Minimum file size in KB to include (default: 4)"
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Maximum number of sessions to include in manifest (default: no limit)",
    )
    args = parser.parse_args()

    generate_manifest(
        args.input_dir,
        args.output,
        args.min_size_kb,
        args.max_sessions,
    )


if __name__ == "__main__":
    main()
