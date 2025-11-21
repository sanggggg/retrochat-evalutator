"""JSONL file handling utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)


def read_jsonl(path: Path, skip_malformed: bool = True) -> list[dict[str, Any]]:
    """Read a JSONL file and return list of parsed objects.

    Args:
        path: Path to the JSONL file.
        skip_malformed: If True, skip malformed lines instead of raising.

    Returns:
        List of parsed JSON objects.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If skip_malformed is False and a line is invalid.
    """
    results = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as e:
                if skip_malformed:
                    logger.warning(f"Skipping malformed line {line_num} in {path}: {e}")
                else:
                    raise

    return results


def stream_jsonl(path: Path, skip_malformed: bool = True) -> Iterator[dict[str, Any]]:
    """Stream a JSONL file line by line for memory efficiency.

    Args:
        path: Path to the JSONL file.
        skip_malformed: If True, skip malformed lines instead of raising.

    Yields:
        Parsed JSON objects one at a time.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If skip_malformed is False and a line is invalid.
    """
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                if skip_malformed:
                    logger.warning(f"Skipping malformed line {line_num} in {path}: {e}")
                else:
                    raise


def write_jsonl(
    path: Path,
    data: list[dict[str, Any]],
    append: bool = False,
) -> None:
    """Write a list of objects to a JSONL file.

    Args:
        path: Path to the output file.
        data: List of objects to write.
        append: If True, append to existing file instead of overwriting.
    """
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def count_lines(path: Path) -> int:
    """Count the number of non-empty lines in a JSONL file.

    Args:
        path: Path to the JSONL file.

    Returns:
        Number of non-empty lines.
    """
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def get_file_size_mb(path: Path) -> float:
    """Get file size in megabytes.

    Args:
        path: Path to the file.

    Returns:
        File size in MB.
    """
    return path.stat().st_size / (1024 * 1024)
