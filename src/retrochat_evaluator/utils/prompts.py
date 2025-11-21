"""Prompt template loading and formatting utilities."""

from pathlib import Path
from typing import Optional
import re


def load_prompt_template(
    path: Path,
    prompts_dir: Optional[Path] = None,
) -> str:
    """Load a prompt template from file.

    Args:
        path: Path to the prompt file (relative or absolute).
        prompts_dir: Base directory for prompts. If provided and path is relative,
                     it will be joined with prompts_dir.

    Returns:
        The prompt template string.

    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
    """
    if not path.is_absolute() and prompts_dir:
        path = prompts_dir / path

    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def format_prompt(
    template: str,
    **variables: str,
) -> str:
    """Format a prompt template with variables.

    Uses simple {variable} substitution.

    Args:
        template: The prompt template string.
        **variables: Variables to substitute.

    Returns:
        The formatted prompt.

    Raises:
        KeyError: If a required variable is not provided.
    """
    return template.format(**variables)


def get_required_variables(template: str) -> list[str]:
    """Extract required variable names from a template.

    Args:
        template: The prompt template string.

    Returns:
        List of variable names found in the template.
    """
    # Match {variable_name} patterns, excluding escaped braces
    pattern = r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}"
    return list(set(re.findall(pattern, template)))


def validate_prompt_variables(
    template: str,
    provided_variables: dict[str, str],
) -> list[str]:
    """Validate that all required variables are provided.

    Args:
        template: The prompt template string.
        provided_variables: Dictionary of provided variables.

    Returns:
        List of missing variable names. Empty if all are provided.
    """
    required = get_required_variables(template)
    return [var for var in required if var not in provided_variables]
