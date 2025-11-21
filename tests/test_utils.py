"""Tests for utility functions."""

import tempfile
from pathlib import Path

import pytest

from retrochat_evaluator.utils.jsonl import (
    read_jsonl,
    write_jsonl,
    stream_jsonl,
    count_lines,
)
from retrochat_evaluator.utils.prompts import (
    load_prompt_template,
    format_prompt,
    get_required_variables,
    validate_prompt_variables,
)


class TestJsonlUtils:
    """Tests for JSONL utilities."""

    def test_read_jsonl(self, mock_session_path: Path):
        """Test reading JSONL file."""
        data = read_jsonl(mock_session_path)
        assert len(data) > 0
        assert isinstance(data, list)
        assert all(isinstance(item, dict) for item in data)

    def test_read_jsonl_file_not_found(self):
        """Test reading non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_jsonl(Path("/nonexistent/file.jsonl"))

    def test_stream_jsonl(self, mock_session_path: Path):
        """Test streaming JSONL file."""
        items = list(stream_jsonl(mock_session_path))
        assert len(items) > 0
        assert all(isinstance(item, dict) for item in items)

    def test_write_jsonl(self):
        """Test writing JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        try:
            data = [{"key": "value1"}, {"key": "value2"}, {"key": "value3"}]
            write_jsonl(temp_path, data)

            # Read back and verify
            loaded = read_jsonl(temp_path)
            assert len(loaded) == 3
            assert loaded[0]["key"] == "value1"
        finally:
            temp_path.unlink()

    def test_write_jsonl_append(self):
        """Test appending to JSONL file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            temp_path = Path(f.name)

        try:
            write_jsonl(temp_path, [{"a": 1}])
            write_jsonl(temp_path, [{"b": 2}], append=True)

            loaded = read_jsonl(temp_path)
            assert len(loaded) == 2
        finally:
            temp_path.unlink()

    def test_count_lines(self, mock_session_path: Path):
        """Test counting non-empty lines."""
        count = count_lines(mock_session_path)
        assert count > 0

    def test_read_jsonl_skip_malformed(self):
        """Test skipping malformed lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"valid": true}\n')
            f.write('invalid json line\n')
            f.write('{"also_valid": true}\n')
            temp_path = Path(f.name)

        try:
            data = read_jsonl(temp_path, skip_malformed=True)
            assert len(data) == 2
        finally:
            temp_path.unlink()

    def test_read_jsonl_raise_on_malformed(self):
        """Test raising on malformed lines."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write('{"valid": true}\n')
            f.write('invalid json line\n')
            temp_path = Path(f.name)

        try:
            with pytest.raises(Exception):
                read_jsonl(temp_path, skip_malformed=False)
        finally:
            temp_path.unlink()


class TestPromptUtils:
    """Tests for prompt utilities."""

    def test_load_prompt_template(self, fixtures_dir: Path):
        """Test loading prompt template."""
        # Create a temporary prompt file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello {name}, welcome to {place}!")
            temp_path = Path(f.name)

        try:
            template = load_prompt_template(temp_path)
            assert "Hello {name}" in template
        finally:
            temp_path.unlink()

    def test_load_prompt_template_not_found(self):
        """Test loading non-existent template."""
        with pytest.raises(FileNotFoundError):
            load_prompt_template(Path("/nonexistent/template.txt"))

    def test_load_prompt_with_prompts_dir(self):
        """Test loading prompt relative to prompts directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompts_dir = Path(tmpdir)
            template_path = prompts_dir / "test.txt"
            template_path.write_text("Test prompt content")

            loaded = load_prompt_template(Path("test.txt"), prompts_dir=prompts_dir)
            assert loaded == "Test prompt content"

    def test_format_prompt(self):
        """Test formatting prompt with variables."""
        template = "Hello {name}, you have {count} messages."
        result = format_prompt(template, name="Alice", count="5")
        assert result == "Hello Alice, you have 5 messages."

    def test_format_prompt_missing_variable(self):
        """Test formatting with missing variable."""
        template = "Hello {name}, {missing}!"
        with pytest.raises(KeyError):
            format_prompt(template, name="Alice")

    def test_get_required_variables(self):
        """Test extracting required variables."""
        template = "Hello {name}, welcome to {place}. Your score is {score}."
        variables = get_required_variables(template)
        assert set(variables) == {"name", "place", "score"}

    def test_get_required_variables_no_variables(self):
        """Test with no variables."""
        template = "Hello world!"
        variables = get_required_variables(template)
        assert variables == []

    def test_get_required_variables_duplicates(self):
        """Test that duplicates are removed."""
        template = "{name} said hello to {name} at {place}"
        variables = get_required_variables(template)
        assert len(variables) == 2
        assert "name" in variables
        assert "place" in variables

    def test_validate_prompt_variables_all_provided(self):
        """Test validation when all variables provided."""
        template = "Hello {name}!"
        missing = validate_prompt_variables(template, {"name": "Alice"})
        assert missing == []

    def test_validate_prompt_variables_missing(self):
        """Test validation when variables missing."""
        template = "Hello {name}, welcome to {place}!"
        missing = validate_prompt_variables(template, {"name": "Alice"})
        assert missing == ["place"]
