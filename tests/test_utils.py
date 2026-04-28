"""Unit tests for Mneme utility functions."""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSlugify:
    """Tests for the slugify utility."""

    def test_basic_slug(self):
        from mneme.utils import slugify
        assert slugify("Hello World") == "hello-world"

    def test_slug_preserves_numbers(self):
        from mneme.utils import slugify
        assert slugify("Test 123") == "test-123"

    def test_slug_lowercases(self):
        from mneme.utils import slugify
        assert slugify("MiXeD CaSe") == "mixed-case"

    def test_slug_normalizes_special_chars(self):
        from mneme.utils import slugify
        result = slugify("Hello, World! @#$%")
        assert result == "hello-world"

    def test_slug_empty_returns_uuid(self):
        from mneme.utils import slugify
        result = slugify("")
        assert result != ""
        # UUID format check: 8-4-4-4-12 hex digits
        parts = result.split("-")
        assert len(parts) == 5

    def test_slug_trims_hyphens(self):
        from mneme.utils import slugify
        result = slugify("  ---hello---  ")
        assert result == "hello"

    def test_slug_multiple_spaces_become_single_hyphen(self):
        from mneme.utils import slugify
        assert slugify("a    b    c") == "a-b-c"

    def test_slug_single_word(self):
        from mneme.utils import slugify
        assert slugify("hello") == "hello"


class TestSummarizeSimple:
    """Tests for summarize_simple utility."""

    def test_short_text_unchanged(self):
        from mneme.utils import summarize_simple
        assert summarize_simple("Short text") == "Short text"

    def test_long_text_truncated(self):
        from mneme.utils import summarize_simple
        long = "a" * 500
        result = summarize_simple(long, max_len=100)
        assert len(result) < 500
        assert result.endswith("...")

    def test_truncation_breaks_at_word(self):
        from mneme.utils import summarize_simple
        text = "hello world this is a long sentence"
        result = summarize_simple(text, max_len=20)
        assert result.endswith("...")
        # Should not cut in the middle of a word
        assert not result[:-3].endswith(" ")

    def test_max_len_zero(self):
        from mneme.utils import summarize_simple
        result = summarize_simple("hello world", max_len=0)
        assert result == ""

    def test_newlines_collapsed(self):
        from mneme.utils import summarize_simple
        text = "line1\nline2\nline3"
        result = summarize_simple(text)
        assert "\n" not in result
        assert "line1 line2 line3" == result


class TestNowIso:
    """Tests for now_iso utility."""

    def test_returns_valid_iso_string(self):
        from mneme.utils import now_iso
        result = now_iso()
        # ISO 8601 with timezone info
        assert "T" in result
        assert "+" in result or "Z" in result

    def test_returns_different_values(self):
        from mneme.utils import now_iso
        t1 = now_iso()
        time.sleep(0.01)
        t2 = now_iso()
        assert t1 < t2


class TestRowAsDict:
    """Tests for row_as_dict utility."""

    def test_basic_conversion(self):
        from mneme.utils import row_as_dict
        result = row_as_dict(["a", "b", 1], ["col1", "col2", "col3"])
        assert result == {"col1": "a", "col2": "b", "col3": 1}

    def test_empty_list(self):
        from mneme.utils import row_as_dict
        result = row_as_dict([], [])
        assert result == {}

    def test_empty_column_names(self):
        from mneme.utils import row_as_dict
        result = row_as_dict(["a"], [])
        assert result == {}


class TestGetCol:
    """Tests for get_col utility."""

    def test_dict_row(self):
        from mneme.utils import get_col
        assert get_col({"key": "value"}, "key") == "value"

    def test_dict_row_missing(self):
        from mneme.utils import get_col
        assert get_col({"key": "value"}, "missing") is None

    def test_list_row(self):
        from mneme.utils import get_col
        assert get_col(["first"], "col") == "first"
