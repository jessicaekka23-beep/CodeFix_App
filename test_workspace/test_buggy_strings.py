"""Tests for buggy_strings module — these tests document the CORRECT behavior."""
import pytest
from buggy_strings import count_words, reverse_words, truncate_string


def test_count_words_basic():
    assert count_words("hello world") == 2


def test_count_words_empty():
    assert count_words("") == 0


def test_count_words_none():
    with pytest.raises(TypeError):
        count_words(None)  # Should raise TypeError, not AttributeError


def test_reverse_words_basic():
    assert reverse_words("hello world") == "world hello"


def test_reverse_words_single():
    assert reverse_words("hello") == "hello"


def test_truncate_basic():
    assert truncate_string("hello world", 5) == "he..."


def test_truncate_short():
    assert truncate_string("hi", 5) == "hi"


def test_truncate_exact():
    assert truncate_string("hello", 5) == "hello"
