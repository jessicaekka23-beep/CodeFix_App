"""Tests for buggy_math module — these tests document the CORRECT behavior."""
import pytest
from buggy_math import calculate_average, find_max, fibonacci, is_palindrome


def test_calculate_average_basic():
    assert calculate_average([1, 2, 3]) == 2.0


def test_calculate_average_single():
    assert calculate_average([5]) == 5.0


def test_calculate_average_empty():
    with pytest.raises(ZeroDivisionError):
        calculate_average([])


def test_find_max_basic():
    assert find_max([1, 3, 2]) == 3


def test_find_max_negative():
    assert find_max([-1, -3, -2]) == -1


def test_find_max_empty():
    assert find_max([]) is None


def test_fibonacci_five():
    assert fibonacci(5) == [0, 1, 1, 2, 3]


def test_fibonacci_one():
    assert fibonacci(1) == [0]


def test_fibonacci_zero():
    assert fibonacci(0) == []


def test_is_palindrome_basic():
    assert is_palindrome("racecar") is True


def test_is_palindrome_case():
    assert is_palindrome("Racecar") is True  # Should handle case


def test_is_palindrome_not():
    assert is_palindrome("hello") is False
