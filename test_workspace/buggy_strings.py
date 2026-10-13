"""Synthetic buggy string processing module.

Contains intentional bugs for testing.
"""


def count_words(text):
    """Count words. BUG: doesn't handle None input."""
    words = text.split()  # Bug: crashes on None
    return len(words)


def reverse_words(sentence):
    """Reverse the order of words in a sentence.
    
    BUG: Reverses characters instead of words.
    """
    return sentence[::-1]  # Bug: reverses chars, not words


def truncate_string(text, max_length):
    """Truncate a string to max_length and add '...' if truncated.
    
    BUG: Incorrect length calculation.
    """
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."  # Bug: total length is max_length + 3
