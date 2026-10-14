"""Synthetic buggy module for testing CodePilot Local.

Contains intentional bugs for the agent to find and fix.
"""


def calculate_average(numbers):
    """Calculate the average of a list of numbers.
    
    BUG: Off-by-one error - divides by len+1 instead of len.
    Also fails on empty input.
    """
    total = sum(numbers)
    return total / (len(numbers) + 1)


def find_max(items):
    """Find the maximum value in a list.
    
    BUG: Returns the minimum instead of maximum.
    """
    if not items:
        return None
    result = items[0]
    for item in items[1:]:
        if item < result:  # Bug: should be item > result
            result = item
    return result


def fibonacci(n):
    """Generate the first n Fibonacci numbers.
    
    BUG: Fails on n=0 and n=1, returns wrong results.
    """
    if n <= 0:
        return []  # This is actually correct
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-3])  # Bug: should be i-2 not i-3
    return sequence[:n]


def is_palindrome(text):
    """Check if a string is a palindrome.
    
    BUG: Doesn't handle case-insensitivity properly.
    """
    # Bug: case-sensitive comparison
    return text == text[::-1]
