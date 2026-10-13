# Test Workspace

This is a synthetic Python project with intentional bugs, used to test CodePilot Local's investigation and fix capabilities.

## Known Bugs

1. `buggy_math.py`: Off-by-one error in `calculate_average`, wrong comparison in `find_max`, index error in `fibonacci`, case sensitivity in `is_palindrome`.
2. `buggy_strings.py`: Missing None check in `count_words`, character-level reversal in `reverse_words`, incorrect truncation length in `truncate_string`.

## Running Tests

```bash
python -m pytest test_workspace/ -v
```
