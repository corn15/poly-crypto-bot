# Test Suite

This directory contains the test suite for the polymarket-crypto-expiry-prediction project.

## Overview

The test suite is built using Python's built-in `unittest` framework and provides comprehensive coverage for the utility functions in the project.

## Test Files

- `test_time.py` - Tests for the `src/utils/time.py` module, specifically the `get_current_time_str` function

## Running Tests

### Method 1: Using the test runner script (Recommended)

```bash
python run_tests.py
```

### Method 2: Using unittest directly

```bash
# Run all tests
python -m unittest discover tests/ -v

# Run specific test file
python -m unittest tests.test_time -v

# Run specific test class
python -m unittest tests.test_time.TestGetCurrentTimeStr -v

# Run specific test method
python -m unittest tests.test_time.TestGetCurrentTimeStr.test_basic_conversion_utc_to_et -v
```

## Test Coverage

### `test_time.py`

This file contains comprehensive tests for the `get_current_time_str` function:

- **Basic functionality**: Tests UTC to Eastern Time conversion
- **Time format handling**: Tests 12-hour format with am/pm
- **Edge cases**: Tests midnight (12:00 AM) and noon (12:00 PM)
- **Month handling**: Tests all 12 months with correct lowercase formatting
- **Day formatting**: Tests zero-padded day format (01-31)
- **DST transitions**: Tests behavior during Daylight Saving Time changes
- **Timezone inputs**: Tests with different input timezones (UTC, PT, London, ET)
- **Year boundaries**: Tests behavior at December 31/January 1 transitions
- **Leap year**: Tests February 29th handling
- **Format consistency**: Validates the output format structure

## Test Data and Scenarios

The tests use realistic datetime scenarios including:

- Standard Time (EST): UTC-5 offset (November-February)
- Daylight Saving Time (EDT): UTC-4 offset (March-October)
- Various input timezones to ensure consistent output
- Edge cases like day boundaries and leap years

## Adding New Tests

When adding new tests:

1. Create test methods that start with `test_`
2. Use descriptive names that explain what is being tested
3. Include docstrings explaining the test scenario
4. Use `self.assertEqual()` and other assertion methods appropriately
5. Consider edge cases and error conditions
6. Group related tests in the same test class

## Example Test Structure

```python
def test_new_functionality(self):
    """Test description explaining what this test validates."""
    # Arrange
    input_data = create_test_input()
    expected_result = "expected-output"
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    self.assertEqual(result, expected_result)
```

## Dependencies

The tests require the following dependencies (already included in `pyproject.toml`):

- `pytz` - For timezone handling
- Standard library `unittest` module
- Standard library `datetime` module