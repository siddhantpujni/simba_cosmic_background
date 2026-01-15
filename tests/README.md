# Tests

This directory contains unit tests, integration tests, and test data for the project.

## Purpose

Tests ensure:
- Code correctness
- Reproducibility
- Regression prevention
- Documentation through examples

## Structure

Mirror the `src/` directory structure:

```
tests/
├── __init__.py
├── conftest.py              # pytest configuration and fixtures
├── test_data/              # Small test datasets
│   ├── mock_catalog.fits
│   └── test_snapshot.h5
├── data/
│   ├── test_loaders.py
│   └── test_processors.py
├── analysis/
│   ├── test_background.py
│   └── test_statistics.py
└── utils/
    └── test_helpers.py
```

## Testing Framework

This project uses `pytest` for testing.

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/analysis/test_background.py

# Run with coverage
pytest --cov=src tests/

# Run with verbose output
pytest -v tests/

# Run only fast tests
pytest -m "not slow" tests/
```

### Writing Tests

Example test file:

```python
"""Tests for cosmic background calculations."""

import numpy as np
import pytest
from src.analysis.background import calculate_background


def test_calculate_background_basic():
    """Test basic background calculation."""
    flux = np.array([1e-29, 2e-29, 3e-29])
    redshift = np.array([1.0, 1.5, 2.0])
    wavelength = 100.0
    
    intensity, uncertainty = calculate_background(flux, redshift, wavelength)
    
    assert intensity > 0
    assert uncertainty > 0
    assert intensity > uncertainty


def test_calculate_background_empty():
    """Test background calculation with empty input."""
    flux = np.array([])
    redshift = np.array([])
    wavelength = 100.0
    
    with pytest.raises(ValueError):
        calculate_background(flux, redshift, wavelength)


def test_calculate_background_negative_flux():
    """Test that negative flux raises error."""
    flux = np.array([1e-29, -1e-29])
    redshift = np.array([1.0, 1.5])
    wavelength = 100.0
    
    with pytest.raises(ValueError, match="negative"):
        calculate_background(flux, redshift, wavelength)


@pytest.mark.slow
def test_calculate_background_large_dataset():
    """Test with large dataset (marked as slow)."""
    flux = np.random.exponential(1e-29, size=1000000)
    redshift = np.random.uniform(0.5, 3.0, size=1000000)
    wavelength = 100.0
    
    intensity, uncertainty = calculate_background(flux, redshift, wavelength)
    assert intensity > 0
```

## Test Types

### Unit Tests
Test individual functions in isolation:
- Fast execution
- Well-defined inputs and outputs
- Mock external dependencies

### Integration Tests
Test multiple components together:
- Data loading + processing
- End-to-end pipelines
- File I/O operations

### Regression Tests
Ensure results don't change unexpectedly:
- Save known-good outputs
- Compare against reference values
- Use `pytest.approx()` for floats

## Fixtures

Define reusable test data in `conftest.py`:

```python
import pytest
import numpy as np

@pytest.fixture
def sample_catalog():
    """Provide a sample galaxy catalog for testing."""
    return {
        'flux': np.array([1e-29, 2e-29, 3e-29]),
        'redshift': np.array([1.0, 1.5, 2.0]),
        'mass': np.array([1e10, 1e11, 1e12])
    }

@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir
```

## Test Data

Keep test data small:
- Use `tests/test_data/` for small files
- Generate synthetic data in fixtures
- Mock large datasets

## Coverage

Aim for high code coverage:

```bash
# Generate coverage report
pytest --cov=src --cov-report=html tests/

# View report
open htmlcov/index.html
```

Target: >80% coverage for critical code.

## Best Practices

1. **Test names**: Descriptive names like `test_function_name_scenario`
2. **One assertion focus**: Test one thing per test function
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Use fixtures**: Avoid duplication
5. **Test edge cases**: Empty inputs, zero, negative values
6. **Mark slow tests**: Use `@pytest.mark.slow`
7. **Parametrize**: Test multiple inputs efficiently

### Parametrized Tests

```python
@pytest.mark.parametrize("wavelength,expected", [
    (100.0, 1.0),
    (250.0, 2.5),
    (500.0, 5.0),
])
def test_wavelength_scaling(wavelength, expected):
    result = some_function(wavelength)
    assert result == pytest.approx(expected)
```

## Continuous Integration

Tests run automatically on:
- Every commit
- Pull requests
- Before merging

Configure in `.github/workflows/tests.yml`.

## Documentation

Tests serve as documentation:
- Show expected usage
- Demonstrate edge cases
- Provide examples
