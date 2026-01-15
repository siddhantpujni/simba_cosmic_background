# Contributing to Simba Cosmic Background

Thank you for your interest in contributing to this research project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up the development environment:
   ```bash
   conda env create -f environment.yml
   conda activate simba_cosmic_background
   pip install -e .[dev]
   ```

## Development Workflow

### Branching Strategy

- `main`: Stable code, ready for production
- Feature branches: `feature/description` or `analysis/description`
- Bug fixes: `bugfix/description`

### Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/my-new-feature
   ```

2. Make your changes following the coding standards below

3. Add tests for new functionality

4. Run tests and checks:
   ```bash
   pytest tests/
   black src/ tests/ scripts/
   flake8 src/ tests/ scripts/
   ```

5. Commit with clear messages:
   ```bash
   git commit -m "Add function to calculate cosmic background"
   ```

6. Push to your fork and create a pull request

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use `black` for automatic formatting (line length: 88)
- Use `flake8` for linting
- Add type hints for function signatures
- Maximum line length: 88 characters

### Documentation

- Write docstrings for all public functions, classes, and modules
- Use NumPy docstring format
- Include examples in docstrings
- Update README and docs when adding features

Example docstring:
```python
def calculate_background(flux, redshift, wavelength):
    """Calculate cosmic background intensity.
    
    Parameters
    ----------
    flux : np.ndarray
        Array of flux values in erg/s/cm^2/Hz
    redshift : np.ndarray
        Redshift of each source
    wavelength : float
        Wavelength in microns
        
    Returns
    -------
    intensity : float
        Background intensity in nW/m^2/sr
    uncertainty : float
        Statistical uncertainty
        
    Examples
    --------
    >>> flux = np.array([1e-29, 2e-29])
    >>> z = np.array([1.0, 2.0])
    >>> intensity, err = calculate_background(flux, z, 100.0)
    """
```

### Testing

- Write unit tests for all new functions
- Use `pytest` for testing
- Aim for >80% code coverage
- Include edge cases and error conditions
- Mark slow tests with `@pytest.mark.slow`

### Commit Messages

Write clear, descriptive commit messages:

```
Add cosmic background calculation function

- Implement calculate_background() in src/analysis/background.py
- Add unit tests in tests/analysis/test_background.py
- Update documentation with usage examples
```

Format:
- First line: Short summary (50 chars or less)
- Blank line
- Detailed description if needed
- Reference issues: "Fixes #123" or "Related to #456"

## Pull Request Process

1. Update documentation reflecting your changes
2. Add tests for new functionality
3. Ensure all tests pass
4. Update CHANGELOG.md if applicable
5. Request review from maintainers
6. Address review comments
7. Squash commits if requested

## Code Review

All contributions require code review. Reviewers will check:
- Code quality and style
- Test coverage
- Documentation completeness
- Scientific correctness
- Performance considerations

## Data Contributions

When contributing data or results:
- Do not commit large data files (>10 MB)
- Use Git LFS for medium-sized files if needed
- Document data sources and provenance
- Include metadata files
- Update data/README.md with access instructions

## Scientific Reproducibility

Ensure your contributions maintain reproducibility:
- Set random seeds explicitly
- Document software versions used
- Include configuration files for analyses
- Provide clear instructions to reproduce results
- Update documentation with any new dependencies

## Reporting Issues

When reporting issues, include:
- Clear description of the problem
- Steps to reproduce
- Expected vs. actual behavior
- System information (OS, Python version)
- Relevant code snippets or error messages

## Questions?

For questions or discussions:
- Open an issue for feature requests or bugs
- Contact maintainers directly for sensitive matters
- Join project meetings/discussions if available

## Attribution

Contributors will be acknowledged in:
- README.md contributor list
- Paper acknowledgments (for significant contributions)
- Release notes

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
