"""Pytest configuration and fixtures for simba_cosmic_background tests.

This file contains shared fixtures and configuration for pytest.
"""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_catalog():
    """Provide a sample galaxy catalog for testing.
    
    Returns
    -------
    dict
        Dictionary containing sample galaxy properties:
        - flux: array of flux values in erg/s/cm^2/Hz
        - redshift: array of redshift values
        - mass: array of stellar masses in Msun
    """
    return {
        'flux': np.array([1e-29, 2e-29, 3e-29]),
        'redshift': np.array([1.0, 1.5, 2.0]),
        'mass': np.array([1e10, 1e11, 1e12])
    }


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary directory for test outputs.
    
    Parameters
    ----------
    tmp_path : Path
        Pytest fixture providing temporary directory
        
    Returns
    -------
    Path
        Path to temporary output directory
    """
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return output_dir


@pytest.fixture
def sample_wavelengths():
    """Provide a sample array of wavelengths for testing.
    
    Returns
    -------
    np.ndarray
        Array of wavelengths in microns
    """
    return np.array([10.0, 50.0, 100.0, 250.0, 500.0])


# Configure pytest marks
def pytest_configure(config):
    """Configure custom pytest markers.
    
    Markers:
    --------
    slow : Mark tests that take significant time to run (>1 second)
        Usage: @pytest.mark.slow
        Run all except slow tests: pytest -m "not slow"
        Run only slow tests: pytest -m slow
        
    integration : Mark integration tests that test multiple components
        Usage: @pytest.mark.integration
        Run only unit tests: pytest -m "not integration"
        Run only integration tests: pytest -m integration
    
    Examples:
    ---------
    >>> @pytest.mark.slow
    >>> def test_large_simulation():
    >>>     # Test that takes several seconds
    >>>     pass
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
