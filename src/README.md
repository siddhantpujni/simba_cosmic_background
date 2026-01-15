# Source Code

This directory contains production-quality Python code for the project. Code here should be well-tested, documented, and reusable.

## Purpose

The `src/` directory is for:
- Reusable functions and classes
- Data processing pipelines
- Analysis algorithms
- Utility modules
- Package that can be imported in notebooks and scripts

## Structure

Organize code into logical modules:

```
src/
├── __init__.py
├── data/
│   ├── __init__.py
│   ├── loaders.py      # Data loading utilities
│   └── processors.py   # Data processing functions
├── analysis/
│   ├── __init__.py
│   ├── background.py   # Cosmic background calculations
│   └── statistics.py   # Statistical analysis tools
├── visualization/
│   ├── __init__.py
│   └── plotting.py     # Plotting utilities
└── utils/
    ├── __init__.py
    ├── constants.py    # Physical constants
    └── helpers.py      # General utility functions
```

## Code Quality Standards

### Documentation
- Use docstrings for all functions, classes, and modules
- Follow NumPy or Google docstring format
- Include examples in docstrings

### Style
- Follow PEP 8 style guidelines
- Use type hints for function signatures
- Maximum line length: 88 characters (Black default)

### Testing
- Write unit tests for all functions
- Aim for >80% code coverage
- Place tests in parallel structure in `tests/` directory

## Example Module Structure

```python
"""Module for cosmic background analysis.

This module provides functions for calculating cosmic background
radiation from simulation outputs.
"""

import numpy as np
from typing import Tuple, Optional

def calculate_background(
    flux: np.ndarray,
    redshift: np.ndarray,
    wavelength: float
) -> Tuple[float, float]:
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
    # Implementation here
    pass
```

## Installation

To use this package in development mode:

```bash
pip install -e .
```

Then import in notebooks or scripts:

```python
from src.analysis import calculate_background
from src.data import load_simba_catalog
```

## Setup Configuration

Create a `setup.py` or `pyproject.toml` in the repository root to make this an installable package.

## Dependencies

List module dependencies in:
- `requirements.txt` - for pip
- `environment.yml` - for conda
- `setup.py` or `pyproject.toml` - for package installation
