# Scripts

This directory contains executable Python scripts for data processing, analysis, and batch operations.

## Purpose

Scripts are for:
- Running complete analysis pipelines
- Batch processing of data
- Command-line tools for common tasks
- One-off analysis tasks
- Reproducing results for the paper

## Organization

Use clear, descriptive names:

```
scripts/
├── download_data.py           # Data acquisition
├── process_simba_snapshots.py # Data processing
├── calculate_background.py    # Main analysis
├── generate_paper_figures.py  # Figure creation
└── run_sensitivity_analysis.py # Tests and validation
```

## Script Structure

Each script should be executable and include:

1. Shebang line
2. Docstring describing purpose
3. Command-line argument parsing
4. Main function
5. `if __name__ == "__main__":` block

### Template

```python
#!/usr/bin/env python
"""Process Simba snapshots to extract galaxy catalogs.

This script reads Simba simulation snapshots and creates
processed catalogs for downstream analysis.

Usage:
    python process_simba_snapshots.py --snapshot 150 --output catalog.fits

Author: Your Name
Date: 2024-01-15
"""

import argparse
import logging
from pathlib import Path
from src.data import load_simba_snapshot, create_catalog

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process Simba snapshots"
    )
    parser.add_argument(
        "--snapshot",
        type=int,
        required=True,
        help="Snapshot number to process"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output catalog filename"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    return parser.parse_args()


def main(snapshot: int, output: Path, verbose: bool = False):
    """Main processing function.
    
    Parameters
    ----------
    snapshot : int
        Snapshot number
    output : Path
        Output file path
    verbose : bool
        Enable verbose logging
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    
    logger.info(f"Processing snapshot {snapshot}")
    
    # Your analysis code here
    data = load_simba_snapshot(snapshot)
    catalog = create_catalog(data)
    catalog.write(output)
    
    logger.info(f"Saved catalog to {output}")


if __name__ == "__main__":
    args = parse_args()
    main(
        snapshot=args.snapshot,
        output=args.output,
        verbose=args.verbose
    )
```

## Running Scripts

From the repository root:

```bash
python scripts/process_simba_snapshots.py --snapshot 150 --output data/processed/catalog.fits
```

With logging to file:
```bash
python scripts/calculate_background.py 2>&1 | tee logs/analysis.log
```

## Best Practices

1. **Use argparse**: Make scripts configurable via command-line arguments
2. **Add logging**: Use Python's `logging` module, not `print()`
3. **Handle errors**: Include try-except blocks and meaningful error messages
4. **Document usage**: Include examples in docstrings
5. **Make reproducible**: Set random seeds, log parameters used
6. **Save outputs**: Write results to files with timestamps
7. **Progress indicators**: Use `tqdm` for long-running operations

## Configuration Files

For complex analyses, use configuration files:

```
scripts/
├── calculate_background.py
└── configs/
    ├── default.yaml
    └── high_resolution.yaml
```

Load with a helper function:
```python
import yaml
from pathlib import Path

def load_config(config_path):
    """Load YAML configuration file.
    
    Parameters
    ----------
    config_path : str or Path
        Path to configuration file
        
    Returns
    -------
    dict
        Configuration dictionary
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path) as f:
        return yaml.safe_load(f)

# Usage in script
config = load_config('configs/default.yaml')
```

## Batch Processing

For cluster computing, create submission scripts:

```
scripts/
├── batch_process.py
└── submit_jobs.sh  # SLURM/PBS submission script
```
