# Simba Cosmic Background

A research project investigating cosmic background phenomena using the Simba cosmological simulation and Powderday radiative transfer code.

## Project Overview

This repository contains code, data, and documentation for analyzing cosmic background radiation using state-of-the-art astrophysical simulations. The project combines large-scale cosmological simulations (Simba) with detailed radiative transfer modeling (Powderday) to study the cosmic infrared background and related phenomena.

## Repository Structure

```
.
├── literature/          # Literature review and reading notes
│   ├── papers/         # PDF files and bibliographic data
│   └── notes/          # Reading notes and summaries
├── data/               # Data storage (not tracked in git)
│   ├── raw/           # Original, immutable data files
│   ├── processed/     # Cleaned and processed datasets
│   ├── simba/         # Simba simulation outputs
│   └── powderday/     # Powderday radiative transfer outputs
├── notebooks/          # Jupyter notebooks for exploration
├── src/                # Source code for production use
├── scripts/            # Analysis and processing scripts
├── figures/            # Generated plots and visualizations
├── paper/              # Manuscript and supplementary materials
│   ├── manuscript/    # Main paper files (LaTeX/text)
│   └── supplements/   # Supplementary figures and tables
├── docs/               # Project documentation
├── tests/              # Unit and integration tests
└── results/            # Analysis outputs
    ├── analysis/      # Analysis results and logs
    └── tables/        # Generated data tables
```

## Getting Started

### Prerequisites

- Python 3.8+
- Conda (recommended for environment management)
- Git LFS (for large data files, if used)
- Access to computing cluster (for large-scale simulations)

### Installation

1. Clone this repository:
```bash
git clone https://github.com/siddhantpujni/simba_cosmic_background.git
cd simba_cosmic_background
```

2. Set up the Python environment:
```bash
# Using conda (recommended)
conda env create -f environment.yml
conda activate simba_cosmic_background

# Or using pip (core dependencies only)
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
# Core package only
pip install -e .

# With development tools (testing, jupyter, code quality)
pip install -e .[dev]

# With all optional dependencies
pip install -e .[dev,docs,analysis]
```

### Data Setup

Large datasets are not tracked in git. See `data/README.md` for instructions on downloading or accessing the required Simba and Powderday data.

## Usage

### Running Analysis Scripts

```bash
python scripts/analyze_background.py --config configs/default.yaml
```

### Jupyter Notebooks

Interactive analysis notebooks are in the `notebooks/` directory:

```bash
jupyter notebook notebooks/
```

## Development

### Code Style

This project follows PEP 8 style guidelines. Use `black` for code formatting:

```bash
black src/ scripts/
```

### Testing

Run tests using pytest:

```bash
pytest tests/
```

## Data Management

- Raw data should be kept in `data/raw/` and never modified
- Processed data goes in `data/processed/`
- Large simulation outputs use subdirectories in `data/simba/` and `data/powderday/`
- Use descriptive filenames with dates: `processed_catalog_2024-01-15.fits`

## Reproducibility

This project follows best practices for reproducible research:

- All analysis scripts are version controlled
- Software dependencies are pinned in `requirements.txt` and `environment.yml`
- Random seeds are set explicitly in analysis code
- Data processing pipelines are documented in `docs/`

## Contributing

1. Create a new branch for your feature
2. Make your changes with descriptive commit messages
3. Add tests for new functionality
4. Update documentation as needed
5. Submit a pull request

## Citation

If you use this code or data in your research, please cite:

```bibtex
@article{simba_cosmic_background,
  author = {Your Name},
  title = {Title of Paper},
  journal = {Journal Name},
  year = {2024},
  note = {In preparation}
}
```

## References

- **Simba**: Davé et al. 2019, MNRAS, 486, 2827
- **Powderday**: Narayanan et al. 2021, ApJS, 252, 12

## License

[Add appropriate license here]

## Contact

For questions or collaboration inquiries, please contact [Your Name] at [email]

## Acknowledgments

- Simba simulation team
- Powderday development team
- [Add funding acknowledgments]