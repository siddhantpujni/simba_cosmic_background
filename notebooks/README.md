# Notebooks

This directory contains Jupyter notebooks for exploratory data analysis, visualization, and prototyping.

## Purpose

Notebooks are ideal for:
- Exploratory data analysis (EDA)
- Quick visualizations and plots
- Prototyping analysis pipelines
- Interactive data investigation
- Demonstrating results to collaborators

## Organization

Use descriptive names with numbers for ordering:

```
notebooks/
├── 01_data_exploration.ipynb
├── 02_galaxy_selection.ipynb
├── 03_sed_analysis.ipynb
├── 04_cosmic_background_calculation.ipynb
└── 05_visualization_figures.ipynb
```

Or organize by topic:
```
notebooks/
├── exploratory/
│   ├── initial_data_check.ipynb
│   └── parameter_distributions.ipynb
├── analysis/
│   ├── background_calculation.ipynb
│   └── systematic_errors.ipynb
└── figures/
    └── paper_figures.ipynb
```

## Best Practices

1. **Clear naming**: Use descriptive names indicating the notebook's purpose
2. **Add headers**: Include purpose, author, and date at the top of each notebook
3. **Document assumptions**: Clearly state any assumptions or parameters
4. **Keep focused**: One analysis or topic per notebook
5. **Restart and run all**: Ensure notebooks run from top to bottom without errors
6. **Version control**: Commit notebooks with output cleared (`jupyter nbconvert --clear-output`)

## Notebook Template

Start each notebook with a header cell:

```markdown
# Title of Analysis

**Author:** Your Name  
**Date:** 2024-01-15  
**Purpose:** Brief description of what this notebook does

## Summary
- Key finding 1
- Key finding 2
```

## Converting to Scripts

When analysis is mature and stable, convert notebooks to Python scripts:

```bash
jupyter nbconvert --to script notebook_name.ipynb
# Then move to scripts/ and refactor as needed
```

## Running Notebooks

Start Jupyter Lab:
```bash
jupyter lab
```

Or Jupyter Notebook:
```bash
jupyter notebook
```

## Dependencies

Ensure all required packages are installed in your environment. Common packages:
- `jupyter`, `jupyterlab`
- `numpy`, `scipy`
- `matplotlib`, `seaborn`
- `pandas`
- `astropy`
- `h5py`

## Tips

- Use `%matplotlib inline` or `%matplotlib widget` for interactive plots
- Set random seeds for reproducibility: `np.random.seed(42)`
- Use `tqdm` for progress bars in long computations
- Save intermediate results to avoid recomputation
