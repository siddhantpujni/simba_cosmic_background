# Results

This directory contains outputs from analysis scripts and notebooks, including processed results, tables, and logs.

## Purpose

Store analysis outputs:
- Numerical results
- Data tables
- Analysis logs
- Intermediate calculations

## Structure

- **`analysis/`**: Analysis outputs, intermediate results, and logs
- **`tables/`**: Generated data tables for papers or reports

```
results/
├── analysis/
│   ├── background_calculation_2024-01-15.h5
│   ├── sensitivity_analysis_2024-01-20.json
│   └── logs/
│       ├── analysis_2024-01-15.log
│       └── processing_2024-01-20.log
└── tables/
    ├── table1_galaxy_properties.csv
    ├── table2_background_values.tex
    └── supplementary_table_s1.csv
```

## File Organization

### Naming Convention

Use descriptive names with dates:
```
[analysis_type]_[YYYY-MM-DD].[ext]
background_calculation_2024-01-15.h5
systematic_errors_2024-01-20.json
```

### Version Control

For iterative analysis:
```
results/analysis/
├── background_v1_2024-01-15.h5
├── background_v2_2024-01-20.h5  # After fixing bug
└── background_v3_2024-01-25.h5  # Final version
```

## File Formats

### Structured Data
- **HDF5** (`.h5`): Large numerical arrays, hierarchical data
- **JSON** (`.json`): Metadata, configuration, summary statistics
- **CSV** (`.csv`): Tables, small datasets
- **FITS** (`.fits`): Astronomical data tables

### Tables for Papers
- **LaTeX** (`.tex`): Tables for direct inclusion in papers
- **CSV** (`.csv`): Raw data for tables
- **Markdown** (`.md`): Human-readable tables

## Metadata

Include metadata files alongside results:

```
results/analysis/background_calculation_2024-01-15.h5
results/analysis/background_calculation_2024-01-15_metadata.json
```

Metadata should include:
- Script or notebook that generated the result
- Input data sources and versions
- Parameters and configuration used
- Date and time of generation
- Software versions (Python, key packages)

Example metadata JSON:
```json
{
  "generated_by": "scripts/calculate_background.py",
  "date": "2024-01-15T14:30:00",
  "input_data": "data/processed/galaxy_catalog_2024-01-10.fits",
  "parameters": {
    "wavelength": 100.0,
    "redshift_range": [0.5, 3.0],
    "min_flux": 1e-30
  },
  "environment": {
    "python": "3.9.7",
    "numpy": "1.21.2",
    "astropy": "5.0.1"
  }
}
```

## Tables for Publication

Store publication-ready tables in `tables/`:

### LaTeX Tables

```latex
% table1_galaxy_properties.tex
\begin{table}
\centering
\caption{Galaxy Properties}
\label{tab:galaxy_props}
\begin{tabular}{lcc}
\hline
Property & Value & Uncertainty \\
\hline
Mass [$M_\odot$] & $10^{11}$ & $0.1$ \\
SFR [$M_\odot$/yr] & $10$ & $2$ \\
\hline
\end{tabular}
\end{table}
```

Include in paper with `\input{../../results/tables/table1_galaxy_properties.tex}`

### CSV Tables

Keep raw data in CSV with good headers:
```csv
# Table 1: Galaxy Properties
# Units: Mass (Msun), SFR (Msun/yr), Redshift (dimensionless)
galaxy_id,mass,mass_err,sfr,sfr_err,redshift
1,1.2e11,0.1e11,15.3,2.1,1.5
2,3.4e11,0.3e11,8.7,1.5,2.1
```

## Logs

Store analysis logs in `results/analysis/logs/`:

```python
import logging

logging.basicConfig(
    filename='results/analysis/logs/analysis_2024-01-15.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## Git Tracking

- Consider `.gitignore` for large result files
- Track small summary files and metadata
- Track publication tables in git
- Use Git LFS for large binary results if needed

## Backup and Archiving

- Keep backups on cluster or institutional storage
- Archive final results with paper submission
- Document archive locations in this README
- Consider data repositories (Zenodo, Figshare) for published results

## Reproducibility

To ensure reproducibility:
1. Always include metadata with results
2. Reference the exact script/notebook version (git commit hash)
3. Document random seeds used
4. Save intermediate results for validation
5. Keep logs of all analysis runs

## Cleaning Up

Periodically review and clean:
- Remove outdated intermediate results
- Archive old versions
- Keep only final results for paper
- Document what was removed
