# Data

This directory contains all datasets used in the project. **Large data files are not tracked in git** - see instructions below for data access and management.

## Structure

- **`raw/`**: Original, immutable data files
- **`processed/`**: Cleaned, transformed, or derived datasets
- **`simba/`**: Simba simulation outputs and catalogs
- **`powderday/`**: Powderday radiative transfer outputs

## Data Management Principles

1. **Never modify raw data** - always work on copies in `processed/`
2. **Document data provenance** - record data sources, download dates, and versions
3. **Use descriptive filenames** - include dates and processing steps
4. **Create metadata files** - add `.txt` or `.json` files describing dataset contents

## Git Ignore

Data files are excluded from git tracking via `.gitignore`. To share data:
- Use institutional data repositories (Zenodo, Figshare)
- Store on computing cluster shared directories
- Document access instructions below

## Accessing Simba Data

Simba simulation data can be accessed from:
- **Location**: [Add path to Simba data]
- **Format**: [HDF5/FITS/etc.]
- **Size**: [Estimate total size]
- **Access**: [Instructions or credentials needed]

Example directory structure:
```
data/simba/
├── snapshot_100/
├── snapshot_150/
└── catalogs/
```

## Accessing Powderday Data

Powderday outputs can be accessed from:
- **Location**: [Add path to Powderday data]
- **Format**: [HDF5/FITS/etc.]
- **Size**: [Estimate total size]
- **Access**: [Instructions]

Example directory structure:
```
data/powderday/
├── galaxy_seds/
├── images/
└── photometry/
```

## Processed Data

Processed datasets should include:
- Clear filename indicating processing steps
- Metadata file describing transformations applied
- Date of creation
- Script used to generate it (referenced in filename or metadata)

Example:
```
data/processed/
├── galaxy_catalog_cleaned_2024-01-15.fits
├── galaxy_catalog_cleaned_2024-01-15_metadata.txt
└── sed_stacks_by_mass_2024-01-20.h5
```

## Data Formats

Recommended formats for different data types:
- **Catalogs**: FITS tables or HDF5
- **Images**: FITS or HDF5
- **Time series**: HDF5 or CSV
- **Small tables**: CSV with good documentation

## Backup Strategy

- Keep backups of processed data on cluster storage
- Document backup locations in this file
- Consider using version control for small, critical datasets
