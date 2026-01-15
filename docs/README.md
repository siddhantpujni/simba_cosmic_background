# Documentation

This directory contains project documentation, including technical notes, user guides, and API documentation.

## Purpose

Documentation helps:
- Onboard new collaborators
- Document design decisions
- Explain complex algorithms
- Provide usage guides
- Track project progress

## Structure

```
docs/
├── getting_started.md       # Quick start guide
├── data_description.md      # Data formats and sources
├── analysis_pipeline.md     # Analysis workflow
├── api/                     # API documentation (auto-generated)
├── notes/                   # Technical notes and decisions
└── tutorials/               # Step-by-step guides
```

## Documentation Types

### User Documentation
- Installation instructions
- Usage examples
- Tutorials and workflows
- FAQs

### Technical Documentation
- Algorithm descriptions
- Design decisions
- Performance notes
- Known issues and limitations

### API Documentation
Generate from docstrings using Sphinx:

```bash
sphinx-apidoc -o docs/api src/
sphinx-build -b html docs/ docs/_build/
```

## Writing Guidelines

1. **Clear structure**: Use headings and sections
2. **Examples**: Include code examples and outputs
3. **Up-to-date**: Update docs when code changes
4. **Audience-aware**: Write for your target reader
5. **Linked**: Cross-reference related documents

## Documentation Tools

### Markdown
Simple, readable, version-controllable:
```markdown
# Title

## Section

Text with **bold** and *italic*.

```python
# Code example
import numpy as np
```
```

### Sphinx
For Python projects, use Sphinx:
```bash
pip install sphinx sphinx-rtd-theme
sphinx-quickstart docs/
```

### Jupyter Book
For tutorial-style documentation with notebooks.

## Maintenance

- Review docs quarterly for accuracy
- Update when adding major features
- Keep examples working with current code
- Archive outdated documentation

## Publishing

Consider hosting documentation on:
- GitHub Pages
- Read the Docs
- GitHub Wiki
- Project website
