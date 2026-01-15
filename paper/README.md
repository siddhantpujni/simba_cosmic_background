# Paper

This directory contains the manuscript and supplementary materials for publication.

## Structure

- **`manuscript/`**: Main paper files (LaTeX, Word, or Markdown)
- **`supplements/`**: Supplementary materials, appendices, and additional figures

## Organization

```
paper/
├── manuscript/
│   ├── main.tex                 # Main LaTeX file
│   ├── abstract.tex             # Abstract
│   ├── introduction.tex         # Introduction section
│   ├── methods.tex              # Methods section
│   ├── results.tex              # Results section
│   ├── discussion.tex           # Discussion section
│   ├── references.bib           # Bibliography
│   └── figures/                 # Symlink to ../../figures/paper/
├── supplements/
│   ├── appendix_a.tex
│   ├── supplementary_figures.pdf
│   └── supplementary_tables.pdf
└── submission/                   # Files for journal submission
    └── response_to_reviewers.txt
```

## LaTeX Setup

### Required Packages

Common packages for astrophysics papers:

```latex
\usepackage{graphicx}   % Figures
\usepackage{amsmath}    % Math
\usepackage{natbib}     % Citations
\usepackage{hyperref}   % Links
\usepackage{aas_macros} % AAS journal macros
```

### Journal Templates

Download appropriate templates:
- **ApJ/AAS**: https://journals.aas.org/authors/aastex.html
- **MNRAS**: https://academic.oup.com/mnras/pages/General_Instructions
- **A&A**: https://www.aanda.org/for-authors/latex-issues

### Building the Paper

Using `pdflatex`:
```bash
cd paper/manuscript
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Using `latexmk` (recommended):
```bash
cd paper/manuscript
latexmk -pdf main.tex
```

### Bibliography Management

Keep references in `references.bib`:

```bibtex
@article{Dave2019,
  author = {Davé, R. and Anglés-Alcázar, D. and Narayanan, D. and others},
  title = {SIMBA: Cosmological simulations with black hole growth},
  journal = {MNRAS},
  year = {2019},
  volume = {486},
  pages = {2827},
  doi = {10.1093/mnras/stz937}
}
```

Use tools like:
- **NASA ADS**: Export BibTeX citations
- **Zotero/Mendeley**: Manage references
- **BibDesk** (Mac): BibTeX editor

## Writing Workflow

1. **Outline first**: Create section structure before writing
2. **One section per file**: Easier to manage and version control
3. **Regular commits**: Commit progress frequently
4. **Clear commit messages**: "Add discussion of systematic errors"
5. **Track changes**: Use `latexdiff` for revision tracking

### Tracking Revisions

```bash
latexdiff old_version.tex new_version.tex > diff.tex
pdflatex diff.tex
```

## Figures in Paper

Link or copy final figures to paper directory:

```bash
cd paper/manuscript
ln -s ../../figures/paper/ figures
```

Include in LaTeX:
```latex
\begin{figure}
\includegraphics[width=\columnwidth]{figures/figure1_schematic.pdf}
\caption{Schematic of the analysis pipeline.}
\label{fig:schematic}
\end{figure}
```

## Supplementary Materials

Place in `supplements/`:
- Additional figures not in main text
- Supplementary tables
- Appendices with technical details
- Extended methodology descriptions

## Collaboration

For collaborative writing:
- Use Overleaf for real-time collaboration
- Or use git branches for parallel writing
- Define clear section ownership
- Regular team reviews of drafts

## Submission Checklist

Before submission:
- [ ] All figures in correct format (PDF/EPS)
- [ ] All references complete and formatted
- [ ] Abstract within word limit
- [ ] Acknowledgments section complete
- [ ] Data availability statement included
- [ ] Author affiliations and ORCID IDs
- [ ] Cover letter prepared
- [ ] Supplementary materials organized
- [ ] Code/data release prepared (if required)

## Version Control

Use git tags for major milestones:
```bash
git tag -a v1.0-submission -m "Initial submission"
git tag -a v2.0-revision -m "Revised after referee comments"
```

## Archiving

Upon acceptance:
- Archive final PDF and LaTeX source
- Deposit code on Zenodo/GitHub release
- Submit to arXiv if appropriate
- Update README with publication details
