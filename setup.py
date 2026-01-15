"""Setup script for simba_cosmic_background package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="simba_cosmic_background",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@institution.edu",
    description="Analysis of cosmic background using Simba simulations and Powderday",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/siddhantpujni/simba_cosmic_background",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "astropy>=5.0.0",
        "h5py>=3.0.0",
        "seaborn>=0.11.0",
        "tqdm>=4.60.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
            "jupyter>=1.0.0",
            "jupyterlab>=3.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
        "analysis": [
            "scikit-learn>=0.24.0",
            "corner>=2.2.0",
            "emcee>=3.0.0",
        ],
    },
)
