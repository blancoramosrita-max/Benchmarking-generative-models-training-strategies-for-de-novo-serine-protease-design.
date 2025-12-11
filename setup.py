#!/usr/bin/env python3
"""
Setup script for Serine Protease Generation package
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt')) as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="serine-protease-generation",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@institution.edu",
    description="Serine protease sequence generation using transformer models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/serine-protease-generation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.12.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "ipywidgets>=7.7.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ],
        "structures": [
            "colabfold>=1.5.0",
            "biotite>=0.36.0",
            "mdanalysis>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sp-train-specialist=specialist_model:main",
            "sp-train-generalist=generalist_model:main",
            "sp-curate-data=data_curation:main",
            "sp-evaluate=evaluation:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)