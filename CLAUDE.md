# Claude Context: Methyl Assignment Project

This document provides AI assistants with essential context about this repository.

## Project Purpose

This project is a Python library for automated methyl assignment of NMR spectra. It uses graph matching algorithms to correlate NOE (Nuclear Overhauser Effect) connectivity patterns from experimental NMR data with structural information from protein models.

The library has three main components:
1. **File Readers**: Parse PDB structures and NMR spectra files
2. **Graph Matching**: Match experimental connectivity to structural connectivity
3. **Output Writers**: Export assignment results in various formats

## Architecture Overview

### Core Problem
Match experimental NMR peak networks (with NOE connectivity) to structural methyl networks derived from protein structures. This is a graph-to-graph matching problem where:
- **Graph A**: NMR peaks as nodes, NOE correlations as edges
- **Graph B**: Methyl groups in protein structure as nodes, spatial distances as edges

### Technology Stack
- **Language**: Python 3.13
- **Package Manager**: uv
- **Key Libraries**:
  - NumPy/SciPy - Numerical computing
  - NetworkX or PyTorch Geometric - Graph processing
  - BioPython - PDB file parsing
  - PyYAML - Configuration files
- **Testing**: pytest
- **Formatting**: black, isort
- **Linting**: ruff

## Code Organization

```
src/
├── methyl_match/
│   ├── __init__.py
│   ├── matching/           # Graph matching algorithms
│   ├── reading/            # File parsers (PDB, NOESY, HMQC)
│   ├── writing/            # Output formatters
│   └── preprocessing/      # Graph construction from raw data

scripts/
├── run_yme1l.py            # Example: Run specific protein
└── run.py                  # General execution script

data/
├── sample1/
│   ├── structure.pdb       # Protein structure
│   ├── noesy.txt           # NOE connectivity data
│   └── config.yaml         # Analysis parameters
└── sample2/
    ├── structure.pdb
    ├── noesy.txt
    └── config.yaml

models/                      # Trained models (if using ML)
├── model1.pkl
└── model2.pkl

tests/
├── test_reading.py
├── test_preprocessing.py
├── test_matching.py
├── test_writing.py
└── integration/            # End-to-end tests
```
## Key Components

### 1. reading Module
- **Purpose**: Parse specialized NMR and structural files
- **Key Classes**:
  - `PDBParser` - Extracts methyl groups from PDB structure files
  - `NOESYParser` - Parses NOESY peak lists (NOE connectivity)
  - `HMQCParser` - Parses HMQC peak lists (chemical shifts)
- **Supported Formats**: XEASY, NMRPipe, Sparky, CSV

### 2. preprocessing Module
- **Purpose**: Convert parsed data into graph representations
- **Key Classes**:
  - `MethylNetworkBuilder` - Creates graph from PDB methyl groups (nodes=methyls, edges=distances)
  - `PeakNetworkBuilder` - Creates graph from NMR peaks (nodes=peaks, edges=NOE correlations)
- **Graph Features**:
  - Node features: positions, chemical shifts, residue types
  - Edge features: distances, NOE intensities

### 3. matching Module
- **Purpose**: Perform graph-to-graph matching
- **Key Classes**:
  - `GraphMatcher` - Abstract base class for matching algorithms
  - `GreedyMatcher` - Simple greedy matching
  - `OptimalMatcher` - Optimal matching (Hungarian algorithm)
  - `MLMatcher` - Machine learning-based matching (if applicable)
- **Output**: Mapping between NMR peaks and structural methyl groups

### 4. writing Module
- **Purpose**: Export assignment results
- **Key Classes**:
  - `TextFormatter` - Human-readable text output
  - `CSVFormatter` - Tabular CSV output
  - `PyMOLFormatter` - Visualization scripts for PyMOL
- **Output Information**: Assignments, confidence scores, validation metrics

## Common Tasks

### Running the Application
```bash
# Using uv (recommended)
uv run python scripts/run.py --data-dir data/sample1

# For specific protein example
uv run python scripts/run_yme1l.py

# Direct Python
python -m methyl_match --data-dir data/sample1
```

### Running Tests
```bash
# All tests
uv run pytest

# With coverage
uv run pytest --cov=methyl_match --cov-report=html

# Specific module tests
uv run pytest tests/test_reading.py -v
uv run pytest tests/test_matching.py -v

# Integration tests
uv run pytest tests/integration/ -v
```

### Code Formatting & Linting
```bash
# Format code
uv run black src/ tests/

# Sort imports
uv run isort src/ tests/

# Lint
uv run ruff check src/ tests/

# Fix auto-fixable lint issues
uv run ruff check --fix src/ tests/
```

## Configuration

Configuration files are stored in each data sample directory as `config.yaml`. Key configuration options:

- **matching_algorithm**: Algorithm to use (greedy, optimal, ml)
- **distance_cutoff**: Maximum distance for structural edges (Å)
- **noe_threshold**: Minimum NOE intensity to include
- **confidence_threshold**: Minimum confidence for assignments
- **output_format**: Output format (text, csv, pymol, all)


## Data Flow

1. **Input**: User provides data directory containing:
   - `structure.pdb` - Protein structure
   - `noesy.txt` - NOE connectivity data
   - `config.yaml` - Analysis parameters

2. **Reading**: Parse files into structured data
   - Extract methyl groups from PDB
   - Parse NOESY peak lists and correlations
   - Load configuration parameters

3. **Preprocessing**: Build graph representations
   - Methyl network: nodes=methyls, edges=spatial distances
   - Peak network: nodes=peaks, edges=NOE correlations

4. **Matching**: Apply graph matching algorithm
   - Compare graph topologies
   - Score potential assignments
   - Find optimal mapping

5. **Output**: Write results to files
   - Text summary of assignments
   - CSV table for downstream analysis
   - PyMOL script for visualization


## Testing

Run tests: `uv run pytest tests/ -v`

Basic import test:
```python
from methyl_match.reading import PDBParser, NOESYParser, HMQCParser
from methyl_match.preprocessing import MethylNetworkBuilder, PeakNetworkBuilder
from methyl_match.matching import GraphMatcher, GreedyMatcher, OptimalMatcher
from methyl_match.writing import TextFormatter, CSVFormatter, PyMOLFormatter
```

Example test structure:
```python
def test_pdb_parser():
    parser = PDBParser("data/sample1/structure.pdb")
    methyls = parser.extract_methyls()
    assert len(methyls) > 0
    assert all(hasattr(m, 'residue_name') for m in methyls)
```


## Development Notes

- **Code Style**: Follow PEP 8
- **Type Hints**: Use type hints for all function signatures
- **Docstrings**: Use Google or NumPy style docstrings
- **Error Handling**: Use specific exceptions, avoid bare `except:`
- **Logging**: Use the `logging` module instead of print statements

## Important Conventions

- **Methyl Group Naming**: Follow IUPAC naming (e.g., LEU-CD1, VAL-CG1)
- **Coordinate System**: PDB uses Ångström units
- **Graph Representation**: Use consistent node/edge feature ordering
- **File Naming**: `<protein>_<experiment>.txt` convention
- **Confidence Scores**: Range from 0.0 (uncertain) to 1.0 (certain)

## When Modifying

### Add New Feature
1. Create issue/branch describing the feature
2. Add tests first (TDD approach)
3. Implement feature in appropriate module
4. Update documentation and examples
5. Ensure all tests pass

### Fix Bug
1. Write failing test that reproduces bug
2. Fix the bug
3. Verify test now passes
4. Add regression test if needed

### Add New Matching Algorithm
1. Inherit from `GraphMatcher` base class
2. Implement required methods: `match()`, `score()`
3. Add to algorithm registry in config
4. Add unit tests with known-good examples
5. Update documentation

### Add New File Format
1. Create parser class in `reading/` module
2. Implement format detection and parsing
3. Add comprehensive tests with sample files
4. Update supported formats list in docs


## Project Structure Template

When adding new modules, follow this structure:

```python
"""
Module description.

This module provides [functionality description].
"""

import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class ClassName:
    """
    Brief description.

    Detailed description of the class purpose and usage.

    Attributes:
        attr1: Description
        attr2: Description
    """

    def __init__(self, param1: str, param2: int = 0):
        """
        Initialize ClassName.

        Args:
            param1: Description
            param2: Description (default: 0)
        """
        self.attr1 = param1
        self.attr2 = param2

    def method(self, param: str) -> Optional[str]:
        """
        Brief description.

        Args:
            param: Description

        Returns:
            Description of return value

        Raises:
            ValueError: When [condition]
        """
        try:
            # Implementation
            pass
        except Exception as e:
            logger.error(f"Error in method: {e}")
            raise


def function_name(param1: str, param2: Optional[int] = None) -> Dict:
    """
    Brief description.

    Args:
        param1: Description
        param2: Description (default: None)

    Returns:
        Description of return value
    """
    # Implementation
    pass
```

## Common Issues & Solutions

### Issue: Parser fails on specific file format
- **Cause**: Format variant not yet supported
- **Solution**: Check file format, add format detection logic, or convert to supported format

### Issue: Graph matching produces poor results
- **Cause**: Insufficient NOE data or incorrect distance cutoffs
- **Solution**: Adjust `distance_cutoff` and `noe_threshold` parameters, verify data quality

### Issue: Import errors after installation
- **Cause**: Dependencies not installed
- **Solution**: Run `uv sync` to ensure all dependencies are installed

## Resources

- **NMR File Formats**: XEASY, NMRPipe, Sparky documentation
- **Graph Matching**: Hungarian algorithm, optimal transport
- **BioPython PDB**: https://biopython.org/wiki/The_Biopython_Structural_Bioinformatics_FAQ
- **Related Tools**: PINE, FLYA, MAGIC for NMR assignment

---

**Last Updated**: 2025-11-21
**Project**: Methyl Assignment via Graph Matching
