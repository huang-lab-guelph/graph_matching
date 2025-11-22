# Methyl Match

Automated methyl assignment for NMR spectroscopy using graph matching algorithms.

## Overview

Methyl Match is a Python library for automated methyl assignment of NMR spectra. It uses graph matching algorithms to correlate methyl-methyl NOE (Nuclear Overhauser Effect) connectivity patterns from 13C-13C-1H NOESY experimental data with structural information from protein models.

## Features

- **PDB Structure Parsing**: Extract methyl groups from protein structures
- **Multi-Format NMR Support**: Parse 13C-13C-1H methyl-methyl NOESY and 1H-13C HMQC peak lists (XEASY, NMRPipe, Sparky, CSV)
- **Graph Construction**: Build graph representations for structure and experimental data
- **Multiple Algorithms**: Support for various matching algorithms (Greedy, Optimal, ML-based)
- **Flexible Output**: Export results in multiple formats (text, CSV, PyMOL scripts)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/methyl_match.git
cd methyl_match

# Install with uv
uv sync

# Or install with pip
pip install -e .
```

## Quick Start

```python
from methyl_match.reading import PDBParser, NOESYParser, HMQCParser

# Parse PDB structure
pdb_parser = PDBParser("protein.pdb")
methyls = pdb_parser.extract_methyls()

# Parse NMR data
noesy_parser = NOESYParser("noesy_peaks.txt")
noe_peaks = noesy_parser.parse()

hmqc_parser = HMQCParser("hmqc_peaks.txt")
hmqc_peaks = hmqc_parser.parse()

print(f"Found {len(methyls)} methyl groups")
print(f"Found {len(noe_peaks)} NOE peaks")
print(f"Found {len(hmqc_peaks)} HMQC peaks")
```

### Interactive Notebooks

Explore the library interactively with [marimo](https://marimo.io) notebooks:

```bash
# Install notebook dependencies
uv sync --extra notebooks

# Launch the reading demo
cd notebooks
uv run marimo edit reading_demo.py
```

See [notebooks/README.md](notebooks/README.md) for more details.

## Development Status

**Progress: 2/6 Phases Complete (33%)**

**Phase 1: Foundation & File Readers** ✅ **COMPLETED**

- ✅ PDB structure parsing (43 methyls from ubiquitin)
- ✅ Multi-format 13C-13C-1H methyl-methyl NOESY parsing (XEASY, Sparky, CSV)
- ✅ Multi-format HMQC parsing with chemical shift matching
- ✅ 34 comprehensive tests (100% passing)
- ✅ Interactive marimo notebook: [reading_demo.py](notebooks/reading_demo.py)
- ✅ Distance and correlation matrix calculations

**Phase 2: Graph Construction** ✅ **COMPLETED**

- ✅ Methyl network construction from PDB structures
- ✅ Peak network construction from 13C-13C-1H NOESY data
- ✅ Rich feature vectors (10D methyl, 4D peak)
- ✅ Multiple edge weight functions
- ✅ 21 preprocessing tests (55 total, 100% passing)
- ✅ Interactive marimo notebook: [graph_construction_demo.py](notebooks/graph_construction_demo.py)
- ✅ NetworkX integration complete

**Next**: Phase 3 - Matching Algorithms

See [PLAN.md](PLAN.md) for the complete development roadmap.

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=methyl_match --cov-report=html

# Run specific test module
uv run pytest tests/test_reading.py -v
```

## Project Structure

```
methyl_match/
├── src/methyl_match/       # Main package
│   ├── reading/            # File parsers ✅
│   ├── preprocessing/      # Graph construction ✅
│   ├── matching/           # Matching algorithms (Phase 3)
│   └── writing/            # Output formatters (Phase 4)
├── tests/                  # Test suite ✅ (55 tests)
├── data/test/              # Example data ✅
├── notebooks/              # Interactive marimo notebooks ✅ (2 notebooks)
├── scripts/                # Utility scripts (Phase 5)
└── docs/                   # Documentation (Phase 6)
```

## Documentation

- [CLAUDE.md](CLAUDE.md): Comprehensive project documentation for AI assistants
- [PLAN.md](PLAN.md): Development roadmap and current status
- [notebooks/README.md](notebooks/README.md): Interactive notebook tutorials

## License

MIT License

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## Citation

If you use Methyl Match in your research, please cite:

```
[Citation information will be added]
```

## Contact

For questions and feedback, please open an issue on GitHub.
