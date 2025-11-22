# Development Plan: Methyl Assignment NMR Library (methyl_match)

## Project Overview

**Package Name**: `methyl_match`
**Purpose**: Automated methyl assignment of NMR spectra using graph matching algorithms
**Approach**: Multi-algorithm library supporting both classical and ML-based methods
**Starting Point**: Fresh implementation based on CLAUDE.md specification

---

## Development Phases

### Phase 1: Foundation & File Readers ✅ COMPLETED

**Goal**: Create robust file parsing infrastructure with real data validation

**Tasks**:
1. ✅ Create PLAN.md for project tracking
2. ✅ Download test data files (PDB structures, NOESY/HMQC peak lists)
3. ✅ Set up project directory structure
4. ✅ Implement `reading/` module:
   - ✅ `PDBParser` - Extract methyl groups from PDB structures (263 lines)
   - ✅ `NOESYParser` - Parse NOESY peak lists (XEASY, NMRPipe, Sparky, CSV) (403 lines)
   - ✅ `HMQCParser` - Parse HMQC chemical shift data (378 lines)
5. ✅ Write comprehensive unit tests using downloaded test files (34 tests)
6. ✅ Verify all parsers work correctly with real data (100% passing)
7. ✅ Create interactive marimo notebook demonstrating functionality

**Deliverables**:
- ✅ `src/methyl_match/reading/` module with all parsers
- ✅ `tests/test_reading.py` with comprehensive test coverage (34 tests, all passing)
- ✅ `data/test/` with sample PDB, NOESY, and HMQC files
- ✅ All tests passing (`uv run pytest tests/test_reading.py -v`)
- ✅ `notebooks/reading_demo.py` - Interactive marimo notebook
- ✅ `pyproject.toml` with all dependencies configured
- ✅ `README.md` with quick start guide

**Success Criteria**: ✅ ALL MET
- ✅ Can parse methyl groups from real PDB structures (43 methyls from ubiquitin)
- ✅ Can parse multiple NOESY/HMQC file formats (auto-detection working)
- ✅ Tests demonstrate correct parsing with real data (100% pass rate)
- ✅ Code follows project conventions (type hints, docstrings, error handling)

**Results**:
- **PDB Parser**: Successfully extracts 43 methyl groups from ubiquitin structure
- **NOESY Parser**: Parses 20 NOE peaks with format auto-detection
- **HMQC Parser**: Parses 20 HMQC peaks with chemical shift matching
- **Test Coverage**: 34 tests covering all major functionality
- **Interactive Demo**: Full marimo notebook with visualizations

---

### Phase 2: Graph Construction

**Goal**: Convert parsed data into graph representations for matching

**Tasks**:
1. Implement `preprocessing/` module:
   - `MethylNetworkBuilder` - Create structure graph from PDB methyls
   - `PeakNetworkBuilder` - Create experimental graph from NMR peaks
2. Define node and edge feature representations
3. Implement distance calculation and edge weight logic
4. Write unit tests for graph builders

**Deliverables**:
- [ ] `src/methyl_match/preprocessing/` module
- [ ] Graph representations with:
  - Nodes: methyl groups (structure) or peaks (experimental)
  - Edges: spatial distances (structure) or NOE correlations (experimental)
  - Features: coordinates, residue types, chemical shifts, intensities
- [ ] `tests/test_preprocessing.py` with unit tests
- [ ] Integration with Phase 1 parsers

**Success Criteria**:
- Can build structure graph from parsed PDB
- Can build experimental graph from parsed NOESY/HMQC
- Graphs contain appropriate node/edge features
- Edge weights computed correctly based on distances/correlations

---

### Phase 3: Matching Algorithms

**Goal**: Implement multiple graph matching algorithms with unified interface

**Tasks**:
1. Create `matching/` module with base class:
   - `GraphMatcher` - Abstract base class defining interface
2. Implement classical algorithms:
   - `GreedyMatcher` - Fast greedy matching
   - `OptimalMatcher` - Hungarian algorithm for optimal assignment
3. (Optional) Implement ML-based algorithm:
   - `MLMatcher` - Deep learning approach (DGMC or similar)
4. Add algorithm selection and configuration logic
5. Write unit tests with known-good test cases

**Deliverables**:
- [ ] `src/methyl_match/matching/` module
- [ ] At least 2 working algorithms (Greedy + Hungarian)
- [ ] Unified interface for easy algorithm swapping
- [ ] `tests/test_matching.py` with unit tests
- [ ] Validation against synthetic data with known answers

**Success Criteria**:
- All algorithms implement common interface
- Produce assignment mappings (peak → methyl)
- Include confidence scores for assignments
- Tests verify correctness on synthetic examples

---

### Phase 4: Output & Utilities

**Goal**: Export assignment results in multiple formats

**Tasks**:
1. Implement `writing/` module:
   - `TextFormatter` - Human-readable summary reports
   - `CSVFormatter` - Tabular data for analysis
   - `PyMOLFormatter` - Visualization scripts with confidence coloring
2. Add confidence scoring logic
3. Add result validation and quality metrics
4. Write unit tests for formatters

**Deliverables**:
- [ ] `src/methyl_match/writing/` module
- [ ] Support for 3+ output formats
- [ ] Confidence scores and quality metrics
- [ ] `tests/test_writing.py` with unit tests
- [ ] Example output files demonstrating each format

**Success Criteria**:
- Can export results in all specified formats
- Output includes assignments, confidence scores, validation metrics
- PyMOL scripts correctly visualize assignments
- Formats are compatible with downstream analysis tools

---

### Phase 5: Scripts & Configuration

**Goal**: Create user-friendly CLI and configuration system

**Tasks**:
1. Create `scripts/` directory with executables:
   - `run.py` - General execution script
   - `run_yme1l.py` - Specific protein example
2. Implement configuration system:
   - `config.yaml` parsing with defaults
   - CLI argument parsing
   - Configuration validation
3. Add example configurations
4. Create usage documentation

**Deliverables**:
- [ ] `scripts/run.py` and example scripts
- [ ] Configuration file support (`config.yaml`)
- [ ] CLI with argument parsing
- [ ] Example configuration files in `data/sample1/` and `data/sample2/`
- [ ] Usage documentation in README

**Success Criteria**:
- Can run complete workflow from command line
- Configuration files override defaults appropriately
- Clear error messages for invalid inputs
- Easy to use for new users

---

### Phase 6: Testing & Documentation

**Goal**: Production-ready library with comprehensive documentation

**Tasks**:
1. Add integration tests:
   - End-to-end workflow tests
   - Multi-algorithm comparison tests
   - Edge case and error handling tests
2. Create documentation:
   - README.md with quickstart guide
   - API documentation
   - Jupyter notebook tutorial
3. Add example workflows with real proteins
4. Set up CI/CD (GitHub Actions or similar)
5. Add contribution guidelines

**Deliverables**:
- [ ] `tests/integration/` with end-to-end tests
- [ ] Comprehensive README.md
- [ ] Jupyter notebook tutorial in `notebooks/`
- [ ] API documentation (Sphinx or similar)
- [ ] CI/CD configuration
- [ ] CONTRIBUTING.md guidelines

**Success Criteria**:
- 80%+ test coverage
- All examples run successfully
- Documentation is clear and comprehensive
- CI/CD runs tests automatically
- Ready for public release

---

## Current Status

- **Completed Phase**: Phase 1 - Foundation & File Readers ✅
- **Next Phase**: Phase 2 - Graph Construction
- **Branch**: `new-python-project`
- **Last Updated**: 2025-11-21

### Phase 1 Summary

**Completed Components**:
- ✅ **PDBParser** (263 lines): Extracts methyl groups (LEU, VAL, ILE, ALA, THR, MET)
- ✅ **NOESYParser** (403 lines): Multi-format support with auto-detection
- ✅ **HMQCParser** (378 lines): Chemical shift parsing and matching
- ✅ **Test Suite** (34 tests): 100% pass rate with real data
- ✅ **Marimo Notebook**: Interactive demo with visualizations
- ✅ **Documentation**: README, PLAN.md, notebooks/README.md

**Key Achievements**:
- Parsed 43 methyl groups from ubiquitin structure (1UBQ)
- Auto-detection working for XEASY, Sparky, CSV formats
- Distance matrix calculations (43×43 methyls)
- NOE correlation matrices
- Chemical shift matching with tolerances
- 100% cross-validation agreement between NOESY and HMQC data

---

## Notes

### Key Design Decisions

1. **Package Name**: `methyl_match` (per CLAUDE.md specification)
2. **Multi-Algorithm Support**: Library will include multiple matching algorithms, not just one approach
3. **Fresh Implementation**: Building from scratch based on CLAUDE.md, not restoring previous code
4. **Test-Driven**: Using real downloaded data files for validation from the start

### Dependencies (from pyproject.toml)

**Core Libraries**:
- NumPy (>=1.24.0) - Numerical computing
- SciPy (>=1.10.0) - Scientific computing
- pandas (>=2.0.0) - Data manipulation
- BioPython (>=1.80) - PDB parsing
- PyYAML (>=6.0) - Configuration files

**Visualization & Notebooks**:
- marimo (>=0.18.0) - Interactive notebooks
- matplotlib (>=3.10.7) - Plotting
- seaborn (>=0.13.2) - Statistical visualizations

**Development & Testing**:
- pytest (>=7.4.0) - Testing framework
- pytest-cov (>=4.1.0) - Coverage reporting
- black (>=23.0.0) - Code formatting
- ruff (>=0.1.0) - Linting
- isort (>=5.12.0) - Import sorting

**Future (for Phase 3)**:
- NetworkX or PyTorch Geometric - Graph processing
- PyTorch - ML algorithms (optional)

### File Naming Conventions

- PDB structures: `<protein>.pdb`
- NOESY data: `<protein>_noesy.txt` or `<protein>_noesy.csv`
- HMQC data: `<protein>_hmqc.txt` or `<protein>_hmqc.csv`
- Configuration: `config.yaml` in data directory

### Testing Strategy

- **Unit Tests**: Each module tested independently
- **Integration Tests**: End-to-end workflow validation
- **Test Data**: Real PDB structures and NMR data from public repositories
- **Validation**: Compare against known assignments where available

---

## Resources

- **CLAUDE.md**: Comprehensive project specification
- **PDB Database**: https://www.rcsb.org/
- **BMRB**: Biological Magnetic Resonance Data Bank (NMR data)
- **NMR File Formats**: XEASY, NMRPipe, Sparky documentation
- **Graph Matching**: Hungarian algorithm, optimal transport literature

---

## Success Metrics

**Phase 1**: Parsers work with real data, tests pass
**Phase 2**: Can build graphs from parsed data
**Phase 3**: At least 2 algorithms produce valid assignments
**Phase 4**: Results exportable in 3+ formats
**Phase 5**: End-to-end workflow runs from CLI
**Phase 6**: Ready for public release (docs, tests, CI/CD)

---

**Next Steps**: Complete Phase 1 by implementing file parsers and tests with downloaded data.
