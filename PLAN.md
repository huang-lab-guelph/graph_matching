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

### Phase 2: Graph Construction ✅ COMPLETED

**Goal**: Convert parsed data into graph representations for matching

**Tasks**:
1. ✅ Implement `preprocessing/` module:
   - ✅ `MethylNetworkBuilder` - Create structure graph from PDB methyls (407 lines)
   - ✅ `PeakNetworkBuilder` - Create experimental graph from 13C-13C-1H methyl-methyl NOE data (426 lines)
2. ✅ Define node and edge feature representations
3. ✅ Implement distance calculation and edge weight logic
4. ✅ Write unit tests for graph builders (21 tests)
5. ✅ Create interactive marimo notebook demonstrating functionality
6. ✅ Correct NOESY format to 13C-13C-1H methyl-methyl

**Deliverables**:
- ✅ `src/methyl_match/preprocessing/` module with graph builders
- ✅ Graph representations with:
  - Nodes: methyl groups (structure) or HMQC peaks (experimental)
  - Edges: spatial distances (structure) or 13C-13C-1H NOE correlations (experimental)
  - Features: 3D coordinates, residue types (10D), chemical shifts, intensities (4D)
- ✅ `tests/test_preprocessing.py` with comprehensive unit tests (21 tests, all passing)
- ✅ Integration with Phase 1 parsers (55 total tests passing)
- ✅ `notebooks/graph_construction_demo.py` - Interactive marimo notebook
- ✅ Multiple edge weight functions (uniform, inverse, exponential)

**Success Criteria**: ✅ ALL MET
- ✅ Can build structure graph from parsed PDB with configurable distance cutoffs
- ✅ Can build experimental graph from parsed 13C-13C-1H NOESY/HMQC
- ✅ Graphs contain appropriate node/edge features (10D methyl, 4D peak)
- ✅ Edge weights computed correctly based on distances/NOE intensities
- ✅ NetworkX integration working correctly

**Results**:
- **MethylNetworkBuilder**: Creates spatial graphs with distance-based edges
  - Configurable distance cutoffs (6-14Å)
  - Multiple edge weight functions
  - Rich 10D node features (coords + residue type + res number)
  - Distance and adjacency matrices
- **PeakNetworkBuilder**: Creates experimental graphs from 13C-13C-1H NOESY
  - Intensity threshold filtering
  - 13C chemical shift matching for methyl-methyl NOE
  - 4D node features (1H/13C shifts + intensity + degree)
  - NOE correlation matrices
- **Test Coverage**: 21 preprocessing tests + 34 reading tests = 55 total (100% passing)
- **Interactive Demo**: Comprehensive notebook with 14 visualization cells
- **Parameter Analysis**: Sensitivity analysis for cutoffs and thresholds

---

### Phase 3: Matching Algorithms ✅ COMPLETED

**Goal**: Implement multiple graph matching algorithms with unified interface

**Tasks**:
1. ✅ Create `matching/` module with base class:
   - ✅ `GraphMatcher` - Abstract base class defining interface (240 lines)
   - ✅ `MatchingResult` - Result dataclass with confidence scores
2. ✅ Implement classical algorithms:
   - ✅ `GreedyMatcher` - Fast greedy matching (241 lines)
   - ✅ `HungarianMatcher` - Optimal LAP solution (284 lines)
   - ✅ `QAPMatcher` - Quadratic assignment with topology (358 lines)
   - ✅ `SpectralMatcher` - Spectral graph matching with pygmtools (466 lines)
3. ✅ Implement matching utilities:
   - ✅ Chemical shift similarity computation
   - ✅ Topology and distance consistency metrics
   - ✅ Confidence scoring (cost, gap, topology methods)
   - ✅ Assignment validation utilities
4. ✅ Algorithm selection and configuration logic
5. ✅ Write comprehensive unit tests (25 tests, all passing)

**Deliverables**:
- ✅ `src/methyl_match/matching/` module (7 files, ~1,800 lines)
- ✅ 4 working algorithms (Greedy, Hungarian, QAP, Spectral)
- ✅ Unified `GraphMatcher` interface for easy algorithm swapping
- ✅ `tests/test_matching.py` with 25 comprehensive unit tests
- ✅ Validation with synthetic data and real graphs
- ✅ `matching/README.md` with algorithm theory and references (600+ lines)

**Success Criteria**: ✅ ALL MET
- ✅ All algorithms implement common `GraphMatcher` interface
- ✅ Produce assignment mappings with confidence scores (0.0-1.0)
- ✅ Include topology consistency and distance consistency metrics
- ✅ Tests verify correctness on synthetic examples (25/25 passing)
- ✅ All 80 tests passing (34 reading + 21 preprocessing + 25 matching)

**Results**:
- **GreedyMatcher**: Fast heuristic O(n²), good for baselines
- **HungarianMatcher**: Optimal LAP solution O(n³), node-based matching
- **QAPMatcher**: Topology-aware FAQ algorithm, best balance performance/accuracy
- **SpectralMatcher**: Eigendecomposition-based, captures global structure
- **Test Coverage**: 25 matching tests covering all algorithms and utilities
- **Documentation**: Comprehensive README with algorithm theory, complexity, references
- **Dependencies**: Added pygmtools for spectral matching

---

### Phase 4: Output & Utilities ✅ COMPLETED

**Goal**: Export assignment results in multiple formats

**Tasks**:
1. ✅ Implement `writing/` module:
   - ✅ `TextFormatter` - Human-readable summary reports (205 lines)
   - ✅ `CSVFormatter` - Tabular data for analysis (247 lines)
   - ✅ `PyMOLFormatter` - Visualization scripts with confidence coloring (348 lines)
2. ✅ Add confidence scoring logic (integrated in base class)
3. ✅ Add result validation and quality metrics
4. ✅ Write unit tests for formatters (25 tests)

**Deliverables**:
- ✅ `src/methyl_match/writing/` module with all formatters (4 files, ~900 lines)
- ✅ Support for 3 output formats (Text, CSV, PyMOL)
- ✅ Confidence scores and quality metrics in all formats
- ✅ `tests/test_writing.py` with comprehensive unit tests (25 tests, all passing)
- ✅ Example output demonstrations in notebooks
- ✅ `notebooks/writing_demo.py` - Generic writing demo
- ✅ `notebooks/yme1l/04_export_yme1l.py` - YME1L export notebook

**Success Criteria**: ✅ ALL MET
- ✅ Can export results in all specified formats
- ✅ Output includes assignments, confidence scores, validation metrics
- ✅ PyMOL scripts correctly visualize assignments with color coding
- ✅ Formats are compatible with downstream analysis tools
- ✅ All 122 tests passing (97 previous + 25 writing tests)

**Results**:
- **TextFormatter**: Human-readable reports with summary statistics, quality indicators (🟢🟡🔴)
- **CSVFormatter**: Tabular data with peak info, methyl info, coordinates, confidence scores
- **PyMOLFormatter**: Visualization scripts with confidence-based coloring and labels
- **Test Coverage**: 25 writing tests covering all formatters and integration scenarios
- **Notebooks**: 2 comprehensive demos (generic + YME1L-specific)

---

## Current Status

- **Completed Phases**:
  - ✅ Phase 1 - Foundation & File Readers
  - ✅ Phase 2 - Graph Construction
  - ✅ Phase 3 - Matching Algorithms
  - ✅ Phase 4 - Output & Utilities (FINAL PHASE)
- **Project Status**: COMPLETE ✅
- **Branch**: `new-python-project`
- **Last Updated**: 2025-11-22

### Phase 1 Summary

**Completed Components**:
- ✅ **PDBParser** (263 lines): Extracts methyl groups (LEU, VAL, ILE, ALA, THR, MET)
- ✅ **NOESYParser** (466 lines): 13C-13C-1H methyl-methyl NOESY with multi-format support
- ✅ **HMQCParser** (378 lines): Chemical shift parsing and matching
- ✅ **Test Suite** (34 tests): 100% pass rate with real data
- ✅ **Marimo Notebook**: [reading_demo.py](notebooks/reading_demo.py) - Interactive demo with visualizations
- ✅ **Documentation**: README, PLAN.md, CLAUDE.md, notebooks/README.md

**Key Achievements**:
- Parsed 43 methyl groups from ubiquitin structure (1UBQ)
- Correct 13C-13C-1H NOESY format for methyl-methyl assignment
- Auto-detection working for XEASY, Sparky, CSV formats
- Distance matrix calculations (43×43 methyls)
- NOE correlation matrices
- Chemical shift matching with tolerances
- 100% cross-validation agreement between NOESY and HMQC data

### Phase 2 Summary

**Completed Components**:
- ✅ **MethylNetworkBuilder** (407 lines): Spatial graphs from PDB structures
- ✅ **PeakNetworkBuilder** (426 lines): Experimental graphs from 13C-13C-1H NOESY
- ✅ **Test Suite** (21 tests): 100% pass rate with integration tests
- ✅ **Marimo Notebook**: [graph_construction_demo.py](notebooks/graph_construction_demo.py) - Comprehensive graph demo
- ✅ **NetworkX Integration**: Full support for graph operations

**Key Achievements**:
- Distance-based methyl network construction (configurable cutoffs)
- 13C-13C-1H methyl-methyl NOE correlation graphs
- Rich feature vectors (10D methyl nodes, 4D peak nodes)
- Multiple edge weight functions (uniform, inverse, exponential)
- Network statistics and visualization tools
- Parameter sensitivity analysis
- 55 total tests passing (34 reading + 21 preprocessing)

### Phase 4 Summary

**Completed Components**:
- ✅ **TextFormatter** (205 lines): Human-readable reports with quality indicators
- ✅ **CSVFormatter** (247 lines): Tabular data export with coordinates and metadata
- ✅ **PyMOLFormatter** (348 lines): Visualization scripts with confidence-based coloring
- ✅ **Test Suite** (25 tests): 100% pass rate covering all formatters
- ✅ **Marimo Notebooks**: [writing_demo.py](notebooks/writing_demo.py) + [04_export_yme1l.py](notebooks/yme1l/04_export_yme1l.py)

**Key Achievements**:
- Three output formats for comprehensive result export
- Confidence threshold filtering in all formatters
- Quality classifications (high/medium/low) with visual indicators (🟢🟡🔴)
- PyMOL visualization with confidence-based coloring (green/yellow/red/gray)
- CSV export with optional coordinates and unassigned peaks tracking
- Text reports with summary statistics and algorithm metadata
- 122 total tests passing (34 reading + 21 preprocessing + 25 matching + 25 writing + 17 overlap)

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

**Phase 2 Added**:
- NetworkX (>=3.0) - Graph processing and operations

**Future (for Phase 3)**:
- PyTorch Geometric - Advanced graph neural networks (optional)
- PyTorch - ML algorithms (optional)

### File Naming Conventions

- PDB structures: `<protein>.pdb`
- 13C-13C-1H methyl-methyl NOESY data: `<protein>_methyl_noesy.txt` or `<protein>_methyl_noesy.csv`
- HMQC data: `<protein>_hmqc.txt` or `<protein>_hmqc.csv`
- Configuration: `config.yaml` in data directory

**Note**: NOESY files should be in 13C-13C-1H format where:
- w1 = 13C chemical shift of first methyl (ppm)
- w2 = 13C chemical shift of second methyl (ppm)
- w3 = 1H chemical shift (ppm)

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

**Phase 1**: ✅ Parsers work with real data, tests pass (34/34 tests)
**Phase 2**: ✅ Can build graphs from parsed data (21/21 tests, 55/55 total)
**Phase 3**: ✅ Multiple algorithms produce valid assignments (25/25 tests, 80/80 total)
**Phase 4**: ✅ Results exportable in 3+ formats (25/25 tests, 122/122 total)

**PROJECT COMPLETE**: All planned phases implemented successfully ✅

---

## Summary of Progress

### Phases Completed: 4/4 (100%) ✅

**Phase 1 (Foundation & File Readers)**: ✅ COMPLETED
- 3 parsers implemented (1,107 lines)
- 34 tests, 100% passing
- 1 interactive notebook with visualizations
- Correct 13C-13C-1H NOESY format

**Phase 2 (Graph Construction)**: ✅ COMPLETED
- 2 graph builders implemented (833 lines)
- 21 tests, 100% passing (55 total)
- 1 comprehensive interactive notebook with 14 visualizations
- NetworkX integration complete
- Rich feature vectors for future matching

**Phase 3 (Matching Algorithms)**: ✅ COMPLETED
- 4 matching algorithms implemented (~1,800 lines)
- 25 tests, 100% passing (80 total)
- Comprehensive README with algorithm theory and references (600+ lines)
- Unified GraphMatcher interface for easy algorithm swapping
- Confidence scoring and validation metrics

**Phase 4 (Output & Utilities)**: ✅ COMPLETED
- 3 output formatters implemented (~900 lines)
- 25 tests, 100% passing (122 total)
- 2 interactive notebooks (generic + YME1L-specific)
- Text, CSV, and PyMOL export formats
- Confidence-based filtering and quality classifications

---

## Project Complete! 🎉

The methyl assignment library is now fully functional with:
- ✅ Complete end-to-end workflow (reading → preprocessing → matching → writing)
- ✅ Multiple file format support (XEASY, Sparky, CSV)
- ✅ Four graph matching algorithms (Greedy, Hungarian, QAP, Spectral)
- ✅ Three export formats (Text, CSV, PyMOL)
- ✅ 122 comprehensive tests (100% passing)
- ✅ Interactive notebooks for demonstration
- ✅ Real protein validation (YME1L)
