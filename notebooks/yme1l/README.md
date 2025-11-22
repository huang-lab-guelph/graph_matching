# YME1L Methyl Assignment Notebooks

This directory contains a progressive series of interactive notebooks demonstrating the complete methyl assignment workflow for the YME1L protein.

## Overview

YME1L is a mitochondrial ATPase protein. These notebooks demonstrate automated methyl resonance assignment using graph matching algorithms.

## Notebooks

### 1. Reading Data (`01_reading_yme1l.py`)

**Purpose**: Load and validate all input files

**Contents**:
- Parse PDB structure file
- Parse HMQC peak list (1H-13C chemical shifts)
- Parse NOESY peak list (13C-13C-1H methyl-methyl NOEs)
- Display summary statistics
- Visualize chemical shift distributions

**Run**: `marimo edit 01_reading_yme1l.py`

**Output**: Validation that all files are correctly formatted and parsed

---

### 2. Graph Construction (`02_graphs_yme1l.py`)

**Purpose**: Build structural and experimental graph representations

**Contents**:
- Build structural graph from PDB (methyls connected by distances)
- Build experimental graph from HMQC+NOESY (peaks connected by NOEs)
- Interactive parameter tuning (distance cutoff, intensity threshold)
- Graph statistics and connectivity analysis
- Degree distribution visualization
- Graph comparison

**Run**: `marimo edit 02_graphs_yme1l.py`

**Key Parameters**:
- Distance cutoff: 6-14Å (default: 10Å)
- Intensity threshold: 0.0-1.0 (default: 0.3)

**Output**: Two NetworkX graphs ready for matching

---

### 3. Matching Algorithms (`03_matching_yme1l.py`)

**Purpose**: Perform automated assignment using all four algorithms

**Contents**:
- Run all four matching algorithms:
  - **GreedyMatcher** - Fast heuristic
  - **HungarianMatcher** - Optimal LAP
  - **QAPMatcher** - Topology-aware (recommended) ⭐
  - **SpectralMatcher** - Spectral matching
- Compare performance metrics
- Display top assignments with confidence scores
- Visualize results across all algorithms
- Algorithm recommendations

**Run**: `marimo edit 03_matching_yme1l.py`

**Output**:
- Assignment predictions for all peaks
- Confidence scores for each assignment
- Performance comparison across algorithms
- Validation metrics

---

## Quick Start

```bash
# Navigate to notebooks directory
cd notebooks/yme1l

# Run notebooks in order
marimo edit 01_reading_yme1l.py
marimo edit 02_graphs_yme1l.py
marimo edit 03_matching_yme1l.py
```

## Data Files

The notebooks use data from `../../data/yme1l/`:

- `yme1l.pdb` - Protein structure (PDB format)
- `hmqc.list` - HMQC peak list (1H-13C chemical shifts)
- `noesy.list` - NOESY peak list (13C-13C-1H methyl-methyl correlations)

## Expected Results

**From Part 1 (Reading)**:
- Successfully parse all files
- Extract methyl groups from structure
- Identify HMQC and NOESY peaks

**From Part 2 (Graphs)**:
- Structural graph: ~100-200 nodes, connected by spatial distances
- Experimental graph: ~100-200 nodes, connected by NOE correlations
- Similar node counts in both graphs (good for matching)

**From Part 3 (Matching)**:
- QAP recommended algorithm: best balance of accuracy/speed
- High confidence assignments (≥0.7): reliable for downstream analysis
- Medium confidence (0.5-0.7): may need validation
- Low confidence (<0.5): ambiguous, may need additional data

## Interpretation Guide

### Confidence Scores

- **🟢 High (≥0.7)**: Reliable assignments, use for structure determination
- **🟡 Medium (0.5-0.7)**: Plausible assignments, validate if critical
- **🔴 Low (<0.5)**: Ambiguous assignments, needs more data or manual check

### Validation Metrics

- **Topology Consistency**: How well NOE connectivity matches spatial distances
  - High (>0.7): Excellent agreement
  - Medium (0.5-0.7): Good agreement
  - Low (<0.5): Poor agreement, check parameters

- **Distance Consistency**: Correlation between NOE intensity and distance
  - High (>0.7): Strong NOE-distance correlation
  - Medium (0.5-0.7): Moderate correlation
  - Low (<0.5): Weak correlation, check data quality

## Parameter Tuning

### Distance Cutoff (Graph Construction)

- **6-8Å**: Very strict, sparse graph, few edges
- **10Å**: Recommended default, balanced connectivity
- **12-14Å**: Permissive, dense graph, many edges

**Effect**: Higher cutoff = more edges = more topology information but slower matching

### Intensity Threshold (Graph Construction)

- **0.2**: Permissive, includes weak NOEs
- **0.3**: Recommended default
- **0.5**: Strict, only strong NOEs

**Effect**: Lower threshold = more edges but more noise; higher = fewer edges but cleaner

### Topology Weight (QAP Matching)

- **0.4-0.5**: Balanced between nodes and edges
- **0.6-0.7**: Emphasize topology (recommended for good NOE data)
- **0.8-0.9**: Strong emphasis on edges

**Effect**: Higher weight = more emphasis on graph structure vs node features

## Troubleshooting

### Issue: Low assignment rate (<50%)

**Solutions**:
- Decrease distance cutoff (try 8Å)
- Decrease intensity threshold (try 0.2)
- Check data quality in Part 1

### Issue: Low confidence scores (<0.5 average)

**Solutions**:
- Increase distance cutoff (try 12Å)
- Adjust intensity threshold
- Try different algorithm (Hungarian for node-based)

### Issue: Poor topology consistency (<0.5)

**Solutions**:
- Adjust distance cutoff to match NOE range
- Filter NOESY peaks by intensity
- Check if NOE data is 13C-13C-1H methyl-methyl format

## Next Steps

After completing these notebooks:

1. **Export Results** (Phase 4) - Generate output files:
   - Text reports
   - CSV tables
   - PyMOL visualization scripts

2. **Validate** - Compare with manual assignments if available

3. **Iterate** - Refine parameters based on validation

4. **Use Results** - Incorporate assignments into structure determination

## References

**Algorithms**:
- Hungarian: Kuhn (1955), optimal LAP
- QAP: Vogelstein et al. (2015), PLOS ONE
- Spectral: Umeyama (1988), IEEE TPAMI

**NMR Assignment Tools**:
- MAGMA: Pritchard & Muhandiram (2017), JACS
- MethylFLYA: Zimmermann et al. (2019), Nature Comm

## Support

For questions or issues:
- Check `../../src/methyl_match/matching/README.md` for algorithm details
- Review `../../PLAN.md` for project overview
- See main `../../README.md` for installation instructions

---

**Last Updated**: 2025-11-22
**Python Version**: 3.12+
**Dependencies**: methyl_match, marimo, numpy, pandas, matplotlib, seaborn, networkx
