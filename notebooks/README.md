# Methyl Match Notebooks

Interactive notebooks demonstrating the capabilities of the methyl_match library.

## Available Notebooks

### [reading_demo.py](reading_demo.py)

**Reading Module Demonstration** (Phase 1)

An interactive marimo notebook showcasing the file parsing capabilities of methyl_match.

**Contents:**
1. **PDB Structure Parsing** - Extract and visualize methyl groups from protein structures
2. **NOESY Peak Parsing** - Parse 13C-13C-1H methyl-methyl NOE connectivity data with multi-format support
3. **HMQC Peak Parsing** - Parse 1H-13C correlation spectra
4. **Integration** - Cross-validate assignments across data sources

**Features demonstrated:**
- Multi-format auto-detection (XEASY, Sparky, CSV, NMRPipe)
- 3D visualization of methyl distributions
- Distance matrix calculations
- NOE correlation matrices
- Chemical shift scatter plots
- Cross-validation of assignments

### [graph_construction_demo.py](graph_construction_demo.py)

**Graph Construction Demonstration** (Phase 2)

An interactive marimo notebook showcasing the graph construction capabilities from structural and experimental data.

**Contents:**
1. **Methyl Network Building** - Create spatial graphs from PDB structures
2. **Peak Network Building** - Create experimental graphs from 13C-13C-1H NOESY and HMQC data
3. **Network Comparison** - Compare structural and experimental networks
4. **Feature Extraction** - Explore node and edge features
5. **Parameter Effects** - Analyze how parameters affect graph structure

**Features demonstrated:**
- Distance-based graph construction with configurable cutoffs
- 13C-13C-1H methyl-methyl NOE correlation graphs
- Multiple edge weight functions (uniform, inverse, exponential)
- Rich node/edge feature vectors for matching algorithms
- Network statistics and visualization
- Parameter sensitivity analysis
- Side-by-side network comparison

## Running the Notebooks

### Prerequisites

Install the notebook dependencies:

```bash
# Install with uv (recommended)
uv sync --extra notebooks

# Or with pip
pip install -e ".[notebooks]"
```

### Launch marimo

```bash
# Navigate to notebooks directory
cd notebooks

# Run the reading demo (Phase 1)
uv run marimo edit reading_demo.py

# Run the graph construction demo (Phase 2)
uv run marimo edit graph_construction_demo.py

# Or run all notebooks
uv run marimo edit .
```

This will open an interactive notebook in your browser where you can:
- Execute cells
- Modify code
- Visualize results
- Export to HTML or PDF

## About marimo

[marimo](https://marimo.io) is a reactive Python notebook that's:
- **Reproducible**: Automatically manages dependencies between cells
- **Interactive**: UI elements are automatically synchronized with Python code
- **Version control friendly**: Notebooks are pure Python files
- **Fast**: Only re-runs affected cells when you make changes

### marimo Key Features

- Reactive execution (changes propagate automatically)
- Interactive widgets
- Rich visualizations (matplotlib, plotly, etc.)
- Export to HTML, PDF, or run as web apps
- Git-friendly (pure Python, no JSON)

## Notebook Structure

Each marimo notebook is organized into cells:

```python
@app.cell  # Markdown cells for documentation
def __(mo):
    mo.md(r"""# Title""")
    return

@app.cell  # Code cells for computation
def __():
    import pandas as pd
    return pd,

@app.cell  # Cells can depend on each other
def __(pd):
    df = pd.DataFrame({'x': [1, 2, 3]})
    return df,
```

## Tips

1. **Start with reading_demo.py** to understand the basics of data parsing
2. **Continue with graph_construction_demo.py** to see how graphs are built
3. **Modify parameters** - Try different files, filters, cutoffs, or thresholds
4. **Add visualizations** - marimo makes it easy to add interactive plots
5. **Export results** - Save figures or export the notebook as HTML

## Data Files

The notebooks use example data from `data/test/`:
- **1ubq.pdb** - Ubiquitin crystal structure (43 methyl groups)
- **2gb1.pdb** - GB1 protein structure
- **example_methyl_noesy.txt** - 13C-13C-1H methyl-methyl NOESY peak list (XEASY format)
- **example_methyl_noesy_sparky.txt** - 13C-13C-1H methyl-methyl NOESY peak list (Sparky format)
- **example_methyl_noesy.csv** - 13C-13C-1H methyl-methyl NOESY peak list (CSV format)
- **example_hmqc.txt** - HMQC peak list (XEASY format)
- **example_hmqc.csv** - HMQC peak list (CSV format)

## Next Steps

As the methyl_match library grows, additional notebooks will be added:

- ✅ **Phase 1**: File reading and parsing ([reading_demo.py](reading_demo.py))
- ✅ **Phase 2**: Graph construction and visualization ([graph_construction_demo.py](graph_construction_demo.py))
- 🔄 **Phase 3**: Graph matching algorithms comparison
- 🔄 **Phase 4**: Output formatting and result visualization
- 🔄 **Phase 5**: Complete workflow examples

## Contributing

To add a new notebook:

1. Create a new `.py` file in this directory
2. Use marimo to create the notebook structure
3. Document the notebook's purpose in this README
4. Test the notebook with example data

## Resources

- [marimo Documentation](https://docs.marimo.io/)
- [methyl_match Documentation](../README.md)
- [Project Development Plan](../PLAN.md)

## Troubleshooting

**marimo not found:**
```bash
uv sync --extra notebooks
```

**Import errors:**
```bash
# Make sure methyl_match is installed
uv sync
```

**Data files not found:**
```bash
# Check you're in the notebooks/ directory
cd notebooks
uv run marimo edit reading_demo.py
```

**Browser doesn't open:**
- Manually open the URL shown in terminal
- Check firewall settings
- Try a different port: `marimo edit reading_demo.py --port 8080`
