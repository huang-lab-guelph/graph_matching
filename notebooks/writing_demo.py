"""
Writing Module Demo: Exporting Methyl Assignment Results

This notebook demonstrates how to export matching results in multiple formats:
1. Text format (human-readable reports)
2. CSV format (tabular data for analysis)
3. PyMOL format (visualization scripts)
"""

import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Writing Module Demo: Exporting Results

    This notebook demonstrates the **writing module** that exports methyl assignment
    results in three formats:

    1. **TextFormatter** - Human-readable summary reports
    2. **CSVFormatter** - Tabular data for downstream analysis
    3. **PyMOLFormatter** - Visualization scripts with confidence coloring

    We'll use test data from Phase 1-3 to show how to export results.
    """)
    return


@app.cell
def _():
    # Import libraries
    import sys
    from pathlib import Path
    import numpy as np

    # Add project to path
    project_root = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    from methyl_match.reading import PDBParser, HMQCParser, NOESYParser
    from methyl_match.preprocessing import MethylNetworkBuilder, PeakNetworkBuilder
    from methyl_match.matching import QAPMatcher, HungarianMatcher, GreedyMatcher
    from methyl_match.writing import TextFormatter, CSVFormatter, PyMOLFormatter

    return (
        CSVFormatter,
        GreedyMatcher,
        HMQCParser,
        HungarianMatcher,
        MethylNetworkBuilder,
        NOESYParser,
        PDBParser,
        PeakNetworkBuilder,
        PyMOLFormatter,
        QAPMatcher,
        TextFormatter,
        project_root,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load Test Data and Run Matching

    First, let's load some test data and run a matching algorithm.
    """)
    return


@app.cell
def _(
    HMQCParser,
    MethylNetworkBuilder,
    NOESYParser,
    PDBParser,
    PeakNetworkBuilder,
    QAPMatcher,
    project_root,
):
    # Load test data
    data_dir = project_root / "data" / "test"

    # Parse files
    pdb_parser = PDBParser(str(data_dir / "1ubq.pdb"))
    methyls_demo = pdb_parser.extract_methyls()

    hmqc_parser = HMQCParser(str(data_dir / "hmqc.txt"))
    hmqc_peaks_demo = hmqc_parser.parse()

    noesy_parser = NOESYParser(str(data_dir / "noesy.txt"))
    noesy_peaks_demo = noesy_parser.parse()

    # Build graphs
    struct_builder = MethylNetworkBuilder(distance_cutoff=10.0)
    network_struct_demo = struct_builder.build_network(methyls_demo)
    G_struct_demo = network_struct_demo.graph

    peak_builder = PeakNetworkBuilder(intensity_threshold=0.3)
    network_exp_demo = peak_builder.build_network(hmqc_peaks_demo, noesy_peaks_demo)
    G_exp_demo = network_exp_demo.graph

    # Run matching
    matcher_demo = QAPMatcher(method='faq', topology_weight=0.6)
    result_demo = matcher_demo.match(G_exp_demo, G_struct_demo)

    print(f"Loaded data:")
    print(f"  Methyls: {len(methyls_demo)}")
    print(f"  HMQC peaks: {len(hmqc_peaks_demo)}")
    print(f"  NOESY peaks: {len(noesy_peaks_demo)}")
    print(f"\nMatching results:")
    print(f"  Assignments: {result_demo.num_assignments}")
    print(f"  Mean confidence: {result_demo.mean_confidence:.3f}")
    return (
        G_exp_demo,
        G_struct_demo,
        data_dir,
        hmqc_peaks_demo,
        matcher_demo,
        methyls_demo,
        network_exp_demo,
        network_struct_demo,
        noesy_peaks_demo,
        result_demo,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 2. Text Format Export

    The **TextFormatter** creates human-readable reports with:
    - Summary statistics
    - Algorithm metadata
    - Detailed assignment list (sorted by confidence)
    - Quality indicators (🟢🟡🔴)
    """)
    return


@app.cell
def _(TextFormatter, network_exp_demo, network_struct_demo, result_demo):
    # Create text formatter
    text_formatter = TextFormatter(
        include_unassigned=True,
        include_metadata=True,
        confidence_threshold=0.0,
        show_quality_icons=True,
    )

    # Format the result
    text_output = text_formatter.format(
        result_demo,
        network_struct_demo,
        network_exp_demo,
        title="UBIQUITIN METHYL ASSIGNMENT RESULTS",
    )

    print(text_output)
    return (text_formatter, text_output)


@app.cell
def _(mo):
    mo.md("""
    ### Save Text Output to File

    You can save the text report to a file:
    """)
    return


@app.cell
def _(TextFormatter, network_exp_demo, network_struct_demo, project_root, result_demo):
    # Save to file
    output_dir = project_root / "output" / "demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    text_formatter_save = TextFormatter()
    text_formatter_save.format(
        result_demo,
        network_struct_demo,
        network_exp_demo,
        output_path=output_dir / "assignments.txt",
    )

    print(f"✓ Saved text report to: {output_dir / 'assignments.txt'}")
    return (output_dir, text_formatter_save)


@app.cell
def _(mo):
    mo.md("""
    ## 3. CSV Format Export

    The **CSVFormatter** creates tabular data files with:
    - Peak information (ID, chemical shifts, intensity)
    - Methyl information (residue, atom name, coordinates)
    - Confidence scores and quality classifications
    - Optional unassigned peaks CSV
    """)
    return


@app.cell
def _(CSVFormatter, network_exp_demo, network_struct_demo, result_demo):
    # Create CSV formatter
    csv_formatter = CSVFormatter(
        include_unassigned=True,
        include_coordinates=True,
        include_chemical_shifts=True,
    )

    # Format the result
    csv_output = csv_formatter.format(
        result_demo,
        network_struct_demo,
        network_exp_demo,
    )

    # Display first few lines
    print("CSV Output (first 10 lines):")
    print("\n".join(csv_output.split("\n")[:10]))
    return (csv_formatter, csv_output)


@app.cell
def _(mo):
    mo.md("""
    ### Save CSV Output to File

    Save CSV for use in Excel, R, Python pandas, etc.:
    """)
    return


@app.cell
def _(CSVFormatter, network_exp_demo, network_struct_demo, output_dir, result_demo):
    # Save to file (will also create unassigned CSV)
    csv_formatter_save = CSVFormatter(include_unassigned=True)
    csv_formatter_save.format(
        result_demo,
        network_struct_demo,
        network_exp_demo,
        output_path=output_dir / "assignments.csv",
    )

    print(f"✓ Saved CSV to: {output_dir / 'assignments.csv'}")
    print(f"✓ Saved unassigned CSV to: {output_dir / 'assignments_unassigned.csv'}")
    return (csv_formatter_save,)


@app.cell
def _(mo):
    mo.md("""
    ## 4. PyMOL Format Export

    The **PyMOLFormatter** creates visualization scripts with:
    - Automatic PDB loading
    - Confidence-based coloring (green/yellow/red)
    - Text labels with peak IDs and confidence
    - Selections for each confidence group
    """)
    return


@app.cell
def _(PyMOLFormatter, data_dir, network_exp_demo, network_struct_demo, result_demo):
    # Create PyMOL formatter
    pymol_formatter = PyMOLFormatter(
        include_unassigned=True,
        include_metadata=True,
        color_by_confidence=True,
        show_labels=True,
        sphere_scale=0.5,
    )

    # Format the result
    pymol_output = pymol_formatter.format(
        result_demo,
        network_struct_demo,
        network_exp_demo,
        pdb_path=data_dir / "1ubq.pdb",
    )

    # Display first 50 lines
    print("PyMOL Script (first 50 lines):")
    print("\n".join(pymol_output.split("\n")[:50]))
    return (pymol_formatter, pymol_output)


@app.cell
def _(mo):
    mo.md("""
    ### Save PyMOL Script to File

    Save the PyMOL script and load it in PyMOL to visualize assignments:

    ```bash
    pymol output/demo/visualization.pml
    ```
    """)
    return


@app.cell
def _(PyMOLFormatter, data_dir, network_exp_demo, network_struct_demo, output_dir, result_demo):
    # Save to file
    pymol_formatter_save = PyMOLFormatter(show_labels=True)
    pymol_formatter_save.format(
        result_demo,
        network_struct_demo,
        network_exp_demo,
        output_path=output_dir / "visualization.pml",
        pdb_path=data_dir / "1ubq.pdb",
    )

    print(f"✓ Saved PyMOL script to: {output_dir / 'visualization.pml'}")
    print(f"\nTo visualize in PyMOL:")
    print(f"  pymol {output_dir / 'visualization.pml'}")
    return (pymol_formatter_save,)


@app.cell
def _(mo):
    mo.md("""
    ## 5. Confidence Threshold Filtering

    All formatters support confidence thresholds to export only high-quality assignments:
    """)
    return


@app.cell
def _(CSVFormatter, TextFormatter, network_exp_demo, network_struct_demo, result_demo):
    # Export only high-confidence assignments (>= 0.7)
    high_conf_text = TextFormatter(confidence_threshold=0.7)
    high_conf_csv = CSVFormatter(confidence_threshold=0.7)

    text_high = high_conf_text.format(result_demo, network_struct_demo, network_exp_demo)
    csv_high = high_conf_csv.format(result_demo, network_struct_demo, network_exp_demo)

    print("High Confidence Text (summary):")
    print("\n".join([line for line in text_high.split("\n") if "Total Assignments" in line or "Mean Confidence" in line]))
    print(f"\nHigh Confidence CSV rows: {len(csv_high.split(chr(10))) - 1}")  # -1 for header
    return (csv_high, high_conf_csv, high_conf_text, text_high)


@app.cell
def _(mo):
    mo.md("""
    ## 6. Comparing Multiple Algorithms

    You can export results from different algorithms and compare them:
    """)
    return


@app.cell
def _(
    G_exp_demo,
    G_struct_demo,
    GreedyMatcher,
    HungarianMatcher,
    QAPMatcher,
    TextFormatter,
    network_exp_demo,
    network_struct_demo,
    output_dir,
):
    # Run multiple algorithms
    algorithms = {
        'greedy': GreedyMatcher(),
        'hungarian': HungarianMatcher(),
        'qap': QAPMatcher(method='faq'),
    }

    # Export results from each
    for algo_name, matcher in algorithms.items():
        result = matcher.match(G_exp_demo, G_struct_demo)

        # Save text report
        formatter = TextFormatter()
        formatter.format(
            result,
            network_struct_demo,
            network_exp_demo,
            output_path=output_dir / f"assignments_{algo_name}.txt",
            title=f"{algo_name.upper()} ALGORITHM RESULTS",
        )

        print(f"✓ Exported {algo_name}: {result.num_assignments} assignments (conf: {result.mean_confidence:.3f})")
    return (algo_name, algorithms, formatter, matcher, result)


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    ✅ **Successfully demonstrated all three output formats:**

    1. **TextFormatter** - Human-readable reports with statistics and quality indicators
    2. **CSVFormatter** - Tabular data for analysis with optional coordinate export
    3. **PyMOLFormatter** - Visualization scripts with confidence-based coloring

    ### Key Features:
    - Confidence threshold filtering
    - Customizable output options (labels, coordinates, metadata)
    - Unassigned peaks tracking
    - Quality classifications (high/medium/low)
    - Multi-algorithm comparison

    ### Files Generated:
    All output files are saved to `output/demo/`:
    - `assignments.txt` - Text report
    - `assignments.csv` - CSV data
    - `assignments_unassigned.csv` - Unassigned peaks
    - `visualization.pml` - PyMOL script
    - `assignments_greedy.txt` - Greedy algorithm results
    - `assignments_hungarian.txt` - Hungarian algorithm results
    - `assignments_qap.txt` - QAP algorithm results

    **Ready for downstream analysis and structure determination!** 🎉
    """)
    return


if __name__ == "__main__":
    app.run()
