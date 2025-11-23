"""
YME1L Part 4: Exporting Results

This notebook exports the matching results from Part 3 in multiple formats:
- Text reports with confidence scores
- CSV data for downstream analysis
- PyMOL visualization scripts
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
    # YME1L Part 4: Exporting Assignment Results

    This notebook exports the methyl assignment results from Part 3 in three formats:

    1. **Text Format** - Human-readable summary reports
    2. **CSV Format** - Tabular data for analysis
    3. **PyMOL Format** - Visualization scripts with confidence coloring

    We'll focus on the **QAP algorithm results** (recommended) but can export any algorithm.
    """)
    return


@app.cell
def _():
    # Import libraries
    import sys
    from pathlib import Path
    import time

    # Add project to path
    project_root = Path.cwd().parent.parent if Path.cwd().name == "yme1l" else Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
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
        time,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load YME1L Data and Run QAP Matching
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
    time,
):
    # Load YME1L data
    data_dir_export = project_root / "data" / "yme1l"

    print("Loading YME1L data...")

    # Parse files
    pdb_parser_export = PDBParser(str(data_dir_export / "yme1l.pdb"))
    methyls_export = pdb_parser_export.extract_methyls()

    hmqc_parser_export = HMQCParser(str(data_dir_export / "hmqc.list"))
    hmqc_peaks_export = hmqc_parser_export.parse()

    noesy_parser_export = NOESYParser(str(data_dir_export / "noesy.list"))
    noesy_peaks_export = noesy_parser_export.parse()

    # Build graphs
    struct_builder_export = MethylNetworkBuilder(distance_cutoff=10.0)
    network_struct_export = struct_builder_export.build_network(methyls_export)
    G_struct_export = network_struct_export.graph

    peak_builder_export = PeakNetworkBuilder(intensity_threshold=0.3)
    network_exp_export = peak_builder_export.build_network(hmqc_peaks_export, noesy_peaks_export)
    G_exp_export = network_exp_export.graph

    print(f"✓ Loaded {len(methyls_export)} methyls, {len(hmqc_peaks_export)} HMQC peaks, {len(noesy_peaks_export)} NOESY peaks")
    print(f"✓ Built graphs: {G_struct_export.number_of_nodes()} structural nodes, {G_exp_export.number_of_nodes()} experimental nodes")

    # Run QAP matching
    print("\nRunning QAP matching...")
    start_time = time.time()
    matcher_export = QAPMatcher(method='faq', topology_weight=0.6, options={'maxiter': 50})
    result_export = matcher_export.match(G_exp_export, G_struct_export)
    elapsed_time = time.time() - start_time

    print(f"✓ QAP matching complete in {elapsed_time:.2f}s")
    print(f"  Assignments: {result_export.num_assignments}")
    print(f"  Mean confidence: {result_export.mean_confidence:.3f}")
    print(f"  Assignment rate: {result_export.assignment_rate * 100:.1f}%")
    return (
        G_exp_export,
        G_struct_export,
        data_dir_export,
        network_exp_export,
        network_struct_export,
        result_export,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 2. Export Text Report

    Create a comprehensive human-readable report with all assignments.
    """)
    return


@app.cell
def _(
    TextFormatter,
    network_exp_export,
    network_struct_export,
    project_root,
    result_export,
):
    # Create output directory
    output_dir_yme1l = project_root / "output" / "yme1l"
    output_dir_yme1l.mkdir(parents=True, exist_ok=True)

    # Create text formatter
    text_formatter_yme1l = TextFormatter(
        include_unassigned=True,
        include_metadata=True,
        confidence_threshold=0.0,
        show_quality_icons=True,
    )

    # Export full report
    text_formatter_yme1l.format(
        result_export,
        network_struct_export,
        network_exp_export,
        output_path=output_dir_yme1l / "yme1l_assignments_qap.txt",
        title="YME1L METHYL ASSIGNMENT RESULTS (QAP)",
    )

    # Show preview
    text_output_preview = text_formatter_yme1l.format(
        result_export,
        network_struct_export,
        network_exp_export,
        title="YME1L METHYL ASSIGNMENT RESULTS (QAP)",
    )

    # Display first 40 lines
    print("\n".join(text_output_preview.split("\n")[:40]))
    print("...")
    print(f"\n✓ Full text report saved to: {output_dir_yme1l / 'yme1l_assignments_qap.txt'}")
    return (output_dir_yme1l,)


@app.cell
def _(mo):
    mo.md("""
    ### Export High-Confidence Assignments Only

    Create a filtered report with only high-confidence assignments (≥0.7):
    """)
    return


@app.cell
def _(
    TextFormatter,
    network_exp_export,
    network_struct_export,
    output_dir_yme1l,
    result_export,
):
    # High confidence only
    text_formatter_high = TextFormatter(
        confidence_threshold=0.7,
        show_quality_icons=True,
    )

    text_formatter_high.format(
        result_export,
        network_struct_export,
        network_exp_export,
        output_path=output_dir_yme1l / "yme1l_assignments_high_confidence.txt",
        title="YME1L HIGH CONFIDENCE ASSIGNMENTS (QAP, ≥0.7)",
    )

    # Count high confidence
    high_conf_count = sum(1 for c in result_export.confidence_scores.values() if c >= 0.7)
    print(f"✓ Exported {high_conf_count} high-confidence assignments (≥0.7)")
    print(f"  File: {output_dir_yme1l / 'yme1l_assignments_high_confidence.txt'}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Export CSV Data

    Export tabular data for analysis in Excel, R, Python pandas, etc.
    """)
    return


@app.cell
def _(
    CSVFormatter,
    network_exp_export,
    network_struct_export,
    output_dir_yme1l,
    result_export,
):
    # Create CSV formatter with all data
    csv_formatter_yme1l = CSVFormatter(
        include_unassigned=True,
        include_coordinates=True,
        include_chemical_shifts=True,
    )

    # Export to CSV
    csv_formatter_yme1l.format(
        result_export,
        network_struct_export,
        network_exp_export,
        output_path=output_dir_yme1l / "yme1l_assignments_qap.csv",
    )

    # Show CSV preview
    csv_output = csv_formatter_yme1l.format(
        result_export,
        network_struct_export,
        network_exp_export,
    )

    print("CSV Preview (first 10 lines):")
    print("\n".join(csv_output.split("\n")[:10]))
    print("...")
    print(f"\n✓ CSV data saved to: {output_dir_yme1l / 'yme1l_assignments_qap.csv'}")
    print(f"✓ Unassigned peaks saved to: {output_dir_yme1l / 'yme1l_assignments_qap_unassigned.csv'}")
    return


@app.cell
def _(mo):
    mo.md("""
    ### Export High-Confidence CSV

    Create a CSV with only high-confidence assignments for reliable downstream use:
    """)
    return


@app.cell
def _(
    CSVFormatter,
    network_exp_export,
    network_struct_export,
    output_dir_yme1l,
    result_export,
):
    # High confidence CSV
    csv_formatter_high = CSVFormatter(
        confidence_threshold=0.7,
        include_coordinates=True,
        include_chemical_shifts=True,
    )

    csv_formatter_high.format(
        result_export,
        network_struct_export,
        network_exp_export,
        output_path=output_dir_yme1l / "yme1l_assignments_high_confidence.csv",
    )

    print(f"✓ High-confidence CSV saved to: {output_dir_yme1l / 'yme1l_assignments_high_confidence.csv'}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Export PyMOL Visualization

    Create a PyMOL script to visualize assignments on the 3D structure with confidence-based coloring:
    - 🟢 Green: High confidence (≥0.7)
    - 🟡 Yellow: Medium confidence (0.5-0.7)
    - 🔴 Red: Low confidence (<0.5)
    - Gray: Unassigned methyls
    """)
    return


@app.cell
def _(
    PyMOLFormatter,
    data_dir_export,
    network_exp_export,
    network_struct_export,
    output_dir_yme1l,
    result_export,
):
    # Create PyMOL formatter
    pymol_formatter_yme1l = PyMOLFormatter(
        include_unassigned=True,
        include_metadata=True,
        color_by_confidence=True,
        show_labels=True,
        sphere_scale=0.6,
    )

    # Export PyMOL script
    pymol_formatter_yme1l.format(
        result_export,
        network_struct_export,
        network_exp_export,
        output_path=output_dir_yme1l / "yme1l_visualization.pml",
        pdb_path=data_dir_export / "yme1l.pdb",
    )

    # Show preview
    pymol_output = pymol_formatter_yme1l.format(
        result_export,
        network_struct_export,
        network_exp_export,
        pdb_path=data_dir_export / "yme1l.pdb",
    )

    print("PyMOL Script Preview (first 45 lines):")
    print("\n".join(pymol_output.split("\n")[:45]))
    print("...")
    print(f"\n✓ PyMOL script saved to: {output_dir_yme1l / 'yme1l_visualization.pml'}")
    print(f"\nTo visualize in PyMOL:")
    print(f"  cd {output_dir_yme1l}")
    print(f"  pymol yme1l_visualization.pml")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Export Results from All Algorithms

    For comparison, export results from all three matching algorithms:
    """)
    return


@app.cell
def _(
    CSVFormatter,
    G_exp_export,
    G_struct_export,
    GreedyMatcher,
    HungarianMatcher,
    QAPMatcher,
    TextFormatter,
    network_exp_export,
    network_struct_export,
    output_dir_yme1l,
    time,
):
    # Run all algorithms (skip Spectral due to size)
    algorithms_export = {
        'greedy': GreedyMatcher(use_topology=True),
        'hungarian': HungarianMatcher(),
        'qap': QAPMatcher(method='faq', topology_weight=0.6, options={'maxiter': 50}),
    }

    print("Exporting results from all algorithms...\n")

    for algo_name_export, matcher_algo in algorithms_export.items():
        print(f"Running {algo_name_export}...")
        start = time.time()
        result_algo = matcher_algo.match(G_exp_export, G_struct_export)
        elapsed = time.time() - start

        # Export text report
        text_fmt = TextFormatter()
        text_fmt.format(
            result_algo,
            network_struct_export,
            network_exp_export,
            output_path=output_dir_yme1l / f"yme1l_assignments_{algo_name_export}.txt",
            title=f"YME1L METHYL ASSIGNMENT RESULTS ({algo_name_export.upper()})",
        )

        # Export CSV
        csv_fmt = CSVFormatter(include_coordinates=True)
        csv_fmt.format(
            result_algo,
            network_struct_export,
            network_exp_export,
            output_path=output_dir_yme1l / f"yme1l_assignments_{algo_name_export}.csv",
        )

        print(f"  ✓ {algo_name_export}: {result_algo.num_assignments} assignments, confidence: {result_algo.mean_confidence:.3f}, time: {elapsed:.2f}s")
        print(f"    Files: yme1l_assignments_{algo_name_export}.txt/csv")

    print("\n✓ All algorithms exported successfully!")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Summary Statistics

    Let's summarize the QAP results by confidence level:
    """)
    return


@app.cell
def _(result_export):
    # Count assignments by confidence level
    high_conf = sum(1 for c in result_export.confidence_scores.values() if c >= 0.7)
    medium_conf = sum(1 for c in result_export.confidence_scores.values() if 0.5 <= c < 0.7)
    low_conf = sum(1 for c in result_export.confidence_scores.values() if c < 0.5)

    print("YME1L QAP Assignment Summary:")
    print("=" * 60)
    print(f"Total Assignments:           {result_export.num_assignments}")
    print(f"Unassigned Peaks:            {result_export.num_unassigned}")
    print(f"Assignment Rate:             {result_export.assignment_rate * 100:.1f}%")
    print(f"Mean Confidence:             {result_export.mean_confidence:.3f}")
    print("")
    print("Confidence Breakdown:")
    print(f"  🟢 High (≥0.7):            {high_conf} ({100 * high_conf / result_export.num_assignments:.1f}%)")
    print(f"  🟡 Medium (0.5-0.7):       {medium_conf} ({100 * medium_conf / result_export.num_assignments:.1f}%)")
    print(f"  🔴 Low (<0.5):             {low_conf} ({100 * low_conf / result_export.num_assignments:.1f}%)")
    print("=" * 60)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    ✅ **Successfully exported YME1L methyl assignment results in three formats:**

    ### Files Generated:

    **QAP Algorithm (Recommended):**
    - `yme1l_assignments_qap.txt` - Full text report with all assignments
    - `yme1l_assignments_high_confidence.txt` - High-confidence assignments only (≥0.7)
    - `yme1l_assignments_qap.csv` - Full CSV data with coordinates
    - `yme1l_assignments_qap_unassigned.csv` - Unassigned peaks list
    - `yme1l_assignments_high_confidence.csv` - High-confidence CSV only
    - `yme1l_visualization.pml` - PyMOL visualization script

    **All Algorithms (for comparison):**
    - `yme1l_assignments_greedy.txt/csv` - Greedy algorithm results
    - `yme1l_assignments_hungarian.txt/csv` - Hungarian algorithm results

    ### Usage:

    **1. View text reports:**
    ```bash
    cat output/yme1l/yme1l_assignments_qap.txt
    ```

    **2. Analyze in Python/R:**
    ```python
    import pandas as pd
    df = pd.read_csv("output/yme1l/yme1l_assignments_high_confidence.csv")
    ```

    **3. Visualize in PyMOL:**
    ```bash
    cd output/yme1l
    pymol yme1l_visualization.pml
    ```

    ### Recommendations:

    - **Use high-confidence assignments (≥0.7)** for downstream analysis
    - **Validate medium-confidence assignments** with additional experiments
    - **Review low-confidence assignments** carefully before using
    - **Compare multiple algorithms** to identify robust assignments

    **Ready for structure determination and functional studies!** 🎉
    """)
    return


if __name__ == "__main__":
    app.run()
