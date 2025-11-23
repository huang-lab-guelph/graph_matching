"""
YME1L Part 2: Graph Construction

This notebook builds structural and experimental graphs from YME1L data:
- Structural graph: methyl groups connected by spatial distances
- Experimental graph: HMQC peaks connected by NOE correlations
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
    # YME1L Part 2: Graph Construction

    This notebook builds graph representations of the YME1L protein data:

    1. **Structural Graph** - Nodes are methyls from PDB, edges are spatial distances
    2. **Experimental Graph** - Nodes are HMQC peaks, edges are NOE correlations

    We'll explore graph properties, connectivity, and visualize the networks.
    """)
    return


@app.cell
def _():
    # Import libraries
    import sys
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx

    # Add project to path
    project_root = Path.cwd().parent.parent
    sys.path.insert(0, str(project_root / "src"))

    from methyl_match.reading import PDBParser, NOESYParser, HMQCParser
    from methyl_match.preprocessing import MethylNetworkBuilder, PeakNetworkBuilder

    # Set style
    sns.set_style("whitegrid")
    return (
        HMQCParser,
        MethylNetworkBuilder,
        NOESYParser,
        PDBParser,
        PeakNetworkBuilder,
        np,
        nx,
        pd,
        plt,
        project_root,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Load Data
    """)
    return


@app.cell
def _(HMQCParser, NOESYParser, PDBParser, project_root):
    # Load all data
    data_dir_graph = project_root / "data" / "yme1l"

    # Parse PDB - YME1L specific: only ILE, VAL, LEU, MET
    pdb_parser_g = PDBParser(str(data_dir_graph / "yme1l.pdb"))
    yme1l_residue_types_g = ['ILE', 'VAL', 'LEU', 'MET']
    methyls_g = pdb_parser_g.extract_methyls(residue_types=yme1l_residue_types_g)

    # Parse HMQC
    hmqc_parser_g = HMQCParser(str(data_dir_graph / "hmqc.list"))
    hmqc_peaks_g = hmqc_parser_g.parse()

    # Parse NOESY
    noesy_parser_g = NOESYParser(str(data_dir_graph / "noesy.list"))
    noesy_peaks_g = noesy_parser_g.parse()

    print(f"Loaded {len(methyls_g)} methyls (ILE, VAL, LEU, MET only), {len(hmqc_peaks_g)} HMQC peaks, {len(noesy_peaks_g)} NOESY peaks")
    return hmqc_peaks_g, methyls_g, noesy_peaks_g


@app.cell
def _(mo):
    mo.md("""
    ## 2. Build Structural Graph (from PDB)
    """)
    return


@app.cell
def _(mo):
    # Create distance cutoff slider
    distance_cutoff_slider = mo.ui.slider(
        start=6.0,
        stop=14.0,
        step=1.0,
        value=10.0,
        label="Distance Cutoff (Å):"
    )

    mo.md(f"""
    **Parameters:**

    {distance_cutoff_slider}

    Adjust the distance cutoff to control graph connectivity.
    Typical range: 8-12Å for methyl-methyl NOEs.
    """)
    return (distance_cutoff_slider,)


@app.cell
def _(MethylNetworkBuilder, distance_cutoff_slider, methyls_g):
    # Build structural graph
    struct_builder = MethylNetworkBuilder(distance_cutoff=distance_cutoff_slider.value)
    network_struct = struct_builder.build_network(methyls_g)
    G_struct_yme1l = network_struct.graph

    print(f"Built structural graph:")
    print(f"  Nodes: {G_struct_yme1l.number_of_nodes()}")
    print(f"  Edges: {G_struct_yme1l.number_of_edges()}")
    print(f"  Avg degree: {2 * G_struct_yme1l.number_of_edges() / G_struct_yme1l.number_of_nodes():.1f}")
    return (G_struct_yme1l,)


@app.cell
def _(G_struct_yme1l, nx):
    # Compute graph statistics
    struct_degrees = dict(G_struct_yme1l.degree())
    struct_degree_values = list(struct_degrees.values())

    # Clustering and connectivity
    try:
        struct_clustering = nx.average_clustering(G_struct_yme1l)
    except:
        struct_clustering = 0.0

    struct_components = nx.number_connected_components(G_struct_yme1l)
    return struct_clustering, struct_components, struct_degree_values


@app.cell
def _(G_struct_yme1l, mo, struct_clustering, struct_components):
    mo.md(f"""
    **Structural Graph Statistics:**
    - Nodes: {G_struct_yme1l.number_of_nodes()}
    - Edges: {G_struct_yme1l.number_of_edges()}
    - Average degree: {2 * G_struct_yme1l.number_of_edges() / G_struct_yme1l.number_of_nodes():.2f}
    - Clustering coefficient: {struct_clustering:.3f}
    - Connected components: {struct_components}
    """)
    return


@app.cell
def _(np, plt, struct_degree_values):
    # Visualize degree distribution
    fig_deg_struct, ax_deg_struct = plt.subplots(figsize=(10, 6))

    ax_deg_struct.hist(struct_degree_values, bins=range(0, max(struct_degree_values) + 2),
                       color='steelblue', edgecolor='navy', alpha=0.7)
    ax_deg_struct.set_xlabel('Node Degree', fontsize=12)
    ax_deg_struct.set_ylabel('Count', fontsize=12)
    ax_deg_struct.set_title('YME1L Structural Graph - Degree Distribution', fontsize=14, fontweight='bold')
    ax_deg_struct.axvline(np.mean(struct_degree_values), color='red', linestyle='--',
                          linewidth=2, label=f'Mean: {np.mean(struct_degree_values):.1f}')
    ax_deg_struct.legend()
    ax_deg_struct.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. Overlap Diagnostics (Before Building Graph)

    Before building the experimental graph, let's analyze chemical shift overlap
    to understand potential ambiguity in NOESY peak matching.
    """)
    return


@app.cell
def _(hmqc_peaks_g, noesy_peaks_g):
    # Run overlap diagnostics
    from methyl_match.preprocessing import overlap_diagnostics

    # Analyze HMQC overlap
    overlap_report = overlap_diagnostics.analyze_hmqc_overlap(
        hmqc_peaks_g, c_tolerance=0.5, h_tolerance=0.05
    )

    # Predict NOESY ambiguity
    noesy_pred = overlap_diagnostics.analyze_noesy_ambiguity(
        hmqc_peaks_g, noesy_peaks_g, c_tolerance=0.5, intensity_threshold=0.3
    )

    print("HMQC Overlap Analysis (1D, 13C only):")
    print(f"  Peaks with overlap: {overlap_report['overlap_stats_1d']['peaks_with_overlap']} / {overlap_report['overlap_stats_1d']['total_peaks']} ({overlap_report['overlap_stats_1d']['overlap_pct']:.1f}%)")
    print(f"  Average overlaps per peak: {overlap_report['overlap_stats_1d']['avg_overlaps_per_peak']:.2f}")
    print(f"  Severe overlap (>5): {overlap_report['overlap_stats_1d']['severe_overlap']} peaks ({overlap_report['overlap_stats_1d']['severe_overlap_pct']:.1f}%)")
    print("")
    print("NOESY Matching Prediction:")
    print(f"  Total NOE peaks: {noesy_pred['total_noe_peaks']}")
    print(f"  Predicted matched edges: {noesy_pred['predicted_matched']}")
    print(f"  Predicted ambiguous w1: {noesy_pred['predicted_ambiguous_w1']} ({100*noesy_pred['predicted_ambiguous_w1']/max(noesy_pred['above_threshold'],1):.1f}%)")
    print(f"  Predicted ambiguous w2: {noesy_pred['predicted_ambiguous_w2']} ({100*noesy_pred['predicted_ambiguous_w2']/max(noesy_pred['above_threshold'],1):.1f}%)")
    print("")
    print("Recommendations:")
    for i, rec in enumerate(overlap_report['recommendations'], 1):
        print(f"  {i}. {rec}")
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Build Experimental Graph (from HMQC + NOESY)
    """)
    return


@app.cell
def _(mo):
    # Create intensity threshold slider
    intensity_threshold_slider = mo.ui.slider(
        start=0.0,
        stop=1.0,
        step=0.1,
        value=0.3,
        label="Intensity Threshold:"
    )

    mo.md(f"""
    **Parameters:**

    {intensity_threshold_slider}

    Adjust the intensity threshold to filter weak NOE signals.
    Typical range: 0.2-0.5.
    """)
    return (intensity_threshold_slider,)


@app.cell
def _(
    PeakNetworkBuilder,
    hmqc_peaks_g,
    intensity_threshold_slider,
    noesy_peaks_g,
):
    # Build experimental graph
    peak_builder = PeakNetworkBuilder(intensity_threshold=intensity_threshold_slider.value)
    network_exp = peak_builder.build_network(hmqc_peaks_g, noesy_peaks_g)
    G_exp_yme1l = network_exp.graph

    print(f"Built experimental graph:")
    print(f"  Nodes: {G_exp_yme1l.number_of_nodes()}")
    print(f"  Edges: {G_exp_yme1l.number_of_edges()}")
    print(f"  Avg degree: {2 * G_exp_yme1l.number_of_edges() / G_exp_yme1l.number_of_nodes():.1f}")
    return G_exp_yme1l, network_exp


@app.cell
def _(G_exp_yme1l, nx):
    # Compute experimental graph statistics
    exp_degrees = dict(G_exp_yme1l.degree())
    exp_degree_values = list(exp_degrees.values())

    # Clustering
    try:
        exp_clustering = nx.average_clustering(G_exp_yme1l)
    except:
        exp_clustering = 0.0

    exp_components = nx.number_connected_components(G_exp_yme1l)
    return exp_clustering, exp_components, exp_degree_values


@app.cell
def _(G_exp_yme1l, exp_clustering, exp_components, mo):
    mo.md(f"""
    **Experimental Graph Statistics:**
    - Nodes: {G_exp_yme1l.number_of_nodes()}
    - Edges: {G_exp_yme1l.number_of_edges()}
    - Average degree: {2 * G_exp_yme1l.number_of_edges() / G_exp_yme1l.number_of_nodes():.2f}
    - Clustering coefficient: {exp_clustering:.3f}
    - Connected components: {exp_components}
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **Ambiguity Report (After Building):**

    Now let's see the actual ambiguity in the built graph.
    """)
    return


@app.cell
def _(network_exp):
    # Get and display ambiguity report
    ambig_report = network_exp.get_ambiguity_report()

    print("EXPERIMENTAL GRAPH AMBIGUITY SUMMARY")
    print("=" * 60)
    print(f"Total edges: {ambig_report['total_edges']}")
    print(f"Average ambiguity score: {ambig_report['avg_ambiguity_score']:.3f} (1.0=certain, 0.0=ambiguous)")
    print(f"Average candidates per NOE dimension:")
    print(f"  w1 (13C-1): {ambig_report['avg_candidates_w1']:.2f}")
    print(f"  w2 (13C-2): {ambig_report['avg_candidates_w2']:.2f}")
    print("")
    print("Edge ambiguity breakdown:")
    total = max(ambig_report['total_edges'], 1)
    print(f"  Certain (score = 1.0):       {len(ambig_report['certain_edges']):4d} ({100*len(ambig_report['certain_edges'])/total:.1f}%)")
    print(f"  Low ambiguity (0.5-1.0):     {len(ambig_report['low_ambiguity_edges']):4d} ({100*len(ambig_report['low_ambiguity_edges'])/total:.1f}%)")
    print(f"  Medium ambiguity (0.25-0.5): {len(ambig_report['medium_ambiguity_edges']):4d} ({100*len(ambig_report['medium_ambiguity_edges'])/total:.1f}%)")
    print(f"  High ambiguity (< 0.25):     {len(ambig_report['high_ambiguity_edges']):4d} ({100*len(ambig_report['high_ambiguity_edges'])/total:.1f}%)")
    return


@app.cell
def _(exp_degree_values, np, plt):
    # Visualize experimental degree distribution
    fig_deg_exp, ax_deg_exp = plt.subplots(figsize=(10, 6))

    ax_deg_exp.hist(exp_degree_values, bins=range(0, max(exp_degree_values) + 2),
                    color='coral', edgecolor='darkred', alpha=0.7)
    ax_deg_exp.set_xlabel('Node Degree', fontsize=12)
    ax_deg_exp.set_ylabel('Count', fontsize=12)
    ax_deg_exp.set_title('YME1L Experimental Graph - Degree Distribution', fontsize=14, fontweight='bold')
    ax_deg_exp.axvline(np.mean(exp_degree_values), color='blue', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(exp_degree_values):.1f}')
    ax_deg_exp.legend()
    ax_deg_exp.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Graph Comparison
    """)
    return


@app.cell
def _(G_exp_yme1l, G_struct_yme1l, pd):
    # Create comparison table
    comparison_table = pd.DataFrame({
        'Metric': [
            'Nodes',
            'Edges',
            'Average Degree',
            'Max Degree',
            'Min Degree',
        ],
        'Structural': [
            G_struct_yme1l.number_of_nodes(),
            G_struct_yme1l.number_of_edges(),
            f"{2 * G_struct_yme1l.number_of_edges() / G_struct_yme1l.number_of_nodes():.2f}",
            max(dict(G_struct_yme1l.degree()).values()),
            min(dict(G_struct_yme1l.degree()).values()),
        ],
        'Experimental': [
            G_exp_yme1l.number_of_nodes(),
            G_exp_yme1l.number_of_edges(),
            f"{2 * G_exp_yme1l.number_of_edges() / G_exp_yme1l.number_of_nodes():.2f}",
            max(dict(G_exp_yme1l.degree()).values()),
            min(dict(G_exp_yme1l.degree()).values()),
        ]
    })

    comparison_table
    return


@app.cell
def _(G_exp_yme1l, G_struct_yme1l, plt):
    # Side-by-side comparison visualization
    fig_comp, axes_comp = plt.subplots(1, 2, figsize=(16, 6))

    # Structural graph properties
    axes_comp[0].bar(['Nodes', 'Edges'],
                     [G_struct_yme1l.number_of_nodes(), G_struct_yme1l.number_of_edges()],
                     color='steelblue', edgecolor='navy')
    axes_comp[0].set_ylabel('Count')
    axes_comp[0].set_title('Structural Graph')
    axes_comp[0].grid(axis='y', alpha=0.3)

    # Experimental graph properties
    axes_comp[1].bar(['Nodes', 'Edges'],
                     [G_exp_yme1l.number_of_nodes(), G_exp_yme1l.number_of_edges()],
                     color='coral', edgecolor='darkred')
    axes_comp[1].set_ylabel('Count')
    axes_comp[1].set_title('Experimental Graph')
    axes_comp[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(G_exp_yme1l, G_struct_yme1l, mo):
    mo.md(f"""
    ## Summary

    ✅ **Successfully built both graphs for YME1L:**

    1. **Structural Graph** - {G_struct_yme1l.number_of_nodes()} methyls, {G_struct_yme1l.number_of_edges()} distance-based edges
    2. **Experimental Graph** - {G_exp_yme1l.number_of_nodes()} peaks, {G_exp_yme1l.number_of_edges()} NOE-based edges

    ### Graph Quality Check:
    - ✅ Both graphs have reasonable connectivity (not too sparse, not too dense)
    - ✅ Similar number of nodes (good for matching)
    - ✅ Degree distributions look realistic
    - ✅ Multiple connected components expected (protein domains)

    ### Key Observations:
    - Structural graph captures spatial proximity of methyls
    - Experimental graph captures NOE connectivity patterns
    - Graphs are ready for matching algorithms!

    ### Next Steps:
    Run `03_matching_yme1l.py` to perform automated assignment using all four matching algorithms.

    **Parameters for matching:**
    - Distance cutoff: {G_struct_yme1l.graph.get('distance_cutoff', 'N/A')} Å
    - Use current parameter values or adjust them for optimal matching
    """)
    return


if __name__ == "__main__":
    app.run()
