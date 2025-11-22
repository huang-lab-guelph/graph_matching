"""
Graph Construction Demo - Methyl Match

This notebook demonstrates the graph construction capabilities of methyl_match,
showing how to build network representations from structural and experimental data.
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
    mo.md(r"""
    # Methyl Match: Graph Construction Demo

    This notebook demonstrates the **preprocessing module** of the methyl_match library,
    which creates graph representations for automated methyl assignment.

    ## Overview

    The preprocessing module builds two types of networks:
    - **Methyl Network** - Graph from PDB structure (nodes=methyls, edges=distances)
    - **Peak Network** - Graph from NMR data (nodes=HMQC peaks, edges=13C-13C-1H methyl-methyl NOE correlations)

    These graphs are then matched to assign NMR peaks to structural methyls.
    """)
    return


@app.cell
def _():
    # Import required modules
    from methyl_match.reading import PDBParser, NOESYParser, HMQCParser
    from methyl_match.preprocessing import MethylNetworkBuilder, PeakNetworkBuilder
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import networkx as nx
    from pathlib import Path

    # Set up plotting style
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

    # Define data paths
    DATA_DIR = Path("../data/test")
    return (
        DATA_DIR,
        HMQCParser,
        MethylNetworkBuilder,
        NOESYParser,
        PDBParser,
        PeakNetworkBuilder,
        np,
        nx,
        pd,
        plt,
        sns,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1. Parse Input Data

    First, we'll load the structural and experimental data that we'll use to build our graphs.
    """)
    return


@app.cell
def _(DATA_DIR, HMQCParser, NOESYParser, PDBParser, mo):
    # Parse all data sources
    pdb_file = DATA_DIR / "1ubq.pdb"
    hmqc_file = DATA_DIR / "example_hmqc.txt"
    noesy_file = DATA_DIR / "example_methyl_noesy.txt"

    # Parse PDB
    pdb_parser = PDBParser(str(pdb_file))
    methyls = pdb_parser.extract_methyls()

    # Parse NMR data
    hmqc_parser = HMQCParser(str(hmqc_file))
    hmqc_peaks = hmqc_parser.parse()

    noesy_parser = NOESYParser(str(noesy_file))
    noe_peaks = noesy_parser.parse()

    mo.md(
        f"""
        ### Data Loaded

        - **Structure**: {len(methyls)} methyl groups from {pdb_file.name}
        - **HMQC**: {len(hmqc_peaks)} peaks (1H-13C correlations)
        - **NOESY**: {len(noe_peaks)} peaks (13C-13C-1H methyl-methyl NOE)
        """
    )
    return hmqc_peaks, methyls, noe_peaks


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2. Build Methyl Network (Structure Graph)

    The methyl network represents the spatial arrangement of methyl groups in the protein structure.
    Edges are created between methyls within a distance cutoff.
    """)
    return


@app.cell
def _(MethylNetworkBuilder, methyls, mo, nx):
    # Build methyl network with different distance cutoffs
    distance_cutoff = 10.0  # Ångströms

    methyl_builder = MethylNetworkBuilder(
        distance_cutoff=distance_cutoff,
        edge_weight_function='inverse',
        include_self_loops=False
    )

    methyl_network = methyl_builder.build_network(methyls)

    mo.md(
        f"""
        ### Methyl Network Statistics

        - **Distance cutoff**: {distance_cutoff} Å
        - **Nodes**: {methyl_network.graph.number_of_nodes()} methyls
        - **Edges**: {methyl_network.graph.number_of_edges()} spatial proximities
        - **Density**: {nx.density(methyl_network.graph):.3f}
        - **Average degree**: {sum(dict(methyl_network.graph.degree()).values()) / methyl_network.graph.number_of_nodes():.2f}
        """
    )
    return methyl_builder, methyl_network


@app.cell
def _(methyl_builder, methyl_network):
    # Get detailed statistics
    methyl_stats = methyl_builder.get_statistics(methyl_network)
    methyl_stats
    return


@app.cell
def _(methyl_network, nx, plt):
    # Visualize methyl network
    fig_methyl, (ax_methyl_graph, ax_methyl_degree) = plt.subplots(1, 2, figsize=(14, 6))

    # Graph visualization
    pos = nx.spring_layout(methyl_network.graph, seed=42, k=0.5, iterations=50)

    # Color nodes by residue type
    residue_colors = {
        'LEU': '#e74c3c',
        'VAL': '#3498db',
        'ILE': '#2ecc71',
        'ALA': '#f39c12',
        'THR': '#9b59b6',
        'MET': '#1abc9c'
    }

    node_colors = [residue_colors.get(methyl_network.methyls[i].residue_name, 'gray')
                   for i in range(len(methyl_network.methyls))]

    nx.draw_networkx_nodes(
        methyl_network.graph, pos,
        node_color=node_colors,
        node_size=200,
        alpha=0.8,
        ax=ax_methyl_graph
    )

    # Draw edges with width proportional to weight
    edges = methyl_network.graph.edges()
    weights = [methyl_network.graph[u][v]['weight'] * 2 for u, v in edges]

    nx.draw_networkx_edges(
        methyl_network.graph, pos,
        width=weights,
        alpha=0.3,
        ax=ax_methyl_graph
    )

    ax_methyl_graph.set_title('Methyl Network (colored by residue type)')
    ax_methyl_graph.axis('off')

    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=res)
                      for res, color in residue_colors.items()]
    ax_methyl_graph.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))

    # Degree distribution
    degrees = [d for n, d in methyl_network.graph.degree()]
    ax_methyl_degree.hist(degrees, bins=range(max(degrees)+2),
                         color='steelblue', edgecolor='black', alpha=0.7)
    ax_methyl_degree.set_xlabel('Degree (number of neighbors)')
    ax_methyl_degree.set_ylabel('Count')
    ax_methyl_degree.set_title('Degree Distribution')
    ax_methyl_degree.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(methyl_network, plt, sns):
    # Visualize distance matrix as heatmap
    fig_dist_heatmap, ax_dist_heatmap = plt.subplots(figsize=(10, 8))

    sns.heatmap(methyl_network.distance_matrix,
                cmap='viridis',
                square=True,
                cbar_kws={'label': 'Distance (Å)'},
                ax=ax_dist_heatmap)

    ax_dist_heatmap.set_title('Methyl Distance Matrix')
    ax_dist_heatmap.set_xlabel('Methyl Index')
    ax_dist_heatmap.set_ylabel('Methyl Index')

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3. Build Peak Network (Experimental Graph)

    The peak network represents NOE correlations between HMQC peaks from experimental data.
    For methyl-methyl assignment, we use 13C-13C-1H NOESY where edges connect peaks
    with observed NOE correlations.
    """)
    return


@app.cell
def _(PeakNetworkBuilder, hmqc_peaks, mo, noe_peaks, nx):
    # Build peak network
    intensity_threshold = 30000  # Minimum NOE intensity

    peak_builder = PeakNetworkBuilder(
        intensity_threshold=intensity_threshold,
        normalize_intensities=True,
        chemical_shift_tolerance={'h': 0.05, 'c': 0.5}
    )

    peak_network = peak_builder.build_network(hmqc_peaks, noe_peaks)

    mo.md(
        f"""
        ### Peak Network Statistics

        - **Intensity threshold**: {intensity_threshold}
        - **Nodes**: {peak_network.graph.number_of_nodes()} HMQC peaks
        - **Edges**: {peak_network.graph.number_of_edges()} NOE correlations (13C-13C-1H)
        - **Density**: {nx.density(peak_network.graph):.3f}
        - **Average degree**: {sum(dict(peak_network.graph.degree()).values()) / peak_network.graph.number_of_nodes():.2f}
        """
    )
    return peak_builder, peak_network


@app.cell
def _(peak_builder, peak_network):
    # Get detailed statistics
    peak_stats = peak_builder.get_statistics(peak_network)
    peak_stats
    return


@app.cell
def _(nx, peak_network, plt):
    # Visualize peak network
    fig_peak, (ax_peak_graph, ax_peak_degree) = plt.subplots(1, 2, figsize=(14, 6))

    # Graph visualization
    pos_peak = nx.spring_layout(peak_network.graph, seed=42, k=0.8, iterations=50)

    # Color nodes by assignment status
    node_colors_peak = ['lightcoral' if peak_network.labels[i].startswith('Peak_')
                       else 'lightgreen' for i in range(len(peak_network.labels))]

    nx.draw_networkx_nodes(
        peak_network.graph, pos_peak,
        node_color=node_colors_peak,
        node_size=300,
        alpha=0.8,
        ax=ax_peak_graph
    )

    # Draw edges with width proportional to NOE intensity
    edges_peak = peak_network.graph.edges()
    weights_peak = [peak_network.graph[u][v]['intensity'] * 5 for u, v in edges_peak]

    nx.draw_networkx_edges(
        peak_network.graph, pos_peak,
        width=weights_peak,
        alpha=0.4,
        edge_color='steelblue',
        ax=ax_peak_graph
    )

    # Draw labels for assigned peaks
    labels_to_show = {i: peak_network.labels[i]
                     for i in range(len(peak_network.labels))
                     if not peak_network.labels[i].startswith('Peak_')}

    nx.draw_networkx_labels(
        peak_network.graph, pos_peak,
        labels=labels_to_show,
        font_size=7,
        ax=ax_peak_graph
    )

    ax_peak_graph.set_title('Peak Network (green=assigned, red=unassigned)')
    ax_peak_graph.axis('off')

    # Degree distribution
    _degrees_peak_viz = [d for n, d in peak_network.graph.degree()]
    ax_peak_degree.hist(_degrees_peak_viz, bins=range(max(_degrees_peak_viz)+2),
                       color='coral', edgecolor='black', alpha=0.7)
    ax_peak_degree.set_xlabel('Degree (number of NOE correlations)')
    ax_peak_degree.set_ylabel('Count')
    ax_peak_degree.set_title('Degree Distribution')
    ax_peak_degree.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(peak_network, plt, sns):
    # Visualize NOE correlation matrix
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))

    sns.heatmap(peak_network.correlation_matrix,
                cmap='hot',
                square=True,
                cbar_kws={'label': 'NOE Intensity'},
                ax=ax_corr)

    ax_corr.set_title('NOE Correlation Matrix (13C-13C-1H)')
    ax_corr.set_xlabel('Peak Index')
    ax_corr.set_ylabel('Peak Index')

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4. Compare Networks

    Let's compare the structural and experimental networks to understand the matching problem.
    """)
    return


@app.cell
def _(methyl_network, mo, nx, peak_network):
    # Compare network properties
    import pandas as _pd_compare
    _comparison_data = {
        'Property': [
            'Nodes',
            'Edges',
            'Density',
            'Avg Degree',
            'Connected Components',
            'Avg Clustering'
        ],
        'Methyl Network': [
            methyl_network.graph.number_of_nodes(),
            methyl_network.graph.number_of_edges(),
            f"{nx.density(methyl_network.graph):.3f}",
            f"{sum(dict(methyl_network.graph.degree()).values()) / methyl_network.graph.number_of_nodes():.2f}",
            nx.number_connected_components(methyl_network.graph),
            f"{nx.average_clustering(methyl_network.graph):.3f}"
        ],
        'Peak Network': [
            peak_network.graph.number_of_nodes(),
            peak_network.graph.number_of_edges(),
            f"{nx.density(peak_network.graph):.3f}",
            f"{sum(dict(peak_network.graph.degree()).values()) / peak_network.graph.number_of_nodes():.2f}",
            nx.number_connected_components(peak_network.graph),
            f"{nx.average_clustering(peak_network.graph):.3f}"
        ]
    }

    comparison_df = _pd_compare.DataFrame(_comparison_data)

    mo.md("""
    ### Network Comparison

    Understanding the similarities and differences between the two networks is key to the matching problem.
    """)

    comparison_df
    return


@app.cell
def _(methyl_network, peak_network, plt):
    # Side-by-side comparison of degree distributions
    fig_compare, (ax_comp_methyl, ax_comp_peak) = plt.subplots(1, 2, figsize=(12, 4))

    # Methyl network degrees
    degrees_methyl = [d for n, d in methyl_network.graph.degree()]
    ax_comp_methyl.hist(degrees_methyl, bins=range(max(degrees_methyl)+2),
                       color='steelblue', edgecolor='black', alpha=0.7)
    ax_comp_methyl.set_xlabel('Degree')
    ax_comp_methyl.set_ylabel('Count')
    ax_comp_methyl.set_title('Methyl Network Degree Distribution')
    ax_comp_methyl.grid(axis='y', alpha=0.3)

    # Peak network degrees
    _degrees_peak_compare = [d for n, d in peak_network.graph.degree()]
    ax_comp_peak.hist(_degrees_peak_compare, bins=range(max(_degrees_peak_compare)+2),
                     color='coral', edgecolor='black', alpha=0.7)
    ax_comp_peak.set_xlabel('Degree')
    ax_comp_peak.set_ylabel('Count')
    ax_comp_peak.set_title('Peak Network Degree Distribution')
    ax_comp_peak.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 5. Node and Edge Features

    Both networks include rich feature vectors that can be used by matching algorithms.
    """)
    return


@app.cell
def _(methyl_network, mo, np, pd):
    # Show methyl network node features
    _methyl_features_sample = []
    for _i in range(min(5, len(methyl_network.node_features))):
        _features = methyl_network.node_features[_i]
        _methyl_features_sample.append({
            'Node': _i,
            'Label': methyl_network.labels[_i],
            'Features': f"{len(_features)}D",
            'X': f"{_features[0]:.2f}",
            'Y': f"{_features[1]:.2f}",
            'Z': f"{_features[2]:.2f}",
            'Residue Type': f"onehot[{np.argmax(_features[3:9])}]"
        })

    df_methyl_features = pd.DataFrame(_methyl_features_sample)

    mo.md(f"""
    ### Methyl Network Node Features

    Each methyl node has a **{len(methyl_network.node_features[0])}D feature vector**:
    - 3D coordinates (X, Y, Z)
    - One-hot residue type (LEU, VAL, ILE, ALA, THR, MET)
    - Normalized residue number
    """)

    df_methyl_features
    return


@app.cell
def _(mo, peak_network):
    # Show peak network node features
    import pandas as _pd_peak
    _peak_features_sample = []
    for _j in range(min(5, len(peak_network.node_features))):
        _peak_features = peak_network.node_features[_j]
        _peak_features_sample.append({
            'Node': _j,
            'Label': peak_network.labels[_j],
            'Features': f"{len(_peak_features)}D",
            '1H shift': f"{_peak_features[0]:.3f}",
            '13C shift': f"{_peak_features[1]:.3f}",
            'Intensity': f"{_peak_features[2]:.4f}",
            'Degree': f"{_peak_features[3]:.1f}"
        })

    df_peak_features = _pd_peak.DataFrame(_peak_features_sample)

    mo.md(f"""
    ### Peak Network Node Features

    Each peak node has a **{len(peak_network.node_features[0])}D feature vector**:
    - 1H chemical shift
    - 13C chemical shift
    - Normalized intensity
    - Node degree (computed from graph)
    """)

    df_peak_features
    return


@app.cell
def _(methyl_network, mo, peak_network):
    # Show edge features
    mo.md(f"""
    ### Edge Features

    **Methyl Network Edges** ({len(methyl_network.edge_features)} edges):
    - Distance (Å)
    - Edge weight (based on distance)

    **Peak Network Edges** ({len(peak_network.edge_features)} edges):
    - NOE intensity (normalized)
    - Edge weight (same as intensity)

    These features can be used by graph neural networks for learning-based matching.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 6. Effect of Parameters

    Let's explore how different parameters affect the graph structure.
    """)
    return


@app.cell
def _(MethylNetworkBuilder, methyls, nx, plt):
    # Test different distance cutoffs
    import pandas as _pd_cutoff
    _cutoffs = [6.0, 8.0, 10.0, 12.0, 14.0]

    _cutoff_results = []
    for _cutoff in _cutoffs:
        _builder = MethylNetworkBuilder(distance_cutoff=_cutoff)
        _network = _builder.build_network(methyls)
        _cutoff_results.append({
            'cutoff': _cutoff,
            'edges': _network.graph.number_of_edges(),
            'density': nx.density(_network.graph),
            'avg_degree': sum(dict(_network.graph.degree()).values()) / _network.graph.number_of_nodes()
        })

    df_cutoff = _pd_cutoff.DataFrame(_cutoff_results)

    # Plot effects
    fig_param, (ax_param_edges, ax_param_density) = plt.subplots(1, 2, figsize=(12, 4))

    ax_param_edges.plot(df_cutoff['cutoff'], df_cutoff['edges'],
                       marker='o', linewidth=2, markersize=8, color='steelblue')
    ax_param_edges.set_xlabel('Distance Cutoff (Å)')
    ax_param_edges.set_ylabel('Number of Edges')
    ax_param_edges.set_title('Effect of Distance Cutoff on Graph Size')
    ax_param_edges.grid(alpha=0.3)

    ax_param_density.plot(df_cutoff['cutoff'], df_cutoff['avg_degree'],
                         marker='o', linewidth=2, markersize=8, color='coral')
    ax_param_density.set_xlabel('Distance Cutoff (Å)')
    ax_param_density.set_ylabel('Average Degree')
    ax_param_density.set_title('Effect of Distance Cutoff on Connectivity')
    ax_param_density.grid(alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(PeakNetworkBuilder, hmqc_peaks, noe_peaks, nx, plt):
    # Test different intensity thresholds
    import pandas as _pd_threshold
    _thresholds = [0, 20000, 30000, 40000, 50000]

    _threshold_results = []
    for _threshold in _thresholds:
        _builder_thresh = PeakNetworkBuilder(intensity_threshold=_threshold)
        _network_thresh = _builder_thresh.build_network(hmqc_peaks, noe_peaks)
        _threshold_results.append({
            'threshold': _threshold,
            'edges': _network_thresh.graph.number_of_edges(),
            'density': nx.density(_network_thresh.graph),
            'avg_degree': sum(dict(_network_thresh.graph.degree()).values()) / _network_thresh.graph.number_of_nodes()
        })

    df_threshold = _pd_threshold.DataFrame(_threshold_results)

    # Plot effects
    fig_thresh, (ax_thresh_edges, ax_thresh_density) = plt.subplots(1, 2, figsize=(12, 4))

    ax_thresh_edges.plot(df_threshold['threshold'], df_threshold['edges'],
                        marker='s', linewidth=2, markersize=8, color='green')
    ax_thresh_edges.set_xlabel('Intensity Threshold')
    ax_thresh_edges.set_ylabel('Number of Edges')
    ax_thresh_edges.set_title('Effect of Intensity Threshold on Graph Size')
    ax_thresh_edges.grid(alpha=0.3)

    ax_thresh_density.plot(df_threshold['threshold'], df_threshold['avg_degree'],
                          marker='s', linewidth=2, markersize=8, color='purple')
    ax_thresh_density.set_xlabel('Intensity Threshold')
    ax_thresh_density.set_ylabel('Average Degree')
    ax_thresh_density.set_title('Effect of Intensity Threshold on Connectivity')
    ax_thresh_density.grid(alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    This notebook demonstrated:

    1. **Methyl Network Construction**: Built spatial graph from PDB structure
    2. **Peak Network Construction**: Built experimental graph from 13C-13C-1H NOESY and HMQC data
    3. **Network Comparison**: Analyzed similarities and differences between networks
    4. **Feature Extraction**: Showed rich node and edge features for matching
    5. **Parameter Effects**: Explored how cutoffs affect graph structure

    ### Key Insights

    - ✅ Both networks are sparse graphs with similar structural properties
    - ✅ Distance cutoff and intensity threshold significantly affect connectivity
    - ✅ Rich feature vectors enable sophisticated matching algorithms
    - ✅ Graph properties (degree, clustering) can guide matching

    ### Next Steps

    - **Phase 3**: Graph matching algorithms (greedy, optimal, ML-based)
    - **Phase 4**: Assignment output and validation
    - **Phase 5**: Complete workflow examples

    ### Key Features Demonstrated

    - ✅ Flexible distance-based graph construction
    - ✅ 13C-13C-1H methyl-methyl NOE correlation graphs
    - ✅ Multiple edge weight functions
    - ✅ Comprehensive feature engineering
    - ✅ Network statistics and visualization
    - ✅ Parameter sensitivity analysis

    See the [documentation](../README.md) for more information.
    """)
    return


if __name__ == "__main__":
    app.run()
