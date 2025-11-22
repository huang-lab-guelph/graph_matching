"""
Matching Algorithms Demo

This interactive notebook demonstrates all graph matching algorithms
in methyl_match and compares their performance on synthetic and real data.
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
    # Graph Matching Algorithms Demo

    This notebook demonstrates the four graph matching algorithms implemented
    in `methyl_match` for automated methyl assignment:

    1. **GreedyMatcher** - Fast heuristic matching
    2. **HungarianMatcher** - Optimal Linear Assignment Problem
    3. **QAPMatcher** - Quadratic Assignment Problem (topology-aware)
    4. **SpectralMatcher** - Spectral graph matching

    We'll compare their performance on:
    - Simple synthetic graphs
    - Realistic protein-like graphs
    - Real NMR data (ubiquitin)
    """)
    return


@app.cell
def _():
    # Import libraries
    import sys
    import numpy as np
    import networkx as nx
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path
    import time

    # Add src to path
    project_root = Path.cwd().parent
    sys.path.insert(0, str(project_root / "src"))

    # Import methyl_match modules
    from methyl_match.reading import PDBParser, NOESYParser, HMQCParser
    from methyl_match.preprocessing import MethylNetworkBuilder, PeakNetworkBuilder
    from methyl_match.matching import (
        GreedyMatcher,
        HungarianMatcher,
        QAPMatcher,
        SpectralMatcher,
        validate_assignment,
    )

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    return (
        GreedyMatcher,
        HMQCParser,
        HungarianMatcher,
        MethylNetworkBuilder,
        NOESYParser,
        PDBParser,
        PeakNetworkBuilder,
        QAPMatcher,
        SpectralMatcher,
        np,
        nx,
        pd,
        plt,
        project_root,
        time,
        validate_assignment,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Part 1: Simple Synthetic Example
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Let's create a simple example where we know the ground truth.
    We'll create two identical graphs with 5 nodes each.
    """)
    return


@app.cell
def _(np, nx):
    # Create ground truth: simple line graph
    def create_synthetic_graphs(n_nodes=5, noise_level=0.0):
        """Create synthetic experimental and structural graphs."""
        # Structural graph (ground truth)
        G_struct = nx.Graph()
        residue_types = ['LEU', 'VAL', 'ILE', 'ALA', 'THR']

        for i in range(n_nodes):
            G_struct.add_node(
                i,
                residue_type=residue_types[i % len(residue_types)],
                position=np.array([i * 3.0, 0.0, 0.0])
            )

        # Add edges (line graph)
        for i in range(n_nodes - 1):
            G_struct.add_edge(i, i + 1, weight=3.0)

        # Experimental graph (same topology, possibly noisy)
        G_exp = nx.Graph()

        # Add same nodes (possibly shuffled or with noise)
        for i in range(n_nodes):
            # Same residue types
            G_exp.add_node(
                i,
                residue_type=residue_types[i % len(residue_types)],
                h_shift=1.0 + i * 0.1 + np.random.normal(0, noise_level),
                c_shift=20.0 + i * 0.5 + np.random.normal(0, noise_level * 5),
            )

        # Add same edges
        for i in range(n_nodes - 1):
            G_exp.add_edge(i, i + 1, weight=0.8 + np.random.normal(0, noise_level))

        return G_exp, G_struct

    # Create graphs
    G_exp_simple, G_struct_simple = create_synthetic_graphs(n_nodes=5, noise_level=0.0)
    return G_exp_simple, G_struct_simple


@app.cell
def _(G_exp_simple, G_struct_simple, mo):
    mo.md(
        f"""
        **Created graphs:**
        - Experimental: {G_exp_simple.number_of_nodes()} nodes, {G_exp_simple.number_of_edges()} edges
        - Structural: {G_struct_simple.number_of_nodes()} nodes, {G_struct_simple.number_of_edges()} edges

        Since we created identical graphs, we expect **perfect matching** (node i → node i).
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Run All Matchers
    """)
    return


@app.cell
def _(
    G_exp_simple,
    G_struct_simple,
    GreedyMatcher,
    HungarianMatcher,
    QAPMatcher,
    SpectralMatcher,
    time,
    validate_assignment,
):
    # Initialize matchers
    matchers_simple = {
        'Greedy': GreedyMatcher(),
        'Hungarian': HungarianMatcher(),
        'QAP': QAPMatcher(method='faq', options={'maxiter': 30}),
        'Spectral': SpectralMatcher(method='sm'),
    }

    # Run all matchers
    results_simple = {}

    for name, matcher in matchers_simple.items():
        start = time.time()
        result = matcher.match(G_exp_simple, G_struct_simple)
        elapsed = time.time() - start

        # Validate
        metrics = validate_assignment(
            result.assignments,
            G_exp_simple,
            G_struct_simple
        )

        results_simple[name] = {
            'result': result,
            'runtime': elapsed,
            'metrics': metrics,
        }
    return (results_simple,)


@app.cell
def __(pd, results_simple):
    # Create comparison table
    comparison_data_simple = []

    for algo_name_simple, algo_data_simple in results_simple.items():
        algo_result_simple = algo_data_simple['result']
        algo_metrics_simple = algo_data_simple['metrics']

        comparison_data_simple.append({
            'Algorithm': algo_name_simple,
            'Assignments': algo_result_simple.num_assignments,
            'Mean Confidence': f"{algo_result_simple.mean_confidence:.3f}",
            'Topology Consistency': f"{algo_metrics_simple['topology_consistency']:.3f}",
            'Distance Consistency': f"{algo_metrics_simple['distance_consistency']:.3f}",
            'Runtime (s)': f"{algo_data_simple['runtime']:.4f}",
        })

    comparison_df_simple = pd.DataFrame(comparison_data_simple)
    comparison_df_simple
    return (comparison_df_simple,)


@app.cell
def _(comparison_df_simple, mo):
    mo.md(
        f"""
        **Results on Simple Graphs:**

        {mo.as_html(comparison_df_simple)}

        All algorithms should achieve perfect matching on this simple case.
        """
    )
    return


@app.cell
def _(plt, results_simple):
    # Visualize confidence scores
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    for idx, (name, data) in enumerate(results_simple.items()):
        result = data['result']

        if result.num_assignments > 0:
            assignments_list = result.get_assignment_list()
            exp_ids = [a[0] for a in assignments_list]
            confidences = [a[2] for a in assignments_list]

            axes[idx].bar(exp_ids, confidences, color='skyblue', edgecolor='navy')
            axes[idx].set_xlabel('Experimental Node ID')
            axes[idx].set_ylabel('Confidence Score')
            axes[idx].set_title(f'{name}\n(Mean: {result.mean_confidence:.2f})')
            axes[idx].set_ylim([0, 1.1])
            axes[idx].axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 2: Realistic Protein-Like Graphs
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Now let's create more realistic graphs that mimic protein structures:
    - 20 nodes (methyls)
    - Distance-based connectivity
    - Mixed residue types
    """)
    return


@app.cell
def _(np, nx):
    def create_realistic_graphs(n_nodes=20, distance_cutoff=10.0):
        """Create realistic protein-like graphs."""
        # Structural graph with 3D positions
        G_struct = nx.Graph()
        residue_types = ['LEU', 'VAL', 'ILE', 'ALA', 'THR', 'MET']

        # Generate semi-random 3D structure
        np.random.seed(42)
        positions = []
        for i in range(n_nodes):
            # Create rough helix-like structure
            angle = i * 100 * np.pi / 180
            x = 5 * np.cos(angle)
            y = 5 * np.sin(angle)
            z = i * 1.5
            positions.append(np.array([x, y, z]))

        # Add nodes
        for i in range(n_nodes):
            G_struct.add_node(
                i,
                residue_type=residue_types[i % len(residue_types)],
                position=positions[i],
                residue_number=i + 1,
            )

        # Add edges based on distance
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < distance_cutoff:
                    G_struct.add_edge(i, j, weight=dist)

        # Experimental graph (same topology, noisier features)
        G_exp = nx.Graph()

        for i in range(n_nodes):
            G_exp.add_node(
                i,
                residue_type=residue_types[i % len(residue_types)],
                h_shift=1.0 + i * 0.05 + np.random.normal(0, 0.1),
                c_shift=20.0 + i * 0.3 + np.random.normal(0, 0.5),
                intensity=np.random.uniform(0.5, 1.0),
            )

        # Add edges with NOE-like intensities
        for i, j in G_struct.edges():
            dist = G_struct[i][j]['weight']
            # NOE intensity ~ 1/r^6
            intensity = min(1.0, 100.0 / (dist ** 3))  # Simplified
            G_exp.add_edge(i, j, weight=intensity)

        return G_exp, G_struct

    # Create realistic graphs
    G_exp_real, G_struct_real = create_realistic_graphs(n_nodes=20, distance_cutoff=10.0)
    return G_exp_real, G_struct_real


@app.cell
def _(G_exp_real, G_struct_real, mo):
    mo.md(
        f"""
        **Created realistic graphs:**
        - Experimental: {G_exp_real.number_of_nodes()} nodes, {G_exp_real.number_of_edges()} edges
        - Structural: {G_struct_real.number_of_nodes()} nodes, {G_struct_real.number_of_edges()} edges

        Average degree: {2 * G_exp_real.number_of_edges() / G_exp_real.number_of_nodes():.1f}
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ### Run All Matchers on Realistic Graphs
    """)
    return


@app.cell
def _(
    G_exp_real,
    G_struct_real,
    GreedyMatcher,
    HungarianMatcher,
    QAPMatcher,
    SpectralMatcher,
    time,
    validate_assignment,
):
    # Initialize matchers
    matchers_real = {
        'Greedy': GreedyMatcher(use_topology=True),
        'Hungarian': HungarianMatcher(),
        'QAP': QAPMatcher(method='faq', topology_weight=0.5, options={'maxiter': 30}),
        'Spectral': SpectralMatcher(method='ipfp'),
    }

    # Run all matchers
    results_real = {}

    for name, matcher in matchers_real.items():
        print(f"Running {name}...")
        start = time.time()
        result = matcher.match(G_exp_real, G_struct_real)
        elapsed = time.time() - start

        # Validate
        metrics = validate_assignment(
            result.assignments,
            G_exp_real,
            G_struct_real
        )

        results_real[name] = {
            'result': result,
            'runtime': elapsed,
            'metrics': metrics,
        }

    print("Done!")
    return (results_real,)


@app.cell
def __(pd, results_real):
    # Create comparison table
    comparison_data_real = []

    for algo_name_real, algo_data_real in results_real.items():
        algo_result_real = algo_data_real['result']
        algo_metrics_real = algo_data_real['metrics']

        comparison_data_real.append({
            'Algorithm': algo_name_real,
            'Assignments': algo_result_real.num_assignments,
            'Mean Confidence': f"{algo_result_real.mean_confidence:.3f}",
            'Topology Consistency': f"{algo_metrics_real['topology_consistency']:.3f}",
            'Distance Consistency': f"{algo_metrics_real['distance_consistency']:.3f}",
            'Runtime (s)': f"{algo_data_real['runtime']:.4f}",
        })

    comparison_df_real = pd.DataFrame(comparison_data_real)
    comparison_df_real
    return (comparison_df_real,)


@app.cell
def _(plt, results_real):
    # Visualize comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Number of assignments
    algorithms = list(results_real.keys())
    n_assignments = [results_real[name]['result'].num_assignments for name in algorithms]

    axes[0, 0].bar(algorithms, n_assignments, color='steelblue')
    axes[0, 0].set_ylabel('Number of Assignments')
    axes[0, 0].set_title('Assignments Made')
    axes[0, 0].set_ylim([0, max(n_assignments) * 1.1])

    # 2. Mean confidence
    confidences = [results_real[name]['result'].mean_confidence for name in algorithms]

    axes[0, 1].bar(algorithms, confidences, color='coral')
    axes[0, 1].set_ylabel('Mean Confidence')
    axes[0, 1].set_title('Average Confidence Score')
    axes[0, 1].set_ylim([0, 1.0])

    # 3. Topology consistency
    topo_consistency = [results_real[name]['metrics']['topology_consistency'] for name in algorithms]

    axes[1, 0].bar(algorithms, topo_consistency, color='lightgreen')
    axes[1, 0].set_ylabel('Topology Consistency')
    axes[1, 0].set_title('Edge Structure Agreement')
    axes[1, 0].set_ylim([0, 1.0])

    # 4. Runtime
    runtimes = [results_real[name]['runtime'] for name in algorithms]

    axes[1, 1].bar(algorithms, runtimes, color='plum')
    axes[1, 1].set_ylabel('Runtime (seconds)')
    axes[1, 1].set_title('Computational Performance')
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Part 3: Real NMR Data (Ubiquitin)
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    Now let's try on real NMR data from ubiquitin.
    We'll load the PDB structure and NMR peak lists.
    """)
    return


@app.cell
def _(HMQCParser, NOESYParser, PDBParser, project_root):
    # Check if test data exists
    test_data_dir = project_root / "data" / "test"

    pdb_file = test_data_dir / "1ubq.pdb"
    noesy_file = test_data_dir / "1ubq_noesy.txt"
    hmqc_file = test_data_dir / "1ubq_hmqc.txt"

    has_real_data = pdb_file.exists() and hmqc_file.exists()

    if has_real_data:
        # Parse files
        pdb_parser = PDBParser(str(pdb_file))
        methyls = pdb_parser.extract_methyls()

        hmqc_parser = HMQCParser(str(hmqc_file))
        hmqc_peaks = hmqc_parser.parse_hmqc()

        if noesy_file.exists():
            noesy_parser = NOESYParser(str(noesy_file))
            noesy_peaks = noesy_parser.parse_noesy()
        else:
            noesy_peaks = []

        print(f"Loaded ubiquitin data:")
        print(f"  Methyls: {len(methyls)}")
        print(f"  HMQC peaks: {len(hmqc_peaks)}")
        print(f"  NOESY peaks: {len(noesy_peaks)}")
    else:
        print("Real NMR data not found. Skipping this section.")
        methyls = None
        hmqc_peaks = None
        noesy_peaks = None
    return has_real_data, hmqc_peaks, methyls, noesy_peaks


@app.cell
def _(
    MethylNetworkBuilder,
    PeakNetworkBuilder,
    has_real_data,
    hmqc_peaks,
    methyls,
    noesy_peaks,
):
    if has_real_data:
        # Build graphs
        methyl_builder = MethylNetworkBuilder()
        G_struct_ubq = methyl_builder.build_network(methyls, distance_cutoff=10.0)

        peak_builder = PeakNetworkBuilder()
        G_exp_ubq = peak_builder.build_network(
            hmqc_peaks, noesy_peaks, intensity_threshold=0.3
        )

        print(f"Built graphs:")
        print(f"  Structural: {G_struct_ubq.number_of_nodes()} nodes, {G_struct_ubq.number_of_edges()} edges")
        print(f"  Experimental: {G_exp_ubq.number_of_nodes()} nodes, {G_exp_ubq.number_of_edges()} edges")
    else:
        G_struct_ubq = None
        G_exp_ubq = None
    return G_exp_ubq, G_struct_ubq


@app.cell
def _(mo):
    mo.md("""
    ### Run Matchers on Real Data
    """)
    return


@app.cell
def _(
    G_exp_ubq,
    G_struct_ubq,
    GreedyMatcher,
    HungarianMatcher,
    QAPMatcher,
    has_real_data,
    time,
    validate_assignment,
):
    if has_real_data and G_exp_ubq is not None and G_struct_ubq is not None:
        # Initialize matchers (skip spectral for speed)
        matchers_ubq = {
            'Greedy': GreedyMatcher(use_topology=True),
            'Hungarian': HungarianMatcher(),
            'QAP': QAPMatcher(method='faq', topology_weight=0.6, options={'maxiter': 50}),
        }

        # Run matchers
        results_ubq = {}

        for name, matcher in matchers_ubq.items():
            print(f"Running {name} on ubiquitin...")
            start = time.time()
            result = matcher.match(G_exp_ubq, G_struct_ubq)
            elapsed = time.time() - start

            # Validate
            metrics = validate_assignment(
                result.assignments,
                G_exp_ubq,
                G_struct_ubq
            )

            results_ubq[name] = {
                'result': result,
                'runtime': elapsed,
                'metrics': metrics,
            }

        print("Done!")
    else:
        results_ubq = None
        matchers_ubq = None
    return (results_ubq,)


@app.cell
def __(has_real_data, pd, results_ubq):
    if has_real_data and results_ubq is not None:
        # Create comparison table
        comparison_data_ubq = []

        for algo_name_ubq, algo_data_ubq in results_ubq.items():
            algo_result_ubq = algo_data_ubq['result']
            algo_metrics_ubq = algo_data_ubq['metrics']

            comparison_data_ubq.append({
                'Algorithm': algo_name_ubq,
                'Assignments': algo_result_ubq.num_assignments,
                'Unassigned': algo_result_ubq.num_unassigned,
                'Mean Confidence': f"{algo_result_ubq.mean_confidence:.3f}",
                'Topology Consistency': f"{algo_metrics_ubq['topology_consistency']:.3f}",
                'Runtime (s)': f"{algo_data_ubq['runtime']:.4f}",
            })

        comparison_df_ubq = pd.DataFrame(comparison_data_ubq)
        comparison_df_ubq
    else:
        comparison_df_ubq = None
    return (comparison_df_ubq,)


@app.cell
def _(has_real_data, plt, results_ubq):
    if has_real_data and results_ubq is not None:
        # Show top assignments from QAP
        qap_result = results_ubq['QAP']['result']

        # Get top 20 assignments by confidence
        top_assignments = qap_result.get_assignment_list()[:20]

        fig, ax = plt.subplots(figsize=(12, 6))

        exp_ids = [f"P{a[0]}" for a in top_assignments]
        confidences = [a[2] for a in top_assignments]

        colors = ['green' if c > 0.7 else 'orange' if c > 0.5 else 'red' for c in confidences]

        ax.barh(exp_ids, confidences, color=colors, edgecolor='black')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Peak ID')
        ax.set_title('Top 20 QAP Assignments (Ubiquitin)')
        ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0.7, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlim([0, 1.0])

        plt.tight_layout()
        plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary & Recommendations

    Based on the demonstrations above:

    ### Algorithm Performance

    1. **QAPMatcher** - **Recommended**
       - Best balance of accuracy and performance
       - Topology-aware (considers edge structure)
       - Fast enough for practical use (seconds)
       - Good confidence scores

    2. **HungarianMatcher** - **Fast Baseline**
       - Guaranteed optimal for node-based matching
       - Very fast O(n³) runtime
       - Doesn't consider topology
       - Good for simple cases or validation

    3. **GreedyMatcher** - **Quick Check**
       - Fastest option
       - Good for initial testing
       - Can be used with topology
       - May miss optimal solutions

    4. **SpectralMatcher** - **Alternative**
       - Captures global structure
       - Can handle noisy data
       - Slower than others
       - Multiple refinement methods available

    ### Typical Workflow

    ```python
    # 1. Start with QAP
    qap_result = QAPMatcher().match(G_exp, G_struct)

    # 2. Filter by confidence
    high_conf = qap_result.filter_by_confidence(0.7)

    # 3. Validate
    metrics = validate_assignment(high_conf.assignments, G_exp, G_struct)

    # 4. Compare with Hungarian
    hun_result = HungarianMatcher().match(G_exp, G_struct)
    ```

    ### Parameter Tuning

    - **QAP topology_weight**: 0.5-0.7 (higher = more emphasis on edges)
    - **Confidence threshold**: 0.5-0.7 for high-confidence assignments
    - **Distance cutoff**: 8-12Å for structural graphs
    - **Intensity threshold**: 0.3-0.5 for NOESY edges
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Next Steps

    - Try on your own protein structures
    - Experiment with different parameter values
    - Compare results against manual assignments
    - Export results using the writing module (Phase 4)
    """)
    return


if __name__ == "__main__":
    app.run()
