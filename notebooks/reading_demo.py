"""Methyl Match Reading Module Demo

This notebook demonstrates the file reading capabilities of the methyl_match library.
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
    # Methyl Match: Reading Module Demo

    This notebook demonstrates the file parsing capabilities of the **methyl_match** library,
    which provides tools for automated methyl assignment in NMR spectroscopy.

    ## Overview

    The reading module can parse:
    - **PDB structures** - Extract methyl groups from protein structures
    - **NOESY peak lists** - Parse NOE connectivity data (multiple formats)
    - **HMQC peak lists** - Parse 2D correlation data (multiple formats)
    """)
    return


@app.cell
def _():
    # Import the reading module
    from methyl_match.reading import PDBParser, NOESYParser, HMQCParser
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    # Set up plotting style
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

    # Define data paths
    DATA_DIR = Path("../data/test")
    return DATA_DIR, HMQCParser, NOESYParser, PDBParser, np, pd, plt, sns


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 1. PDB Structure Parsing

    Extract methyl groups from protein structures. We'll use ubiquitin (1UBQ) as an example.
    """)
    return


@app.cell
def _(DATA_DIR, PDBParser, mo):
    # Parse PDB structure
    pdb_file = DATA_DIR / "1ubq.pdb"
    pdb_parser = PDBParser(str(pdb_file))
    methyls = pdb_parser.extract_methyls()

    mo.md(
        f"""
        ### Ubiquitin Structure

        - **File**: `{pdb_file.name}`
        - **Methyl groups found**: {len(methyls)}
        - **Residue types**: LEU, VAL, ILE, ALA, THR, MET
        """
    )
    return methyls, pdb_parser


@app.cell
def _(methyls, pd):
    # Create a DataFrame of methyl groups
    methyl_data = []
    for m in methyls:
        methyl_data.append({
            'Label': m.label,
            'Residue': m.residue_name,
            'Number': m.residue_number,
            'Atom': m.atom_name,
            'Chain': m.chain_id,
            'X': m.coordinates[0],
            'Y': m.coordinates[1],
            'Z': m.coordinates[2]
        })

    df_methyls = pd.DataFrame(methyl_data)
    df_methyls
    return (df_methyls,)


@app.cell
def _(df_methyls, plt):
    # Visualize methyl distribution by residue type
    fig_methyls, (ax_methyl_bar, ax_methyl_3d) = plt.subplots(1, 2, figsize=(12, 4))

    # Count by residue type
    residue_counts = df_methyls['Residue'].value_counts()
    ax_methyl_bar.bar(residue_counts.index, residue_counts.values, color='steelblue')
    ax_methyl_bar.set_xlabel('Residue Type')
    ax_methyl_bar.set_ylabel('Count')
    ax_methyl_bar.set_title('Methyl Groups by Residue Type')
    ax_methyl_bar.grid(axis='y', alpha=0.3)

    # 3D scatter of methyl positions
    ax_methyl_3d = plt.subplot(122, projection='3d')
    residue_types = df_methyls['Residue'].unique()
    colors_methyl = plt.cm.tab10(range(len(residue_types)))
    for i, res in enumerate(residue_types):
        mask_methyl = df_methyls['Residue'] == res
        ax_methyl_3d.scatter(df_methyls[mask_methyl]['X'],
                   df_methyls[mask_methyl]['Y'],
                   df_methyls[mask_methyl]['Z'],
                   label=res, alpha=0.7, s=50, c=[colors_methyl[i]])
    ax_methyl_3d.set_xlabel('X (Å)')
    ax_methyl_3d.set_ylabel('Y (Å)')
    ax_methyl_3d.set_zlabel('Z (Å)')
    ax_methyl_3d.set_title('3D Distribution of Methyl Groups')
    ax_methyl_3d.legend(loc='upper left', bbox_to_anchor=(1.1, 1))

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(methyls, np, pdb_parser, plt):
    # Calculate and visualize distance matrix
    dist_matrix, labels = pdb_parser.get_distance_matrix(methyls)

    fig_dist, ax_dist = plt.subplots(figsize=(10, 8))
    im_dist = ax_dist.imshow(dist_matrix, cmap='viridis', aspect='auto')
    ax_dist.set_title('Pairwise Distance Matrix of Methyl Groups (Å)')
    ax_dist.set_xlabel('Methyl Index')
    ax_dist.set_ylabel('Methyl Index')
    cbar_dist = plt.colorbar(im_dist, ax=ax_dist)
    cbar_dist.set_label('Distance (Å)', rotation=270, labelpad=20)
    plt.tight_layout()

    # Statistics
    non_zero_distances = dist_matrix[np.triu_indices_from(dist_matrix, k=1)]
    dist_stats = {
        'Min': f"{non_zero_distances.min():.2f} Å",
        'Max': f"{non_zero_distances.max():.2f} Å",
        'Mean': f"{non_zero_distances.mean():.2f} Å",
        'Median': f"{np.median(non_zero_distances):.2f} Å"
    }

    plt.gca(), dist_stats
    return (labels,)


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 2. NOESY Peak List Parsing

    Parse NOE connectivity data from NOESY spectra. The parser supports multiple formats:
    - XEASY (tab/space-separated)
    - Sparky (assignment-based)
    - CSV
    - NMRPipe
    """)
    return


@app.cell
def _(DATA_DIR, NOESYParser, mo):
    # Parse NOESY peaks
    noesy_file = DATA_DIR / "example_methyl_noesy.txt"
    noesy_parser = NOESYParser(str(noesy_file))
    detected_format = noesy_parser.detect_format()
    noe_peaks = noesy_parser.parse()

    mo.md(
        f"""
        ### NOESY Peak List

        - **File**: `{noesy_file.name}`
        - **Detected format**: {detected_format}
        - **Peaks found**: {len(noe_peaks)}
        - **Assigned peaks**: {len([p for p in noe_peaks if p.assignment1 and p.assignment2])}
        """
    )
    return noe_peaks, noesy_parser


@app.cell
def _(noe_peaks, pd):
    # Create DataFrame of NOE peaks
    noe_data = []
    for peak in noe_peaks:
        noe_data.append({
            'Index': peak.index,
            'w1 (ppm)': peak.w1,
            'w2 (ppm)': peak.w2,
            'w3 (ppm)': peak.w3 if peak.w3 else None,
            'Intensity': peak.intensity,
            'From': peak.assignment1,
            'To': peak.assignment2
        })

    df_noe = pd.DataFrame(noe_data)
    df_noe.head(10)
    return (df_noe,)


@app.cell
def _(df_noe, plt):
    # Visualize NOE intensity distribution
    fig_noe, (ax_noe_hist, ax_noe_scatter) = plt.subplots(1, 2, figsize=(12, 4))

    # Intensity histogram
    ax_noe_hist.hist(df_noe['Intensity'], bins=15, color='coral', edgecolor='black', alpha=0.7)
    ax_noe_hist.set_xlabel('Intensity')
    ax_noe_hist.set_ylabel('Count')
    ax_noe_hist.set_title('NOE Peak Intensity Distribution')
    ax_noe_hist.grid(axis='y', alpha=0.3)

    # Chemical shift scatter
    ax_noe_scatter.scatter(df_noe['w1 (ppm)'], df_noe['w2 (ppm)'],
                s=df_noe['Intensity']/500, alpha=0.6, c=df_noe['Intensity'],
                cmap='plasma')
    ax_noe_scatter.set_xlabel('w1 (1H ppm)')
    ax_noe_scatter.set_ylabel('w2 (1H ppm)')
    ax_noe_scatter.set_title('NOE Peak Distribution (size = intensity)')
    ax_noe_scatter.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(labels, noe_peaks, noesy_parser, np, plt, sns):
    # Create correlation matrix
    corr_matrix = noesy_parser.get_correlation_matrix(noe_peaks, labels)

    # Find connected methyls
    connected_pairs = np.sum(corr_matrix > 0, axis=0)

    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    mask_corr = corr_matrix == 0
    sns.heatmap(corr_matrix, mask=mask_corr, cmap='YlOrRd',
                square=True, linewidths=0.5, cbar_kws={'label': 'NOE Intensity'},
                ax=ax_corr)
    ax_corr.set_title('NOE Correlation Matrix')
    ax_corr.set_xlabel('Methyl Index')
    ax_corr.set_ylabel('Methyl Index')
    plt.tight_layout()

    plt.gca(), f"Average connections per methyl: {connected_pairs.mean():.1f}"
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 3. HMQC Peak List Parsing

    Parse 2D 1H-13C correlation spectra. These peaks provide chemical shift assignments for each methyl group.
    """)
    return


@app.cell
def _(DATA_DIR, HMQCParser, mo):
    # Parse HMQC peaks
    hmqc_file = DATA_DIR / "example_hmqc.txt"
    hmqc_parser = HMQCParser(str(hmqc_file))
    hmqc_format = hmqc_parser.detect_format()
    hmqc_peaks = hmqc_parser.parse()
    chemical_shifts = hmqc_parser.get_chemical_shifts(hmqc_peaks)

    mo.md(
        f"""
        ### HMQC Peak List

        - **File**: `{hmqc_file.name}`
        - **Detected format**: {hmqc_format}
        - **Peaks found**: {len(hmqc_peaks)}
        - **Assigned peaks**: {len(chemical_shifts)}
        """
    )
    return (hmqc_peaks,)


@app.cell
def _(hmqc_peaks, pd):
    # Create DataFrame of HMQC peaks
    hmqc_data = []
    for peak_hmqc in hmqc_peaks:
        hmqc_data.append({
            'Index': peak_hmqc.index,
            '1H (ppm)': peak_hmqc.h_shift,
            '13C (ppm)': peak_hmqc.c_shift,
            'Intensity': peak_hmqc.intensity,
            'Assignment': peak_hmqc.assignment
        })

    df_hmqc = pd.DataFrame(hmqc_data)
    df_hmqc.head(10)
    return (df_hmqc,)


@app.cell
def _(df_hmqc, plt):
    # Visualize HMQC spectrum
    fig_hmqc, (ax_hmqc_spectrum, ax_hmqc_hist) = plt.subplots(1, 2, figsize=(12, 4))

    # 2D HMQC spectrum
    scatter_hmqc = ax_hmqc_spectrum.scatter(df_hmqc['1H (ppm)'], df_hmqc['13C (ppm)'],
                         s=df_hmqc['Intensity']/800, alpha=0.6,
                         c=df_hmqc['Intensity'], cmap='viridis')
    ax_hmqc_spectrum.set_xlabel('1H Chemical Shift (ppm)')
    ax_hmqc_spectrum.set_ylabel('13C Chemical Shift (ppm)')
    ax_hmqc_spectrum.set_title('HMQC Spectrum (size = intensity)')
    ax_hmqc_spectrum.invert_xaxis()
    ax_hmqc_spectrum.grid(True, alpha=0.3)
    plt.colorbar(scatter_hmqc, ax=ax_hmqc_spectrum, label='Intensity')

    # Intensity distribution
    ax_hmqc_hist.hist(df_hmqc['Intensity'], bins=15, color='green', edgecolor='black', alpha=0.7)
    ax_hmqc_hist.set_xlabel('Intensity')
    ax_hmqc_hist.set_ylabel('Count')
    ax_hmqc_hist.set_title('HMQC Peak Intensity Distribution')
    ax_hmqc_hist.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## 4. Integration: Cross-Validation

    Let's cross-validate the assignments across different data sources.
    """)
    return


@app.cell
def _(df_hmqc, df_noe, mo):
    # Find common assignments
    noe_assignments = set()
    for _, row in df_noe.iterrows():
        if row['From']:
            noe_assignments.add(row['From'])
        if row['To']:
            noe_assignments.add(row['To'])

    hmqc_assignments = set(df_hmqc['Assignment'].dropna())
    common_assignments = noe_assignments & hmqc_assignments

    mo.md(
        f"""
        ### Cross-Validation Results

        - **NOESY assignments**: {len(noe_assignments)}
        - **HMQC assignments**: {len(hmqc_assignments)}
        - **Common assignments**: {len(common_assignments)}
        - **Agreement**: {100 * len(common_assignments) / max(len(noe_assignments), len(hmqc_assignments)):.1f}%
        """
    )
    return (common_assignments,)


@app.cell
def _(common_assignments):
    # Show sample of common assignments
    list(sorted(common_assignments))[:10]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    This notebook demonstrated:

    1. **PDB Parsing**: Extracted 43 methyl groups from ubiquitin structure
    2. **NOESY Parsing**: Loaded NOE connectivity data with automatic format detection
    3. **HMQC Parsing**: Loaded 2D correlation data with chemical shifts
    4. **Integration**: Cross-validated assignments across data sources

    ### Next Steps

    - Phase 2: Graph construction from parsed data
    - Phase 3: Graph matching algorithms
    - Phase 4: Assignment output and visualization

    ### Key Features

    - ✅ Multi-format support (XEASY, Sparky, CSV, NMRPipe)
    - ✅ Automatic format detection
    - ✅ Comprehensive data validation
    - ✅ Distance and correlation matrix calculations
    - ✅ Chemical shift matching with tolerances

    See the [GitHub repository](https://github.com/yourusername/methyl_match) for more information.
    """)
    return


if __name__ == "__main__":
    app.run()
