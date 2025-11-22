"""
YME1L Part 1: Reading and Parsing Data Files

This notebook demonstrates reading and parsing YME1L protein data:
- PDB structure file
- HMQC peak list
- NOESY peak list
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
    # YME1L Part 1: Reading Data Files

    This notebook loads and validates the YME1L protein data files:
    1. **PDB structure** - 3D coordinates of methyl groups
    2. **HMQC peak list** - 1H-13C chemical shifts
    3. **NOESY peak list** - 13C-13C-1H methyl-methyl NOE correlations

    We'll parse each file and display summary statistics.
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

    # Add project to path
    project_root = Path.cwd().parent.parent
    sys.path.insert(0, str(project_root / "src"))

    from methyl_match.reading import PDBParser, NOESYParser, HMQCParser

    # Set style
    sns.set_style("whitegrid")
    return HMQCParser, NOESYParser, PDBParser, np, pd, plt, project_root


@app.cell
def _(mo):
    mo.md("""
    ## 1. PDB Structure File
    """)
    return


@app.cell
def _(PDBParser, project_root):
    # Define data directory
    data_dir = project_root / "data" / "yme1l"
    pdb_file = data_dir / "yme1l.pdb"

    # Parse PDB
    print(f"Reading: {pdb_file}")
    pdb_parser = PDBParser(str(pdb_file))
    methyls_yme1l = pdb_parser.extract_methyls()

    print(f"\nExtracted {len(methyls_yme1l)} methyl groups")
    return data_dir, methyls_yme1l


@app.cell
def _(methyls_yme1l, pd):
    # Create summary table
    methyl_summary = []
    for m in methyls_yme1l[:10]:  # Show first 10
        methyl_summary.append({
            'Label': m.label,
            'Residue': m.residue_name,
            'Number': m.residue_number,
            'Atom': m.atom_name,
            'X': f"{m.coordinates[0]:.2f}",
            'Y': f"{m.coordinates[1]:.2f}",
            'Z': f"{m.coordinates[2]:.2f}",
        })

    methyl_df = pd.DataFrame(methyl_summary)
    methyl_df
    return


@app.cell
def _(methyls_yme1l, mo):
    # Count residue types
    from collections import Counter
    residue_counts = Counter(m.residue_name for m in methyls_yme1l)

    mo.md(f"""
    **Methyl Group Statistics:**
    - Total methyls: {len(methyls_yme1l)}
    - Residue type distribution:
      - LEU: {residue_counts.get('LEU', 0)}
      - VAL: {residue_counts.get('VAL', 0)}
      - ILE: {residue_counts.get('ILE', 0)}
      - ALA: {residue_counts.get('ALA', 0)}
      - THR: {residue_counts.get('THR', 0)}
      - MET: {residue_counts.get('MET', 0)}
    """)
    return (residue_counts,)


@app.cell
def _(methyls_yme1l, plt, residue_counts):
    # Visualize residue distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    residues = list(residue_counts.keys())
    counts = list(residue_counts.values())

    ax.bar(residues, counts, color='steelblue', edgecolor='navy')
    ax.set_xlabel('Residue Type')
    ax.set_ylabel('Count')
    ax.set_title(f'YME1L Methyl Groups by Residue Type (Total: {len(methyls_yme1l)})')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. HMQC Peak List
    """)
    return


@app.cell
def _(HMQCParser, data_dir):
    # Parse HMQC
    hmqc_file = data_dir / "hmqc.list"

    print(f"Reading: {hmqc_file}")
    hmqc_parser = HMQCParser(str(hmqc_file))

    # Detect format
    detected_format = hmqc_parser.detect_format()
    print(f"Detected format: {detected_format}")

    # Parse peaks
    hmqc_peaks_yme1l = hmqc_parser.parse()
    print(f"\nParsed {len(hmqc_peaks_yme1l)} HMQC peaks")
    return (hmqc_peaks_yme1l,)


@app.cell
def _(hmqc_peaks_yme1l, pd):
    # Show first 10 peaks
    hmqc_summary = []
    for i, peak in enumerate(hmqc_peaks_yme1l[:10]):
        hmqc_summary.append({
            'Index': peak.index,
            'Assignment': peak.assignment or 'Unassigned',
            '1H (ppm)': f"{peak.h_shift:.3f}",
            '13C (ppm)': f"{peak.c_shift:.3f}",
            'Intensity': f"{peak.intensity:.2f}" if peak.intensity else "N/A",
        })

    hmqc_df = pd.DataFrame(hmqc_summary)
    hmqc_df
    return


@app.cell
def _(hmqc_peaks_yme1l, np):
    # Get chemical shift ranges
    h_shifts_hmqc = np.array([p.h_shift for p in hmqc_peaks_yme1l])
    c_shifts_hmqc = np.array([p.c_shift for p in hmqc_peaks_yme1l])
    return c_shifts_hmqc, h_shifts_hmqc


@app.cell
def _(c_shifts_hmqc, h_shifts_hmqc, mo):
    mo.md(f"""
    **HMQC Peak Statistics:**
    - Total peaks: {len(h_shifts_hmqc)}
    - 1H range: {h_shifts_hmqc.min():.2f} - {h_shifts_hmqc.max():.2f} ppm
    - 13C range: {c_shifts_hmqc.min():.2f} - {c_shifts_hmqc.max():.2f} ppm
    - Mean 1H: {h_shifts_hmqc.mean():.2f} ppm
    - Mean 13C: {c_shifts_hmqc.mean():.2f} ppm
    """)
    return


@app.cell
def _(c_shifts_hmqc, h_shifts_hmqc, plt):
    # Visualize HMQC spectrum
    fig_hmqc, ax_hmqc = plt.subplots(figsize=(10, 8))

    ax_hmqc.scatter(h_shifts_hmqc, c_shifts_hmqc, c='blue', s=50, alpha=0.6, edgecolors='black')
    ax_hmqc.set_xlabel('1H Chemical Shift (ppm)', fontsize=12)
    ax_hmqc.set_ylabel('13C Chemical Shift (ppm)', fontsize=12)
    ax_hmqc.set_title('YME1L HMQC Spectrum', fontsize=14, fontweight='bold')
    ax_hmqc.invert_xaxis()  # NMR convention
    ax_hmqc.invert_yaxis()
    ax_hmqc.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 3. NOESY Peak List
    """)
    return


@app.cell
def _(NOESYParser, data_dir):
    # Parse NOESY
    noesy_file = data_dir / "noesy.list"

    print(f"Reading: {noesy_file}")
    noesy_parser = NOESYParser(str(noesy_file))

    # Detect format
    detected_format_noesy = noesy_parser.detect_format()
    print(f"Detected format: {detected_format_noesy}")

    # Parse peaks
    noesy_peaks_yme1l = noesy_parser.parse()
    print(f"\nParsed {len(noesy_peaks_yme1l)} NOESY peaks")
    return (noesy_peaks_yme1l,)


@app.cell
def _(noesy_peaks_yme1l, pd):
    # Show first 10 NOESY peaks
    noesy_summary = []
    for noesy_idx, noesy_peak in enumerate(noesy_peaks_yme1l[:10]):
        noesy_summary.append({
            'Index': noesy_peak.index,
            'Assignment': noesy_peak.assignment1 or 'Unassigned',
            'w1 13C (ppm)': f"{noesy_peak.w1:.3f}",
            'w2 13C (ppm)': f"{noesy_peak.w2:.3f}",
            'w3 1H (ppm)': f"{noesy_peak.w3:.3f}",
            'Intensity': f"{noesy_peak.intensity:.2f}" if noesy_peak.intensity else "N/A",
        })

    noesy_df = pd.DataFrame(noesy_summary)
    noesy_df
    return


@app.cell
def _(noesy_peaks_yme1l, np):
    # Get intensity distribution
    noesy_intensities = np.array([p.intensity for p in noesy_peaks_yme1l if p.intensity])
    return (noesy_intensities,)


@app.cell
def _(mo, noesy_intensities, noesy_peaks_yme1l, np):
    mo.md(f"""
    **NOESY Peak Statistics:**
    - Total peaks: {len(noesy_peaks_yme1l)}
    - Peaks with intensity: {len(noesy_intensities)}
    - Intensity range: {noesy_intensities.min():.2e} - {noesy_intensities.max():.2e}
    - Mean intensity: {noesy_intensities.mean():.2e}
    - Median intensity: {np.median(noesy_intensities):.2e}
    """)
    return


@app.cell
def _(noesy_intensities, np, plt):
    # Visualize intensity distribution
    fig_int, ax_int = plt.subplots(figsize=(10, 6))

    ax_int.hist(np.log10(noesy_intensities), bins=50, color='coral', edgecolor='black', alpha=0.7)
    ax_int.set_xlabel('log10(Intensity)', fontsize=12)
    ax_int.set_ylabel('Count', fontsize=12)
    ax_int.set_title('YME1L NOESY Intensity Distribution', fontsize=14, fontweight='bold')
    ax_int.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    mo.md("""
    ## Summary

    ✅ **Successfully loaded all YME1L data files:**

    1. **PDB Structure** - Extracted methyl groups with 3D coordinates
    2. **HMQC Peaks** - 1H-13C chemical shifts for all methyls
    3. **NOESY Peaks** - 13C-13C-1H methyl-methyl NOE correlations

    All files parsed successfully! Ready to proceed to **Part 2: Graph Construction**.

    ### Key Observations:
    - YME1L has a good number of methyl groups distributed across different residue types
    - HMQC spectrum shows typical methyl chemical shift ranges
    - NOESY data contains intensity information for distance constraints
    - Data quality appears good for automated assignment

    ### Next Steps:
    Run `02_graphs_yme1l.py` to build structural and experimental networks.
    """)
    return


if __name__ == "__main__":
    app.run()
