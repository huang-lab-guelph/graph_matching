#!/usr/bin/env python
"""Generate synthetic NMR data from PDB structures for testing."""

import argparse
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nmr_graph_matching.data.pdb_parser import PDBParser
from nmr_graph_matching.graphs.methyl_network import MethylNetworkBuilder


def generate_synthetic_hmqc(methyl_groups, methyl_names, output_file):
    """Generate synthetic HMQC peak list from methyl groups.

    Args:
        methyl_groups: List of MethylGroup objects
        methyl_names: List of methyl identifiers
        output_file: Path to output file
    """
    num_methyls = len(methyl_groups)

    # Generate chemical shifts based on residue type
    # Typical ranges for methyl resonances
    h_shifts = []
    c_shifts = []

    for methyl in methyl_groups:
        # Add residue-type specific shifts with noise
        if methyl.residue_name == 'LEU':
            h_shifts.append(np.random.normal(0.9, 0.1))
            c_shifts.append(np.random.normal(24.0, 2.0))
        elif methyl.residue_name == 'VAL':
            h_shifts.append(np.random.normal(0.95, 0.1))
            c_shifts.append(np.random.normal(21.0, 2.0))
        elif methyl.residue_name == 'ILE':
            if 'CD1' in methyl.atom_name:
                h_shifts.append(np.random.normal(0.85, 0.1))
                c_shifts.append(np.random.normal(13.0, 1.5))
            else:  # CG2
                h_shifts.append(np.random.normal(0.9, 0.1))
                c_shifts.append(np.random.normal(17.0, 1.5))
        elif methyl.residue_name == 'ALA':
            h_shifts.append(np.random.normal(1.4, 0.1))
            c_shifts.append(np.random.normal(18.0, 1.5))
        elif methyl.residue_name == 'THR':
            h_shifts.append(np.random.normal(1.2, 0.1))
            c_shifts.append(np.random.normal(21.0, 1.5))
        elif methyl.residue_name == 'MET':
            h_shifts.append(np.random.normal(2.1, 0.1))
            c_shifts.append(np.random.normal(17.0, 1.5))
        else:
            h_shifts.append(np.random.uniform(0.5, 1.5))
            c_shifts.append(np.random.uniform(15.0, 25.0))

    # Generate intensities
    intensities = np.random.uniform(5e4, 2e5, num_methyls)

    # Write XEASY format
    with open(output_file, 'w') as f:
        f.write("# Number of dimensions 2\n")
        f.write("# FORMAT xeasy2D\n")
        f.write("# Synthetic HMQC peak list generated from PDB structure\n")
        f.write("# Peak_ID  H_shift  C_shift  Color  Type  Volume  Vol_Err  Assignment\n")

        for i in range(num_methyls):
            f.write(f"{i+1:<8} {h_shifts[i]:<8.3f} {c_shifts[i]:<8.3f} ")
            f.write(f"1        N     {intensities[i]:.2e}  1000     ")
            f.write(f"{methyl_names[i]}\n")

    print(f"  Generated {num_methyls} HMQC peaks")


def generate_synthetic_noesy(methyl_groups, methyl_names, h_shifts, c_shifts, output_file, distance_cutoff=10.0):
    """Generate synthetic NOESY peak list based on methyl-methyl distances.

    Args:
        methyl_groups: List of MethylGroup objects
        methyl_names: List of methyl identifiers
        h_shifts: List of H chemical shifts (from HMQC)
        c_shifts: List of C chemical shifts (from HMQC)
        output_file: Path to output file
        distance_cutoff: Maximum distance for observable NOE (Angstroms)
    """
    num_methyls = len(methyl_groups)

    # Compute distance matrix
    coords = np.array([m.coordinates for m in methyl_groups])
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))

    # Write XEASY 3D format
    crosspeaks = []

    with open(output_file, 'w') as f:
        f.write("# Number of dimensions 3\n")
        f.write("# FORMAT xeasy3D\n")
        f.write("# Synthetic NOESY peak list generated from PDB structure\n")
        f.write(f"# Distance cutoff: {distance_cutoff} Angstroms\n")
        f.write("# Peak_ID  C1_shift  H_shift  C2_shift  Color  Type  Volume  Vol_Err\n")

        peak_id = 1
        for i in range(num_methyls):
            for j in range(i + 1, num_methyls):
                distance = distances[i, j]

                if distance < distance_cutoff:
                    # NOE intensity inversely proportional to r^6
                    # Adding noise
                    base_intensity = 1e5 / (distance ** 3)  # Simplified
                    intensity = base_intensity * np.random.uniform(0.7, 1.3)

                    # Add some noise to make realistic
                    noise_factor = np.random.uniform(0.9, 1.1)

                    f.write(f"{peak_id:<8} {c_shifts[i]:<9.3f} {h_shifts[i]:<9.3f} ")
                    f.write(f"{c_shifts[j]:<9.3f} 1        N     ")
                    f.write(f"{intensity * noise_factor:.2e}  1000\n")

                    crosspeaks.append((i, j, distance, intensity))
                    peak_id += 1

    print(f"  Generated {len(crosspeaks)} NOESY cross-peaks")
    print(f"  Distance range: {min(cp[2] for cp in crosspeaks):.1f} - {max(cp[2] for cp in crosspeaks):.1f} Å")


def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic NMR data from PDB structure'
    )
    parser.add_argument('--pdb', type=str, required=True,
                       help='Path to input PDB file')
    parser.add_argument('--output-dir', type=str, default='data/raw/synthetic',
                       help='Output directory for generated files')
    parser.add_argument('--distance-cutoff', type=float, default=10.0,
                       help='Distance cutoff for NOEs (Angstroms)')
    parser.add_argument('--prefix', type=str, default=None,
                       help='Prefix for output files (default: PDB basename)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine output prefix
    if args.prefix:
        prefix = args.prefix
    else:
        prefix = Path(args.pdb).stem

    print(f"\n{'='*60}")
    print(f"Generating Synthetic NMR Data")
    print(f"{'='*60}")
    print(f"Input PDB: {args.pdb}")
    print(f"Output directory: {output_dir}")
    print(f"Prefix: {prefix}")
    print()

    # Parse PDB
    print("Parsing PDB structure...")
    pdb_parser = PDBParser()
    methyl_groups = pdb_parser.parse(args.pdb)

    if len(methyl_groups) == 0:
        print("ERROR: No methyl groups found in PDB file!")
        print("Ensure the structure contains residues: LEU, VAL, ILE, ALA, THR, MET")
        return 1

    methyl_names = pdb_parser.get_full_names()

    print(f"  Found {len(methyl_groups)} methyl groups")
    summary = pdb_parser.get_summary()
    for res_type, count in sorted(summary.items()):
        if res_type != 'TOTAL':
            print(f"    {res_type}: {count}")
    print()

    # Generate HMQC
    hmqc_file = output_dir / f"{prefix}_hmqc.txt"
    print(f"Generating HMQC peak list: {hmqc_file}")
    generate_synthetic_hmqc(methyl_groups, methyl_names, hmqc_file)
    print()

    # Read back H and C shifts for NOESY generation
    h_shifts = []
    c_shifts = []
    with open(hmqc_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    h_shifts.append(float(parts[1]))
                    c_shifts.append(float(parts[2]))
                except:
                    continue

    # Generate NOESY
    noesy_file = output_dir / f"{prefix}_noesy.txt"
    print(f"Generating NOESY peak list: {noesy_file}")
    generate_synthetic_noesy(
        methyl_groups,
        methyl_names,
        h_shifts,
        c_shifts,
        noesy_file,
        distance_cutoff=args.distance_cutoff
    )
    print()

    # Copy PDB to output directory
    import shutil
    pdb_output = output_dir / f"{prefix}.pdb"
    shutil.copy(args.pdb, pdb_output)
    print(f"Copied PDB to: {pdb_output}")
    print()

    print(f"{'='*60}")
    print("✓ Synthetic data generation complete!")
    print(f"{'='*60}")
    print("\nGenerated files:")
    print(f"  - {pdb_output}")
    print(f"  - {hmqc_file}")
    print(f"  - {noesy_file}")
    print("\nNext steps:")
    print(f"  1. Review generated files in {output_dir}/")
    print("  2. Use for testing/training:")
    print(f"     uv run python scripts/train_model.py --data-dir {output_dir}")
    print()

    return 0


if __name__ == '__main__':
    sys.exit(main())
