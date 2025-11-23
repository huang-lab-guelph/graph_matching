"""
Debug script for SpectralMatcher issue.

This script runs a minimal test case to debug the type conversion error.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from methyl_match.reading import PDBParser, NOESYParser, HMQCParser
from methyl_match.preprocessing import MethylNetworkBuilder, PeakNetworkBuilder
from methyl_match.matching import SpectralMatcher

# Load data
data_dir = project_root / "data" / "yme1l"

print("Loading data...")
pdb_parser = PDBParser(str(data_dir / "yme1l.pdb"))
methyls = pdb_parser.extract_methyls()

hmqc_parser = HMQCParser(str(data_dir / "hmqc.list"))
hmqc_peaks = hmqc_parser.parse()

noesy_parser = NOESYParser(str(data_dir / "noesy.list"))
noesy_peaks = noesy_parser.parse()

# Build graphs
print("Building graphs...")
struct_builder = MethylNetworkBuilder(distance_cutoff=10.0)
network_struct = struct_builder.build_network(methyls)
G_struct = network_struct.graph

peak_builder = PeakNetworkBuilder(intensity_threshold=0.3)
network_exp = peak_builder.build_network(hmqc_peaks, noesy_peaks)
G_exp = network_exp.graph

n_exp = G_exp.number_of_nodes()
n_struct = G_struct.number_of_nodes()
print(f"Structural graph: {n_struct} nodes, {G_struct.number_of_edges()} edges")
print(f"Experimental graph: {n_exp} nodes, {G_exp.number_of_edges()} edges")

# Check types
print(f"\nDEBUG: n_exp type = {type(n_exp)}, value = {n_exp}")
print(f"DEBUG: n_struct type = {type(n_struct)}, value = {n_struct}")
print(f"DEBUG: max(n_exp, n_struct) type = {type(max(n_exp, n_struct))}")

# Calculate expected affinity matrix size
max_size = max(n_exp, n_struct)
affinity_size = max_size * max_size
print(f"\nWARNING: Affinity matrix will be ({affinity_size}, {affinity_size})")
print(f"That's {affinity_size**2 / 1e6:.1f} million elements!")

if affinity_size > 10000:
    print("\n⚠️  Matrix is too large! Let's use a smaller test case.")
    print("Skipping full test to avoid memory/time issues.")
    sys.exit(0)

# Try SpectralMatcher
print("\nTesting SpectralMatcher with method='ipfp'...")
matcher = SpectralMatcher(method='ipfp')

try:
    result = matcher.match(G_exp, G_struct)
    print(f"SUCCESS! Made {result.num_assignments} assignments")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
