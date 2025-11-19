# NMR Graph Matching for Methyl Assignment

Deep learning-based graph matching for automated methyl assignment in NMR spectroscopy using Graph Neural Networks.

## Overview

Matches experimental NMR peaks with protein structure methyls by learning correspondences between two graphs:
- **Graph A**: NMR peaks (nodes) + NOE correlations (edges)
- **Graph B**: Structural methyl positions (nodes) + spatial distances (edges)

**Model**: Deep Graph Matching Consensus (DGMC) with GNN encoders, consensus refinement, and Sinkhorn matching.

## Quick Start

### Installation

```bash
# Install with uv
uv sync

# Verify installation
uv run python -c "import nmr_graph_matching; print('✓ Ready')"
```

### Get Training Data (Choose One)

**Option 1: MAGIC Test Data (2 min)**
```bash
./scripts/download_test_data.sh
```

**Option 2: Synthetic Data (5 min)**
```bash
# Download a PDB
curl https://files.rcsb.org/download/1UBQ.pdb -o data/raw/1UBQ.pdb

# Generate synthetic NMR data
uv run python scripts/generate_synthetic_data.py \
    --pdb data/raw/1UBQ.pdb \
    --output-dir data/raw
```

**Option 3: Real Data from BMRB**
- Visit https://bmrb.io/
- Search "methyl TROSY"
- Download peak lists

### Train

```bash
uv run python scripts/train_model.py \
    --config configs/default_config.yaml \
    --data-dir data/raw \
    --epochs 100
```

### Predict

```bash
uv run python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --pdb protein.pdb \
    --hmqc hmqc.txt \
    --noesy noesy.txt \
    --output results.txt \
    --format both \
    --pymol
```

## Data Requirements

For each protein, provide 3 files:

| File | Format | Contains |
|------|--------|----------|
| `protein.pdb` | PDB | 3D structure with methyls (LEU, VAL, ILE, ALA, THR, MET) |
| `protein_hmqc.txt` | XEASY/NMRPipe/Sparky/CSV | 2D ¹H-¹³C correlation peaks |
| `protein_noesy.txt` | XEASY/NMRPipe/CSV | 3D NOE cross-peaks |

**Naming convention**: `<name>.pdb`, `<name>_hmqc.txt`, `<name>_noesy.txt`

**Minimum dataset**: 5-10 proteins with known assignments for training

## Project Structure

```
src/nmr_graph_matching/
├── data/           # PDB & NMR parsers
├── graphs/         # Graph construction
├── models/         # DGMC neural network
├── training/       # Training pipeline
└── utils/          # Output formatting

scripts/
├── train_model.py              # Train models
├── predict.py                  # Make predictions
├── generate_synthetic_data.py  # Generate test data
└── download_test_data.sh       # Get MAGIC data

configs/
└── default_config.yaml         # Model & training config
```

## Python API

```python
from nmr_graph_matching import (
    PDBParser, HMQCParser, NOESYParser,
    MethylNetworkBuilder, PeakNetworkBuilder,
    DGMCModel, OutputFormatter
)

# Parse inputs
pdb_parser = PDBParser()
methyls = pdb_parser.parse("protein.pdb")

hmqc_parser = HMQCParser()
peaks = hmqc_parser.parse("hmqc.txt")

noesy_parser = NOESYParser()
noesy_parser.parse("noesy.txt")
crosspeaks = noesy_parser.match_with_hmqc(peaks)

# Build graphs
methyl_graph = MethylNetworkBuilder(distance_cutoff=10.0).build(methyls)
peak_graph = PeakNetworkBuilder().build(peaks, crosspeaks)

# Predict
model = DGMCModel(peak_feature_dim=4, methyl_feature_dim=11)
model.load_state_dict(torch.load("model.pt")['model_state_dict'])
model.eval()

with torch.no_grad():
    matching_matrix, _ = model(peak_graph, methyl_graph)

# Format results
formatter = OutputFormatter()
results = formatter.format_assignments(
    matching_matrix, peak_graph.peak_ids,
    peak_shifts, methyl_graph.methyl_names
)
formatter.to_csv(results, "assignments.csv")
```

## Configuration

Edit `configs/default_config.yaml`:

```yaml
model:
  embedding_dim: 64        # Node embedding size
  hidden_dim: 128          # Hidden layer size
  gnn_type: "gcn"          # 'gcn' or 'gat'
  num_encoder_layers: 3    # GNN depth

data:
  distance_cutoff: 10.0    # Max distance for methyl edges (Å)
  shift_normalization: "standard"

training:
  num_epochs: 100
  learning_rate: 0.001
  batch_size: 1

loss_weights:
  matching_weight: 1.0     # Cross-entropy
  distance_weight: 0.1     # NOE-distance consistency
  permutation_weight: 0.01 # Doubly-stochastic
```

## Supported File Formats

**HMQC/NOESY**: XEASY, NMRPipe, Sparky, CSV (auto-detected)

**XEASY Example**:
```
# Number of dimensions 2
Peak_ID  Shift1  Shift2  Volume
1        0.845   24.3    1.23e+05
```

**CSV Example**:
```csv
peak_id,H_shift,C_shift,intensity
1,0.845,24.3,123000
```

## Output Format

```
================================================================================
NMR Methyl Assignment Results
================================================================================

Peak_ID  H_shift    C_shift    Assignment      Confidence   NOE_Compl
--------------------------------------------------------------------------------
1        0.845      24.300     L15CD1          0.950        0.870
           →  L42CD2          0.230
2        1.234      19.500     V23CG1          0.820        0.750
...

Total assignments: 45
High-confidence (≥0.5): 38
================================================================================
```

## Data Sources

1. **MAGIC**: `./scripts/download_test_data.sh` - Real test data
2. **PDB**: https://www.rcsb.org/ - Protein structures
3. **BMRB**: https://bmrb.io/ - Experimental NMR data
4. **AlphaFold**: https://alphafold.ebi.ac.uk/ - Predicted structures
5. **Synthetic**: Generate from any PDB with `generate_synthetic_data.py`

## Quick Test Dataset

```bash
# Download 3 proteins
for pdb in 1UBQ 2GB1 1CLL; do
    curl "https://files.rcsb.org/download/${pdb}.pdb" -o "data/raw/${pdb}.pdb"
    uv run python scripts/generate_synthetic_data.py \
        --pdb "data/raw/${pdb}.pdb" \
        --output-dir data/raw \
        --prefix "${pdb}"
done

# Train
uv run python scripts/train_model.py \
    --config configs/default_config.yaml \
    --data-dir data/raw \
    --epochs 50
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No methyl groups found | PDB missing LEU/VAL/ILE residues or incorrect atom naming |
| No NOESY cross-peaks | Increase tolerance: `NOESYParser(tolerance=0.1)` |
| CUDA out of memory | Use CPU `--device cpu` or reduce `embedding_dim`/`hidden_dim` |
| Poor accuracy | Need 5-10+ proteins for training, adjust `distance_weight` |
| File format error | Specify format: `parser.parse(file, format='xeasy')` |

## Testing

```bash
# Run unit tests
uv run pytest tests/ -v

# Test data loading
uv run python -c "
from nmr_graph_matching import PDBParser
parser = PDBParser()
methyls = parser.parse('data/raw/1UBQ.pdb')
print(f'Found {len(methyls)} methyls')
"
```

## Model Architecture

1. **Graph Encoders**: GCN/GAT layers encode both graphs
2. **Cross-Graph Similarity**: Compute initial matching scores
3. **Consensus Refinement**: Iterative GNN using neighborhood consistency
4. **Sinkhorn Normalization**: Convert to doubly-stochastic matrix
5. **Assignment**: Hungarian algorithm for discrete matching

**Loss Functions**:
- Matching loss (cross-entropy with ground truth)
- Distance consistency (penalize NOE-distance violations)
- Permutation loss (encourage doubly-stochastic)

## References

- **MAGIC Algorithm**: https://pmc.ncbi.nlm.nih.gov/articles/PMC5764113/
- **DGMC Paper**: https://arxiv.org/abs/2001.09621 (ICLR 2020)
- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/

## Citation

```bibtex
@software{nmr_graph_matching,
  title = {NMR Graph Matching for Methyl Assignment},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/graph_matching}
}
```

## License

This project is provided as-is for research and educational purposes.

## Additional Documentation

- **CLAUDE.md**: Context for AI assistants working on this repo
- **notebooks/example_usage.ipynb**: Interactive tutorial
- See `scripts/` for additional utilities
