# Claude Context: NMR Graph Matching Project

This document provides AI assistants with essential context about this repository.

## Project Purpose

Deep learning system for automated methyl assignment in NMR spectroscopy using graph neural networks. Matches experimental NMR peak networks with structural methyl networks.

## Architecture Overview

### Core Problem
Match two graphs:
- **Graph A (Peak Network)**: NMR peaks (nodes) + NOE correlations (edges)
- **Graph B (Methyl Network)**: Methyl positions (nodes) + spatial distances (edges)

### Technology Stack
- **Deep Learning**: PyTorch + PyTorch Geometric
- **Model**: DGMC (Deep Graph Matching Consensus) with GCN/GAT encoders
- **Data**: BioPython (PDB), nmrglue (NMR), NumPy/SciPy
- **Package Manager**: uv

## Code Organization

```
src/nmr_graph_matching/
├── data/           # PDB & NMR parsers (pdb_parser.py, nmr_parser.py)
├── graphs/         # Graph builders (methyl_network.py, peak_network.py)
├── models/         # DGMC model (dgmc.py, matching.py)
├── training/       # Training (dataset.py, losses.py, trainer.py)
└── utils/          # Output (output_formatter.py)

scripts/
├── train_model.py              # Training script
├── predict.py                  # Inference script
├── generate_synthetic_data.py  # Synthetic data generation
└── download_test_data.sh       # Get MAGIC test data
```

## Key Components

### 1. Data Parsing
- **PDBParser**: Extracts methyl groups (LEU, VAL, ILE, ALA, THR, MET) from PDB structures
- **HMQCParser**: Reads 2D ¹H-¹³C correlation peak lists (XEASY, NMRPipe, Sparky, CSV)
- **NOESYParser**: Reads 3D NOE peak lists, matches cross-peaks with HMQC

### 2. Graph Construction
- **MethylNetworkBuilder**: Creates PyG Data with node features [x,y,z, residue_type_one_hot, res_num], edge features [distance, weight]
- **PeakNetworkBuilder**: Creates PyG Data with node features [H_shift, C_shift, intensity, confidence], edge features [NOE_intensity, confidence]

### 3. Model
- **DGMCModel**: Graph encoders → similarity matrix → consensus refinement → Sinkhorn → assignments
- **Loss Functions**: Matching (cross-entropy), Distance consistency (NOE-distance correlation), Permutation (doubly-stochastic)

### 4. Training
- **NMRDataset**: Loads triplets (PDB, HMQC, NOESY), constructs graphs, extracts ground truth
- **Trainer**: Training loop, validation, checkpointing, best model selection

### 5. Output
- **OutputFormatter**: Formats results as text/CSV/PyMOL, computes confidence scores and NOE completeness

## File Requirements

Each protein needs 3 files:
- `protein.pdb` - 3D structure
- `protein_hmqc.txt` - 2D methyl peaks
- `protein_noesy.txt` - 3D NOE cross-peaks

Naming convention: `<name>.pdb`, `<name>_hmqc.txt`, `<name>_noesy.txt`

## Common Tasks

### Training
```bash
uv run python scripts/train_model.py \
    --config configs/default_config.yaml \
    --data-dir data/raw \
    --epochs 100
```

### Prediction
```bash
uv run python scripts/predict.py \
    --checkpoint checkpoints/best_model.pt \
    --pdb protein.pdb \
    --hmqc hmqc.txt \
    --noesy noesy.txt \
    --output results.txt
```

### Generate Synthetic Data
```bash
uv run python scripts/generate_synthetic_data.py \
    --pdb structure.pdb \
    --output-dir data/raw
```

## Configuration

`configs/default_config.yaml` contains:
- Model architecture (embedding_dim, hidden_dim, gnn_type)
- Data processing (distance_cutoff, shift_normalization)
- Training (learning_rate, num_epochs, batch_size)
- Loss weights (matching_weight, distance_weight, permutation_weight)

## Key Classes & Methods

### PDBParser
- `parse(pdb_file)` → List[MethylGroup]
- `get_summary()` → Dict[residue_type, count]
- `compute_distance_matrix()` → np.ndarray

### HMQCParser / NOESYParser
- `parse(filepath, format=None)` → List[Peak]
- `match_with_hmqc(hmqc_peaks)` → List[NOECrosspeaks]

### MethylNetworkBuilder / PeakNetworkBuilder
- `build(...)` → Data (PyTorch Geometric)
- `get_adjacency_matrix(data)` → np.ndarray

### DGMCModel
- `forward(peak_graph, methyl_graph)` → matching_matrix, embeddings
- `predict_assignments(matching_matrix, method='hungarian')` → assignments, confidences

### Trainer
- `train_epoch()` → Dict[loss_components]
- `validate()` → Dict[metrics]
- `save_checkpoint(filename)`

## Data Sources

1. **MAGIC repository**: `./scripts/download_test_data.sh`
2. **Synthetic from PDB**: `scripts/generate_synthetic_data.py`
3. **BMRB database**: https://bmrb.io/ (real experimental data)
4. **PDB structures**: https://www.rcsb.org/
5. **AlphaFold**: https://alphafold.ebi.ac.uk/

## Testing

Run tests: `uv run pytest tests/ -v`

Basic import test:
```python
from nmr_graph_matching import (
    PDBParser, HMQCParser, NOESYParser,
    MethylNetworkBuilder, PeakNetworkBuilder,
    DGMCModel, Trainer, OutputFormatter
)
```

## Common Issues

1. **No methyl groups found**: PDB missing LEU/VAL/ILE residues or incorrect atom names
2. **No NOESY cross-peaks**: Increase tolerance (default 0.05 ppm) or check format
3. **CUDA OOM**: Reduce embedding_dim/hidden_dim or use CPU
4. **Poor accuracy**: Need more training data (5-10 proteins minimum)

## Development Notes

- **Batch size**: Typically 1 for graph matching (different graph sizes)
- **Normalization**: Chemical shifts use standard normalization by default
- **Distance cutoff**: 10Å default for methyl networks (NOE observable range)
- **Loss weights**: matching=1.0, distance=0.1, permutation=0.01 (defaults)

## Scientific Background

Based on:
- **MAGIC algorithm**: https://pmc.ncbi.nlm.nih.gov/articles/PMC5764113/
- **DGMC paper**: https://arxiv.org/abs/2001.09621

Methyl-TROSY NMR is used for large proteins. Assignment challenge: match observed peaks to specific methyls in structure using NOE distance constraints.

## Important Conventions

- Peak IDs are 1-indexed in files but 0-indexed in code
- Chemical shifts: H (0.5-2.0 ppm), C (15-30 ppm) for methyls
- Ground truth in training: assignment strings like "L15CD1" in HMQC files
- Residue naming: One-letter + number + atom (e.g., "L15CD1" = Leu 15 CD1)

## When Modifying

- **Add GNN layer**: Edit `models/dgmc.py` GraphEncoder class
- **New loss**: Add to `training/losses.py`, integrate in CombinedLoss
- **New file format**: Add parser method to `data/nmr_parser.py`
- **Visualization**: Methods in graph builders use matplotlib/networkx
- **Tests**: Add to `tests/test_basic.py`

## Dependencies Notes

- PyTorch Geometric requires specific PyTorch version compatibility
- nmrglue may need specific NumPy version
- Use `uv sync` to handle all compatibility automatically
