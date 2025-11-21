# Project GEMINI: Methyl Assignment using Deep Learning

This document outlines the development plan and progress for a project to create a methyl assignment algorithm using a deep learning approach.

## 1. Project Goal

The primary goal is to develop a robust and accurate algorithm for the automatic assignment of methyl groups in large proteins using NMR data. This project leverages deep learning for graph matching to improve upon existing methods.

## 2. Tech Stack

- **Language:** Python 3.13
- **Environment/Package Manager:** `uv`
- **Coding Style:** Google Python Style Guide
- **Core Libraries:** `torch`, `torch_geometric`, `networkx`, `biopython`

## 3. Project Structure

A modular structure has been adopted to ensure separation of concerns:

```
.
├── .gitignore
├── GEMINI.md
├── README.md
├── requirements.txt
├── data/
│   ├── train/
│   ├── test/
│   └── inference/
├── src/
│   ├── __init__.py
│   ├── data_preprocessing/
│   │   ├── loader.py
│   │   └── pdb_parser.py
│   ├── graph_matching/
│   │   ├── model.py
│   │   └── loss.py
│   └── main.py
└── tests/
    ├── __init__.py
    ├── test_data_preprocessing.py
    └── test_graph_matching.py
```

## 4. Deep Learning Approach

The project uses a Graph Matching Network (GMN) architecture for learning the similarity between an experimental NMR graph and a ground-truth graph derived from a PDB file.

The model consists of:
- A GNN encoder to learn node embeddings for each graph.
- A final classifier that takes the aggregated graph embeddings and predicts a similarity score.

For the final peak assignment, a nearest-neighbor search is performed in the learned embedding space to find the best match for each experimental peak among the ground-truth peaks.

## 5. Progress and Next Steps

- [x] Initial project setup and research.
- [x] Create project structure and initial files.
- [x] Implement data loading and preprocessing modules.
- [x] Implement ground truth graph generation from PDB files.
- [x] Refactor model to a Graph Matching Network (GMN) architecture.
- [x] Implement a suitable loss function (BCE loss on similarity score).
- [x] Implement a full training and evaluation pipeline.
- [x] Establish a synthetic ground truth mapping for initial training.
- [x] Implement a nearest-neighbor assignment method.
- [x] Run inference on a new sample and save assignments to a file.
- [ ] Refine ground truth mapping with more robust label matching.
- [ ] Add more diverse negative samples for training.
- [ ] Implement comprehensive evaluation metrics (e.g., precision, recall).
- [ ] Expand unit tests for the new architecture and training logic.
- [ ] Clean up temporary files and code.

## 6. Alternative Tools and Considerations

- **Maximum Common Subgraph (MCS)**: While we are focusing on a DL approach, classical MCS algorithms could be considered as a baseline or for hybrid approaches.
- **Other GNN architectures**: Other advanced GNN architectures could be explored for performance improvement.
