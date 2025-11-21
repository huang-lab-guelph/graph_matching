# Project GEMINI: Methyl Assignment using Deep Learning

This document outlines the development plan and progress for a project to create a methyl assignment algorithm using a deep learning approach.

## 1. Project Goal

The primary goal is to develop a robust and accurate algorithm for the automatic assignment of methyl groups in large proteins using NMR data. This project will leverage deep learning for graph matching to improve upon existing methods.

## 2. Tech Stack

- **Language:** Python 3.13
- **Environment/Package Manager:** `uv`
- **Coding Style:** Google Python Style Guide
- **Testing:** `pytest`

## 3. Project Structure

A modular structure will be adopted to ensure separation of concerns.

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
│   │   ├── __init__.py
│   │   └── loader.py
│   ├── graph_matching/
│   │   ├── __init__.py
│   │   └── model.py
│   └── main.py
└── tests/
    ├── __init__.py
    ├── test_data_preprocessing.py
    └── test_graph_matching.py
```

- **`data/`**: Will contain the training, testing, and inference datasets.
- **`src/`**: The main source code of the project.
    - **`data_preprocessing/`**: Modules for loading and preprocessing the NMR data.
    - **`graph_matching/`**: The core deep learning model for graph matching.
    - **`main.py`**: The entry point for running the assignment process.
- **`tests/`**: Unit tests for the main components.

## 4. Deep Learning Approach

Based on initial research, we will investigate the use of Graph Neural Networks (GNNs) for the graph matching task. The following libraries and resources are promising:

- **`Pygmtools`**: A dedicated Python toolkit for graph matching with deep learning solvers. This is a strong candidate for our core library.
- **`Deep Graph Library (DGL)`** and **`PyTorch Geometric (PyG)`**: More general GNN libraries that can be used to build custom graph matching models.

The initial approach will be to represent the methyl groups and their NOE connections as graphs and use a GNN-based model to learn a similarity metric for matching.

## 5. Relevant Resources

- **MAGIC Algorithm (for reference)**:
    - [PMC Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC5764113/)
    - [GitHub Repo](https://github.com/NMRsoftware/MAGIC)
- **Potential DL Libraries**:
    - [Pygmtools](https://pygmtools.readthedocs.io/en/latest/)
    - [Deep Graph Library (DGL)](https://www.dgl.ai/)
    - [PyTorch Geometric (PyG)](https://pyg.org/)

## 6. Progress and Next Steps

- [x] Initial project setup and research.
- [x] Create project structure and initial files.
- [x] Implement data loading and preprocessing modules.
- [x] Implement real data loader and graph construction.
- [x] Implement the graph matching model using a chosen library (pygmtools classic solver with GNN encoder integrated).
- [x] Develop training and evaluation scripts (basic matching pipeline implemented).
- [x] Write unit tests.
- [x] Document the code and usage in `README.md`.
- [x] Implement ground truth graph generation from PDB files.

## 7. Alternative Tools and Considerations

- **Maximum Common Subgraph (MCS)**: While we are focusing on a DL approach, classical MCS algorithms could be considered as a baseline or for hybrid approaches.