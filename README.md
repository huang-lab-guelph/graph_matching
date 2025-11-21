# Methyl Assignment with Graph Matching Networks

This project implements a deep learning-based approach for automatic methyl assignment in large proteins from NMR data. It uses a Graph Matching Network (GMN) to learn the similarity between an experimental NMR graph and a ground-truth graph derived from a PDB file, and then predicts the peak assignments.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd graph_matching_gemini
    ```

2.  **Create a virtual environment and install dependencies using `uv`:**
    ```bash
    # Create a virtual environment
    uv venv

    # Activate the virtual environment
    # On Linux/macOS:
    source .venv/bin/activate
    # On Windows (Command Prompt):
    # .venv\Scripts\activate.bat
    # On Windows (PowerShell):
    # .venv\Scripts\Activate.ps1

    # Install dependencies
    uv pip install -r requirements.txt
    ```

## Usage

The main script `src/main.py` handles the training, evaluation, and inference pipelines.

To run the full pipeline (training, evaluation, and inference on the provided sample):
```bash
# Ensure your virtual environment is activated
source .venv/bin/activate

# Run the main script
uv run python -m src.main
```

The script will:
1.  Train the `GraphMatchingModel` on the sample data in `data/train`.
2.  Evaluate the trained model.
3.  Run inference on the sample data in `data/inference/yme1l_201`.
4.  Save the peak assignments to `data/inference/yme1l_201/result.txt`.

## Project Structure

The project is organized as follows:
- **`data/`**: Contains training and inference data.
- **`src/`**: Main source code.
  - `data_preprocessing/`: Modules for loading and preprocessing NMR and PDB data.
  - `graph_matching/`: The GMN model and loss function.
  - `main.py`: The main script to run the pipelines.
- **`tests/`**: Unit tests.

## Deep Learning Model

The core of the project is the `GraphMatchingModel`, which implements a Graph Matching Network. The model takes two graphs as input and learns to predict their similarity. For assignment, it uses a nearest-neighbor approach in the learned embedding space to find the best match for each experimental peak.
