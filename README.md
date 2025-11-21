# Methyl Assignment with Deep Learning

This project contains a deep learning-based approach for automatic methyl assignment in large proteins from NMR data.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
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

### Running the main pipeline

The `main.py` script serves as the entry point for the training, evaluation, and inference pipelines.

To run the dummy pipelines:
```bash
# Ensure your virtual environment is activated
source .venv/bin/activate # (or .venv\Scripts\activate.bat for Windows)
uv run python src/main.py
```

### Running Tests

To run the unit tests:
```bash
# Ensure your virtual environment is activated
source .venv/bin/activate # (or .venv\Scripts\activate.bat for Windows)
uv run pytest
```

## Project Structure

A modular structure is adopted for clarity and separation of concerns:

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