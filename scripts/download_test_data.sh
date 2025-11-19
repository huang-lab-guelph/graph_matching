#!/bin/bash
# Script to download test data from MAGIC repository

set -e  # Exit on error

echo "=========================================="
echo "Downloading NMR Test Data from MAGIC"
echo "=========================================="
echo ""

# Create temporary directory
TEMP_DIR=$(mktemp -d)
echo "Using temporary directory: $TEMP_DIR"

# Clone MAGIC repository
echo "Cloning MAGIC repository..."
git clone --depth 1 https://github.com/NMRsoftware/MAGIC.git "$TEMP_DIR/MAGIC"

# Create data directories
echo "Creating data directories..."
mkdir -p data/raw
mkdir -p data/processed

# Copy test data
echo "Copying test data..."
if [ -d "$TEMP_DIR/MAGIC/test_data" ]; then
    cp -r "$TEMP_DIR/MAGIC/test_data/"* data/raw/
    echo "✓ Test data copied to data/raw/"
else
    echo "Warning: test_data directory not found in MAGIC repository"
    echo "Checking for alternative data locations..."
    find "$TEMP_DIR/MAGIC" -name "*.pdb" -o -name "*.peaks" -o -name "*.txt"
fi

# Clean up
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

# List downloaded files
echo ""
echo "=========================================="
echo "Downloaded files:"
echo "=========================================="
find data/raw -type f -name "*.pdb" -o -name "*.txt" -o -name "*.peaks" | sort

echo ""
echo "✓ Download complete!"
echo ""
echo "Next steps:"
echo "  1. Review the data in data/raw/"
echo "  2. Verify file formats match expected formats"
echo "  3. Run: uv run python scripts/train_model.py --data-dir data/raw"
echo ""
