#!/bin/bash
# Convenience script to run the Streamlit app with the correct Python environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Please run:"
    echo "  python3 -m venv .venv"
    echo "  source .venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

source .venv/bin/activate

# Run streamlit
echo "Starting BERT Sentiment Analyzer..."
streamlit run app/streamlit_app.py "$@"
