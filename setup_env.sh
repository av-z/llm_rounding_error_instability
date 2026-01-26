echo "Setting up Python Virtual Environment..."

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 could not be found."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate

pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

mkdir -p results
mkdir -p data

echo "Setup Complete."
