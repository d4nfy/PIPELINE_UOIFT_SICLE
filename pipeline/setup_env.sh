#!/bin/bash
# setup script for uoift-sicle pipeline environment

set -e

echo "uoift-sicle environment setup"
echo

if ! command -v python3 &> /dev/null; then
    echo "python3 not found. install python 3.11+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 --version)
echo "found: $PYTHON_VERSION"
echo

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    echo "virtual environment already exists at ./$VENV_DIR"
    read -p "recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "removing existing environment..."
        rm -rf "$VENV_DIR"
    else
        echo "using existing environment."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    echo "virtual environment created at ./$VENV_DIR"
    echo
fi

echo "activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "upgrading pip..."
pip install --upgrade pip setuptools wheel
echo

echo "installing dependencies from requirements.txt..."
pip install -r requirements.txt
echo

echo "verifying installations..."
echo

python3 << 'EOF'
import importlib

packages = [
    ('numpy', '1.24'),
    ('scipy', '1.11'),
    ('skimage', '0.21'),
    ('PIL', '10'),
    ('cv2', '4'),
    ('stardist', None),
    ('tensorflow', None),
    ('cellpose', None),
]

print(f"{'package':20s} {'version':10s}")

all_ok = True
for pkg_name, expected_ver in packages:
    try:
        pkg = importlib.import_module(pkg_name)
        version = getattr(pkg, '__version__', 'unknown')
        print(f"  {pkg_name:20s} {version}")
    except ImportError:
        print(f"  {pkg_name:20s} NOT INSTALLED")
        all_ok = False

if all_ok:
    print("\nall packages installed.")
else:
    print("\nsome packages are missing.")

EOF

echo
echo "setup complete."
echo
echo "to activate: source venv/bin/activate"
echo "to run:      python run.py --method boundary_aware"
echo
