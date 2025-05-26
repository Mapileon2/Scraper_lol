#!/bin/bash
set -e

echo "=== Starting Vercel Build ==="
echo "Installing Python dependencies..."

# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip and install pip-tools
python -m pip install --upgrade pip
pip install pip-tools

# Install only the essential dependencies first
pip install --no-cache-dir -r requirements-vercel.txt

echo "=== Build completed successfully ==="
