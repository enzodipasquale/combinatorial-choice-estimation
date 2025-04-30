#!/bin/bash

echo "Installing dependencies..."

# Step 1: Install core Python requirements
pip install -r requirements.txt

# Step 2: Install your local package in editable mode
pip install -e .


echo "Setup complete. 