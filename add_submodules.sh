#!/bin/bash

# Script to add private repositories as submodules
# Run this AFTER creating the private repositories

set -e

echo "Adding private repositories as submodules..."
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the root of your combinatorial-choice-estimation repository"
    exit 1
fi

# Get the private repo URLs from user
echo "Please provide the URLs of your private repositories:"
echo ""
read -p "Enter the URL for applications private repo (e.g., https://github.com/username/combinatorial-choice-applications.git): " APPLICATIONS_URL
read -p "Enter the URL for experiments private repo (e.g., https://github.com/username/combinatorial-choice-experiments.git): " EXPERIMENTS_URL
read -p "Enter the URL for benchmarking private repo (e.g., https://github.com/username/combinatorial-choice-benchmarking.git): " BENCHMARKING_URL

echo ""
echo "Adding submodules..."

# Remove existing folders if they exist
if [ -d "applications" ]; then
    echo "Removing existing applications folder..."
    rm -rf applications
fi

if [ -d "experiments" ]; then
    echo "Removing existing experiments folder..."
    rm -rf experiments
fi

if [ -d "benchmarking" ]; then
    echo "Removing existing benchmarking folder..."
    rm -rf benchmarking
fi

# Add submodules
echo "Adding applications as submodule..."
git submodule add "$APPLICATIONS_URL" applications

echo "Adding experiments as submodule..."
git submodule add "$EXPERIMENTS_URL" experiments

echo "Adding benchmarking as submodule..."
git submodule add "$BENCHMARKING_URL" benchmarking

# Initialize and update submodules
echo "Initializing and updating submodules..."
git submodule update --init --recursive

echo ""
echo "Submodules added successfully!"
echo ""
echo "Next steps:"
echo "1. Commit the changes:"
echo "   git add .gitmodules applications experiments benchmarking"
echo "   git commit -m 'Add applications, experiments, and benchmarking as private submodules'"
echo ""
echo "2. Push to your public repository:"
echo "   git push origin main"
echo ""
echo "3. For collaborators to get the submodules:"
echo "   git clone --recursive https://github.com/YOUR_USERNAME/combinatorial-choice-estimation.git"
echo "   OR if already cloned:"
echo "   git submodule update --init --recursive"
echo ""
echo "Note: Collaborators will need access to the private repositories to access the submodule contents." 