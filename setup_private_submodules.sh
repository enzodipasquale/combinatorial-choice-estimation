#!/bin/bash

# Script to set up private submodules for applications, experiments, and benchmarking folders
# This will help keep these folders private while maintaining the main repo as public

set -e

echo "Setting up private submodules for applications, experiments, and benchmarking folders..."
echo ""

# Step 1: Create temporary directories for the private repos
echo "Step 1: Creating temporary directories..."
mkdir -p /tmp/private_applications
mkdir -p /tmp/private_experiments
mkdir -p /tmp/private_benchmarking

# Step 2: Copy the folders to temporary locations
echo "Step 2: Copying folders to temporary locations..."
cp -r applications/* /tmp/private_applications/
cp -r experiments/* /tmp/private_experiments/
cp -r benchmarking/* /tmp/private_benchmarking/

# Step 3: Instructions for creating private repos
echo ""
echo "Step 3: Manual steps required:"
echo "=================================="
echo ""
echo "1. Go to GitHub and create THREE NEW PRIVATE repositories:"
echo "   - Name: 'combinatorial-choice-applications' (or similar)"
echo "   - Name: 'combinatorial-choice-experiments' (or similar)"
echo "   - Name: 'combinatorial-choice-benchmarking' (or similar)"
echo "   - Make sure all are PRIVATE repositories"
echo ""
echo "2. Clone the private repositories locally:"
echo "   git clone https://github.com/YOUR_USERNAME/combinatorial-choice-applications.git /tmp/private_applications_repo"
echo "   git clone https://github.com/YOUR_USERNAME/combinatorial-choice-experiments.git /tmp/private_experiments_repo"
echo "   git clone https://github.com/YOUR_USERNAME/combinatorial-choice-benchmarking.git /tmp/private_benchmarking_repo"
echo ""
echo "3. Copy the contents to the private repos:"
echo "   cp -r /tmp/private_applications/* /tmp/private_applications_repo/"
echo "   cp -r /tmp/private_experiments/* /tmp/private_experiments_repo/"
echo "   cp -r /tmp/private_benchmarking/* /tmp/private_benchmarking_repo/"
echo ""
echo "4. Commit and push to the private repos:"
echo "   cd /tmp/private_applications_repo && git add . && git commit -m 'Initial commit' && git push"
echo "   cd /tmp/private_experiments_repo && git add . && git commit -m 'Initial commit' && git push"
echo "   cd /tmp/private_benchmarking_repo && git add . && git commit -m 'Initial commit' && git push"
echo ""
echo "5. Come back to this directory and run the next script:"
echo "   ./add_submodules.sh"
echo ""
echo "The temporary files are in:"
echo "  - /tmp/private_applications"
echo "  - /tmp/private_experiments"
echo "  - /tmp/private_benchmarking"
echo ""
echo "You can delete these after setting up the private repos." 