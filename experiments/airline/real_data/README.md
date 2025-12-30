# Real Airline Data Analysis

This folder contains scripts for analyzing real airline network data using the bundlechoice framework.

## Data Sources

We'll use real data on:
- **Routes**: Real airline routes between cities
- **Cities**: City statistics (population, GDP, coordinates, etc.)
- **Airlines**: Real airlines with their hub locations

## Structure

- `load_real_data.py`: Script to load and process real airline data
- `visualize_real_networks.py`: Visualization of real airline networks
- `data/`: Directory for raw data files (CSV, JSON, etc.)
- `processed/`: Directory for processed data ready for bundlechoice

## Data Format

The processed data should match the bundlechoice format:
- Cities: array of (lat, lon) or (x, y) coordinates
- Markets: all possible origin-destination pairs
- Airlines: each airline has a set of hub cities
- Features: city-level and route-level characteristics


