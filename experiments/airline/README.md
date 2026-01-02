# Airline Network Analysis

This folder contains tools for analyzing airline networks using the bundlechoice framework.

## Structure

- **`visualize_airline_networks.py`** - Visualization script for synthetic airline networks
- **`real_data/`** - Real airline data
  - **`fetch_airline_routes.py`** - Script to download real airline route data from OpenFlights
  - **`data/`** - Data files (routes, hubs, cities)
  - **`plots/`** - Generated visualizations

## Scenario Builder

The airline scenario builder is in `bundlechoice/factory/airline.py` and can be accessed via:

```python
from bundlechoice.scenarios import ScenarioLibrary

scenario = ScenarioLibrary.airline().build()
```

## Quick Start

### Synthetic Networks
```bash
python visualize_airline_networks.py
```

### Real Data
```bash
cd real_data
python fetch_airline_routes.py  # Downloads routes and identifies hubs
```

This will create:
- `data/airline_routes_real.csv` - Routes for all airlines
- `data/airline_hubs_all.csv` - Hub cities for all airlines (auto-generated)

## Data

After running `fetch_airline_routes.py`, you'll have:
- **Routes**: 64,637 routes across 511 airlines
- **Hubs**: Hub cities identified for each airline (cities with ≥10 routes)
- **Cities**: Major cities with coordinates and population

