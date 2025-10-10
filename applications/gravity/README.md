# Gravity Model - Export Choice Simulation

Simulates 10,000 firms choosing export destinations based on gravity model.

## Run It

```bash
cd applications/gravity

# Simulate export choices
python simulate_simple.py

# Visualize results
python visualize_results.py
```

## Data

All in `datasets/`:
- `country_features.csv` - 50 countries (GDP, population, etc.)
- `distances.csv` - Pairwise distances (50×50)
- `simulated_choices.csv` - Results (10,000 firms × 50 countries)
- `simulated_choices.npz` - Binary format with metadata
- `trade_flow_analysis.png` - 6-panel visualization

## Results

**10,000 firms × 50 countries**
- 15 avg export partners
- Top corridor: USA→China (1,719 firms)
- Top destinations: China (8,676), USA (8,079), India (7,002)

## Visualization

6-panel analysis shows:
1. Partner distribution histogram
2. Top 15 destinations
3. Bilateral heatmap (15×15)
4. Distance decay pattern
5. GDP gravity effect
6. Top 12 bilateral corridors

---

**Status:** ✅ Clean, working, ready to use!
