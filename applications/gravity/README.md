# Gravity Model - Export Choice Simulation

Realistic simulation of 10,000 firms choosing export destinations based on gravity model.

## Run

```bash
cd applications/gravity

# Simulate
python simulate_simple.py

# Visualize
python visualize_results.py

# View plot
open datasets/trade_flow_analysis.png
```

## Results

**10,000 firms × 50 countries**

- **4.4 avg partners** (range: 0-13)
- **91% sparsity** 
- **0% self-trade** ✅ (firms don't export to home)

**Top Destinations:**
- China: 72.4%
- USA: 69.3%
- India: 28.9%

**Top Corridor:** USA → China (1,724 firms)

## Data Files

`datasets/`:
- `country_features.csv` - 50 countries (GDP, population, etc.)
- `distances.csv` - Pairwise distances (50×50)
- `simulated_choices.csv` - Results (10,000 × 50 binary)
- `simulated_choices.npz` - Binary format
- `trade_flow_analysis.png` - 6-panel visualization

## Visualization

6 panels showing:
1. **Partner distribution** - Bell curve around 4.4
2. **Top destinations** - China/USA dominate
3. **Bilateral heatmap** - 15×15 top countries
4. **Distance decay** - Clear negative pattern
5. **GDP effect** - Positive correlation (0.139)
6. **Top corridors** - USA↔China bilateral flows

## Key Features

✅ **No self-trade** - Home country excluded  
✅ **Heterogeneous** - Firms vary (0-13 partners)  
✅ **Sparse** - 91% sparsity (realistic!)  
✅ **Gravity effects** - Distance & GDP matter  
✅ **Clean** - Only essentials kept  

---

**Status:** ✅ Production-ready!
