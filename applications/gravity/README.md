# Gravity Model Application

Simulated firm export choices to 50 countries using real-world data and the BundleChoice framework.

## Quick Start

```bash
# 1. Generate country data (GDP, distances, etc.)
python scripts/1_generate_data.py

# 2. Generate bilateral tariffs
python scripts/2_fetch_tariffs.py

# 3. Calibrate firm distribution
python scripts/3_fetch_real_data.py

# 4. Validate all data
python scripts/4_validate_data.py

# 5. Simulate export choices (activate venv first!)
source ../../.bundle/bin/activate
mpirun -n 4 python scripts/5_simulate.py --num_firms 5000

# 6. Fetch real trade data for comparison
python scripts/6_fetch_trade_data.py

# 7. Generate visualizations
python scripts/7_visualize.py
python scripts/8_visualize_comparative.py
```

## What This Does

**Simulates realistic firm export decisions** where:
- 5,000 firms choose which countries to export to
- Firms distributed across 50 countries (proportional to real firm counts)
- Decisions based on gravity model: GDP, distance, language, tariffs, etc.
- Uses QuadSupermodular solver to handle complementarities

**Key Achievement**: By reducing complementarities (θ_quadratic ≈ 0), achieved:
- ✅ **74% sparsity** (firms export to ~13/50 countries)
- ✅ **Realistic patterns** (large markets in most bundles)
- ✅ **Strong gravity** (0.59 GDP correlation)

## Project Structure

```
gravity/
├── README.md                    # This file
├── scripts/                     # All Python scripts (numbered)
│   ├── 1_generate_data.py      # Fetch country features from APIs
│   ├── 2_fetch_tariffs.py      # Generate bilateral tariffs (FTAs)
│   ├── 3_fetch_real_data.py    # Calibrate firm distribution
│   ├── 4_validate_data.py      # Quality checks
│   ├── 5_simulate.py           # Generate obs_bundles (main!)
│   ├── 6_fetch_trade_data.py   # Real trade for validation
│   ├── 7_visualize.py          # Basic plots
│   └── 8_visualize_comparative.py  # Simulated vs real
├── docs/                        # Documentation
│   ├── DATA_GUIDE.md           # Feature descriptions
│   └── COMPLEMENTARITY_FINDINGS.md  # Key insights
└── data/                        # All data files
    ├── features/               # Country + pairwise data
    ├── simulation/             # obs_bundles.csv/.npz
    └── plots/                  # All visualizations (9 PNG)
```

## Data Generated

### Features (51 modular + 8 pairwise)
- **Country**: GDP, population, trade openness, literacy, infrastructure...
- **Pairwise**: distances, tariffs, common language, region, FTAs...

### Simulation Results
- `obs_bundles.csv` - 5000 × 50 binary matrix
- `obs_bundles.npz` - includes home countries, theta_true

### Visualizations (9 plots)
1. Bilateral heatmap
2. Trade network graph
3. Gravity patterns (distance, GDP, intensity)
4. Regional integration matrix
5. Simulated vs real comparison (4-panel)
6. Gravity equation fit (4-panel)
7. Trade statistics (6-panel)
8. Trade intensity analysis
9. Complementarity reduction impact

## Key Parameters

**For realistic sparse patterns**:
```python
--num_firms 5000
--modular gdp_billions population_millions  # Strong systematic effects
--quadratic distances common_language common_region
--theta_modular 3.0 0.5                     # Strong
--theta_quadratic 0.0 0.001 0.001          # Near ZERO (key!)
--sigma 0.8                                 # Low noise
--exclude_home                              # No home exports
```

**Why θ_quadratic ≈ 0?**
- Positive values → complementarities → "snowball effect" → all export to all
- Near-zero → minimal complementarities → sparse, realistic choices

See `docs/COMPLEMENTARITY_FINDINGS.md` for full analysis.

## Customization

**Different countries:**
```bash
python scripts/1_generate_data.py --num_countries 100 --sort_by population
```

**Different covariates:**
```bash
mpirun -n 4 python scripts/5_simulate.py \
  --modular gdp_billions literacy_pct \
  --quadratic distances bilateral_tariffs
```

**Different parameters:**
```bash
mpirun -n 4 python scripts/5_simulate.py \
  --theta_modular 5.0 1.0 \
  --theta_quadratic 0.0 0.0
```

## Next Steps

1. **Parameter Estimation**: Use BundleChoice to recover θ from choices
2. **Counterfactuals**: Remove tariffs, add FTAs, measure impact
3. **Model Validation**: Compare to real firm-level export data
4. **Publication**: Monte Carlo validation + real application

## Documentation

- `docs/DATA_GUIDE.md` - Detailed feature descriptions
- `docs/COMPLEMENTARITY_FINDINGS.md` - How we achieved sparsity
- All scripts have detailed docstrings

---

**Status**: ✅ Complete pipeline with realistic sparse trade patterns!
