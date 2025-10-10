# Quick Reference Card

## Run the Pipeline

```bash
cd applications/gravity

# 1-4: Generate data (run once)
python scripts/1_generate_data.py          # Country features
python scripts/2_fetch_tariffs.py          # Bilateral tariffs
python scripts/3_fetch_real_data.py        # Firm distribution
python scripts/4_validate_data.py          # Quality checks

# 5: Simulate (main!)
source ../../.bundle/bin/activate
mpirun -n 4 python scripts/5_simulate.py --num_firms 5000 --exclude_home

# 6-8: Analysis & plots
python scripts/6_fetch_trade_data.py       # Real trade data
python scripts/7_visualize.py              # Basic plots
python scripts/8_visualize_comparative.py  # Simulated vs real
```

## Key Files

| File | Description |
|------|-------------|
| `data/simulation/obs_bundles.csv` | **Main output** - 5000×50 export choices |
| `data/plots/*.png` | 9 visualizations |
| `docs/COMPLEMENTARITY_FINDINGS.md` | How we achieved sparsity |
| `docs/DATA_GUIDE.md` | Feature descriptions |

## Customize Simulation

**Different modular features:**
```bash
python scripts/5_simulate.py \
  --modular gdp_billions literacy_pct infrastructure_index
```

**Different quadratic features:**
```bash
python scripts/5_simulate.py \
  --quadratic distances bilateral_tariffs common_language
```

**Adjust parameters:**
```bash
python scripts/5_simulate.py \
  --theta_modular 5.0 1.0 \
  --theta_quadratic 0.0 0.0 \
  --sigma 0.5
```

## Key Finding

**θ_quadratic ≈ 0 → Sparse patterns** ✨

- High values (0.5) → complementarities → all export to all
- Near-zero (0.001) → minimal complementarities → sparse (74%)

See `docs/COMPLEMENTARITY_FINDINGS.md` for full analysis.

## Project Structure

```
scripts/     → All Python (numbered by order)
docs/        → Documentation  
data/
  features/  → Country + pairwise data
  simulation/→ obs_bundles (main output)
  plots/     → 9 visualizations
```

