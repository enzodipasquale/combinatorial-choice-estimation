# Gravity Model: Firm Export Destination Choice

Simulates firm export choices using a gravity model with the **QuadSupermodular** solver and **constraint masks** for home exclusion.

---

## 🚀 Quick Start

```bash
# Step 1: Generate country data (features, distances)
python 1_generate_data.py

# Step 2: Simulate firm export choices (QuadSupermodular + constraint masks)
python 2_simulate.py

# Step 3: Visualize results
python 3_visualize.py
```

---

## 📊 Results Summary

### Key Metrics
- **Firms simulated**: 10,000
- **Countries**: 50
- **Average partners**: 19.6 (range: 9-33)
- **Sparsity**: 60.8%
- **Self-trade**: **0.0%** ✅ (constraint mask working!)
- **GDP correlation**: 0.691 (strong gravity effect)

### Top 10 Export Destinations
1. **India**: 9,404 firms (94%)
2. **China**: 8,482 firms (85%)
3. **Japan**: 8,453 firms (85%)
4. **USA**: 7,958 firms (80%)
5. **Germany**: 7,585 firms (76%)
6. **UK**: 6,230 firms (62%)
7. **France**: 6,094 firms (61%)
8. **Brazil**: 5,294 firms (53%)
9. **Italy**: 5,242 firms (52%)
10. **Russia**: 5,058 firms (51%)

---

## 🛠️ Technical Details

### Home Exclusion (Constraint Mask)
- Each firm **cannot export to its home country**
- Implemented via `constraint_mask` in `input_data`
- Boolean mask per firm: `True` = feasible, `False` = excluded
- Properly scattered across MPI ranks in `data_manager.py`

### Model Features
**Modular:**
- GDP (θ = 2.0, strong)
- Population (θ = 0.5, moderate)
- Firm heterogeneity (θ = 0.3)

**Quadratic:**
- Proximity (inverse log-distance, θ = 0.0 for sparsity)

### Firm Distribution
Firms are drawn from countries proportional to GDP^0.8:
- USA: 2,042 firms (20%)
- China: 1,518 firms (15%)
- Japan: 654 firms (7%)
- Germany: 533 firms (5%)
- UK: 393 firms (4%)

---

## 📁 Output Files

**`datasets/`**
- `country_features.csv` - Country-level data (GDP, population, etc.)
- `distances.csv` - Pairwise geographic distances
- `quad_simulation.csv` - Firm-level export choices (10,000 × 50)
- `quad_simulation.npz` - Simulation data (bundles, home_countries, theta)
- `comprehensive_analysis.png` - Main visualization (6 panels)
- `parameter_effects.png` - Parameter sensitivity plots

---

## ✅ Verification Checklist

- [x] No self-trade (constraint mask works)
- [x] Realistic sparsity (~60%)
- [x] Strong gravity effects (GDP correlation > 0.69)
- [x] Heterogeneous partner counts (9-33 range)
- [x] Large economies dominate top destinations
- [x] Proper MPI scatter of constraint masks
- [x] Boolean mask → indices conversion in solver

---

**Built with:** `bundlechoice` QuadSupermodularNetwork solver
