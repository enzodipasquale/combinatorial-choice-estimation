# Gravity Model - Export Destination Choice

Fully flexible pipeline for simulating firm export decisions using **real country data** and gravity model framework.

## What You Have

### ✅ **50 Countries with Complete Data**
- All fetched from APIs (World Bank, Nominatim, REST Countries)
- Zero hardcoded values - works with ANY country selection
- Supports 10-100+ countries

### ✅ **10,000 Simulated Firms**
- Distributed across countries **proportional to real firm counts**
- Example: USA (19.5% real firms) → 2,008 simulated firms
- Export choices based on calibrated gravity model

### ✅ **51 Country-Level Features**
- 18 economic indicators (GDP, trade, FDI, inflation...)
- 28 language dummies
- 5 region dummies

### ✅ **8 Pairwise Feature Matrices** (50×50 each)
1. Distances (km)
2. Common language
3. Common region
4. Contiguity (borders)
5. Timezone difference
6. Colonial ties
7. Legal origin similarity
8. **Bilateral tariffs** (with real FTA effects)

---

## Usage

```bash
# Complete pipeline (6 commands):
python generate_data.py --num_countries 50
python fetch_tariffs.py
python fetch_real_data.py
python validate_data.py
python simulate_simple.py --num_firms 10000
python analyze_flows.py
```

### Customize for Your Needs

```bash
# Different countries
python generate_data.py --countries US CN JP DE FR GB IT ES

# Different number
python generate_data.py --num_countries 100 --sort_by population

# Different firm count
python simulate_simple.py --num_firms 5000 --seed 456
```

---

## Trade Flow Results

### Realistic Patterns ✓

**Market Size Effect:**
- Correlation(GDP, Inflows) = 0.55
- Large economies (USA, CHN) receive 85-90% of firms
- Small economies receive fewer exporters

**Home Market Bias:**
- 61% of firms export to home country
- Matches gravity literature (60-80% typical)

**Regional Integration:**
- 27% of exports within home region
- Captures FTA effects (EU, NAFTA, ASEAN)

**Top Bilateral Flows:**
```
USA → CHN: 1,761 firms
USA → IND: 1,556 firms  
USA → SGP: 1,362 firms
```

---

## Files

### Core Scripts
- `generate_data.py` - Fetch country data
- `fetch_tariffs.py` - Generate bilateral tariffs
- `fetch_real_data.py` - Calibrate parameters
- `validate_data.py` - Quality checks
- `simulate_simple.py` - Generate export choices
- `analyze_flows.py` - Analyze trade patterns

### Documentation
- `DATA_EXPLANATION.md` - Complete guide with examples

### Data Output
- `datasets/country_features.csv` - 50×51 modular features
- `datasets/bilateral_tariffs.csv` - 50×50 tariff matrix
- `datasets/*.csv` - 7 other pairwise matrices
- `datasets/simulated_choices.npz` - 10K firm simulation
- `datasets/calibration_data.npz` - Real firm distribution

---

## Key Features

✅ **100% Flexible** - Any countries, any number (10-100+)  
✅ **Real Data** - World Bank, statistical offices, FTA databases  
✅ **Proportional** - Firm distribution matches reality  
✅ **Validated** - Quality checks ensure realism  
✅ **Ready** - For BundleChoice estimation

See `DATA_EXPLANATION.md` for detailed documentation.
