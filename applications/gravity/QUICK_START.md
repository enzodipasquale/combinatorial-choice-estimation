# Gravity Model - Quick Start

## What You Get

A complete dataset for estimating firm export destination choices with:
- **Real country economic data** (World Bank)
- **Geographic relationships** (distances, regions, languages)
- **Flexible**: 10 to 100+ countries, any combination
- **Ready for BundleChoice** estimation

---

## Pipeline

### Step 1: Generate Data (5-20 min depending on N)

```bash
cd applications/gravity

# Option A: Top 50 countries by GDP
python generate_data.py --num_countries 50

# Option B: Custom countries (e.g., G20)
python generate_data.py --countries US CN JP DE GB FR IT CA KR RU BR IN MX AU SA TR ID AR ZA

# Option C: Top 100 by population
python generate_data.py --num_countries 100 --sort_by population
```

**Output**: 7 files in `datasets/`
- `country_features.csv`: N × 51 matrix (country characteristics)
- `distances.csv`: N × N matrix (km)
- `common_language.csv`: N × N binary
- `common_region.csv`: N × N binary
- `contiguity.csv`: N × N binary
- `timezone_difference.csv`: N × N continuous
- `colonial_ties.csv`, `legal_origin_similarity.csv`: N × N binary

### Step 2: Validate Data (10 sec)

```bash
python validate_data.py
```

Checks:
- GDP/population ranges realistic
- Distance matrix symmetric
- No missing critical data
- Feature consistency

### Step 3: Calibrate (30 sec)

```bash
python fetch_real_data.py
```

Fetches:
- Real firm counts per country (World Bank)
- Literature-calibrated parameters (gravity equation studies)

**Output**: `calibration_data.npz`

### Step 4: Simulate (working on MPI debug)

```bash
mpirun -n 4 python simulate_choices.py --num_firms 1000 --use_calibration
```

---

## Data Structure Summary

### What Each File Contains

| File | Type | Dimensions | Content |
|------|------|------------|---------|
| `country_features.csv` | Table | N × 51 | GDP, pop, language dummies, region dummies, etc. |
| `distances.csv` | Matrix | N × N | Kilometers between capitals |
| `common_language.csv` | Matrix | N × N | 1 if shared language |
| `common_region.csv` | Matrix | N × N | 1 if same continent |
| `contiguity.csv` | Matrix | N × N | 1 if neighbors |
| `timezone_difference.csv` | Matrix | N × N | Hours difference |
| `colonial_ties.csv` | Matrix | N × N | 1 if historical ties |

### For BundleChoice

**Modular features** (51 total):
- 18 economic indicators
- 28 language dummies  
- 5 region dummies

**Quadratic features** (6 matrices):
- Distances → proximity
- Language, region, border → complementarities

---

## Example Use Cases

### Scenario 1: European Union
```bash
python generate_data.py --countries DE FR IT ES NL BE AT PL GR PT
```
→ Study intra-EU trade, regional integration effects

### Scenario 2: Asia-Pacific
```bash
python generate_data.py --countries CN JP KR AU SG MY TH VN ID PH
```
→ ASEAN integration, China's role

### Scenario 3: Global Top 50
```bash
python generate_data.py --num_countries 50
```
→ Global trade patterns, all major economies

### Scenario 4: Emerging Markets
```bash
python generate_data.py --countries BR RU IN CN ZA MX TR ID TH VN
```
→ South-South trade

---

## Key Features

✅ **Fully Flexible**
- Any number of countries (tested up to 100)
- Any combination of country codes
- Ranked by GDP or population

✅ **Real Data**
- All from APIs (World Bank, Nominatim, REST Countries)
- Latest available data (2020-2023)
- No hardcoded values

✅ **Validated**
- Automatic quality checks
- Realistic ranges verified
- Consistency enforced

✅ **Calibrated**
- Firm distribution from real business data
- Parameters from gravity equation literature
- Empirically grounded

---

## Next Steps

Once data is generated:
1. **Explore**: Look at `datasets/*.csv`
2. **Validate**: Run `python validate_data.py`
3. **Simulate**: Generate firm choices (debugging in progress)
4. **Estimate**: Use BundleChoice row generation to recover parameters

See `DATA_EXPLANATION.md` for detailed feature descriptions.
