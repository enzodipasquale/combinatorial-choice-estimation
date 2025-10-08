# Gravity Model Data Explanation

## What Data We Generate

The pipeline creates **two types of features** for modeling firm export destination choices:

---

## 1. Country-Level Features (Modular)

**File**: `country_features.csv`  
**Dimensions**: `(N countries) × (M features)`

These are characteristics of **individual countries** that affect their attractiveness as export destinations.

### A. Economic Indicators (18 features)
From World Bank API (2020-2023):

| Feature | Description | Example Values |
|---------|-------------|----------------|
| `gdp_billions` | GDP in billions USD | USA: $21,354B |
| `population_millions` | Total population | CHN: 1,411M |
| `gdp_per_capita` | GDP per capita | CHE: $92,000 |
| `gdp_growth_pct` | Annual growth rate | VNM: 8.0% |
| `trade_openness_pct` | (Exports+Imports)/GDP | SGP: 350% |
| `exports_billions` | Total exports | DEU: $1,571B |
| `imports_billions` | Total imports | USA: $3,376B |
| `tariff_rate` | Average tariff | IND: 13.8% |
| `fdi_inflow_billions` | Foreign investment | USA: $285B |
| `inflation_pct` | Consumer price inflation | ARG: 94.8% |
| `population_density` | People per km² | SGP: 8,592 |
| `urban_population_pct` | Urban population | QAT: 99.2% |
| `internet_users_pct` | Internet penetration | ISL: 99.0% |
| `mobile_per_100` | Mobile subscriptions | ARE: 209 |
| `electricity_access_pct` | Access to electricity | Most: 100% |
| `literacy_rate` | Adult literacy | - |
| `unemployment_pct` | Unemployment rate | ZAF: 33.5% |
| `household_consumption_pct` | Consumption % GDP | - |

### B. Language Dummies (~28 features)
Binary indicators for official languages:
- `eng` (English): USA, GBR, AUS, CAN, IND, SGP...
- `spa` (Spanish): ESP, MEX, ARG, COL, CHL...
- `fra` (French): FRA, BEL, CHE...
- `ara` (Arabic): SAU, ARE, EGY...
- etc.

### C. Region Dummies (5 features)
Binary indicators for continents:
- `Asia`: CHN, JPN, IND, KOR...
- `Europe`: DEU, GBR, FRA, ITA...
- `America`: USA, BRA, CAN, MEX...
- `Africa`: NGA, ZAF, EGY...
- `Oceania`: AUS, NZL...

**Total**: ~51 country-level features

---

## 2. Country-Pair Features (Quadratic)

**Files**: Multiple `(N × N)` matrices  
**Dimensions**: `(N countries) × (N countries)` for each feature

These capture **interactions between pairs of countries** (complementarities, network effects).

### A. Distances (`distances.csv`)
**Continuous** | Geodesic distance in kilometers

```
        USA    CHN    DEU    ...
USA       0   11,671  7,882  ...
CHN   11,671     0   7,242  ...
DEU    7,882  7,242     0   ...
```

- **Range**: Typically 800 km (neighbors) to 20,000 km (opposite sides of Earth)
- **Mean**: ~8,000 km
- **Interpretation**: Geographic trade costs, shipping costs
- **For supermodular**: Transformed to proximity = max_dist - log(dist)

### B. Common Language (`common_language.csv`)
**Binary** | 1 if countries share official language, 0 otherwise

```
        USA  CHN  GBR  IND
USA      1    0    1    1   (English)
CHN      0    1    0    0   (Chinese)
GBR      1    0    1    1   (English)
IND      1    0    1    1   (English)
```

- **Interpretation**: Communication ease, cultural proximity
- **Literature effect**: +35% trade boost

### C. Common Region (`common_region.csv`)
**Binary** | 1 if same continent, 0 otherwise

```
        USA  MEX  CHN  DEU
USA      1    1    0    0   (Americas)
MEX      1    1    0    0   (Americas)
CHN      0    0    1    0   (Asia)
DEU      0    0    0    1   (Europe)
```

- **Interpretation**: Regional trade agreements (NAFTA, EU, ASEAN)
- **Literature effect**: +65% trade boost

### D. Contiguity (`contiguity.csv`)
**Binary** | 1 if countries share a border, 0 otherwise

- Approximated as distance < 100 km
- **Interpretation**: Zero transportation cost, customs unions
- **Examples**: USA-CAN, DEU-FRA, CHN-IND

### E. Timezone Difference (`timezone_difference.csv`)
**Continuous** | Absolute difference in hours (0-12)

- Calculated from longitude: Δhours = |lon₁ - lon₂| / 15°
- **Interpretation**: Real-time coordination costs, business hours overlap
- **Examples**: USA-CHN: ~13 hours, USA-GBR: ~5 hours

### F. Colonial Ties (`colonial_ties.csv`)
**Binary** | 1 if historical colonial relationship, 0 otherwise

- Approximated as: same language + different region
- **Examples**: GBR-IND, ESP-MEX, FRA-DZA
- **Interpretation**: Historical institutions, legal systems, cultural ties

### G. Legal Origin Similarity (`legal_origin_similarity.csv`)
**Binary** | 1 if similar legal systems, 0 otherwise

- Common law (GBR, USA, IND, AUS, CAN...) vs Civil law
- **Interpretation**: Contract enforcement, business practices

---

## 3. Firm-Level Data (For Simulation)

**File**: `firm_choices.csv`  
**Dimensions**: `(F firms) × (N countries + 1)`

### Structure
```csv
firm_id,home_country,USA,CHN,JPN,DEU,...
0,CHN,True,True,True,False,...
1,USA,True,False,True,True,...
```

### Additional Data (`simulation_data.npz`)
- `obs_bundles`: (F × N) boolean choices
- `home_countries`: (F,) home country index for each firm
- `item_modular`: (N × 4) normalized country features
- `item_quadratic`: (N × N × 3) pairwise features
- `agent_modular`: (F × N × 1) firm-destination features
- `theta_true`: (K,) true parameter vector
- `sigma`: Error standard deviation

---

## 4. Calibration Data

**File**: `calibration_data.npz`

### Firm Distribution
- `firm_weights`: Probability each country hosts firms
- `total_real_firms`: Estimated global firm count
- Based on World Bank enterprise indicators or GDP^0.8

### Gravity Parameters
Literature-calibrated values (Head & Mayer 2014):
- `theta_gdp`: 0.85 (market size effect)
- `theta_proximity`: 1.1 (inverse of distance elasticity -1.1)
- `theta_language`: 0.35 (common language boost)
- `theta_region`: 0.65 (regional integration)
- `theta_home`: 2.5 (home market bias)

---

## How To Use

### Generate for Any Countries

```bash
# Top 50 by GDP
python generate_data.py --num_countries 50

# Top 100 by population
python generate_data.py --num_countries 100 --sort_by population

# Custom set (e.g., EU countries)
python generate_data.py --countries DE FR IT ES NL BE AT

# Custom set (e.g., BRICS)
python generate_data.py --countries BR RU IN CN ZA
```

### What Gets Created

For N countries, you get:
- **1 file** with N × 51 country features
- **6 files** with N × N pairwise features
- **All data fetched from real APIs** (World Bank, Nominatim, REST Countries)

### Data Sources

| Data Type | Source | Method |
|-----------|--------|--------|
| GDP, trade, population | World Bank | API (pandas_datareader) |
| Coordinates | OpenStreetMap | Nominatim geocoding |
| Languages | REST Countries | v3.1 API |
| Regions | UN classification | country_converter |
| Firm counts | World Bank | Enterprise indicators |
| Parameters | Academic literature | Meta-analysis |

---

## Interpretation for BundleChoice

### Modular Features → Linear Utility
```
U(country_j) = Σ θ_modular_k · X_j,k
```
- Firm values large GDP, high trade openness, etc.

### Quadratic Features → Complementarities
```
U(bundle S) += ΣΣ θ_quadratic_k · X_ij,k · 1{i∈S, j∈S}
```
- Proximity: Nearby countries complement each other (regional hubs)
- Language: Exporting to multiple English-speaking countries has synergies
- Region: Regional networks (EU, ASEAN, NAFTA)

### Example Interpretation
A firm choosing `{USA, CAN, MEX}` gets:
- **Modular**: Sum of individual country utilities
- **Quadratic**: Bonuses for:
  - USA-CAN proximity + common language
  - USA-MEX proximity + regional integration (NAFTA)
  - All three in Americas

This captures **network effects** and **economies of scope** in international trade!
