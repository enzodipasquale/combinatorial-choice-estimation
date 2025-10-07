# Gravity Model Data Generator

Generates comprehensive real-world data for gravity model of export destination choice.

## Usage

```bash
python generate_data.py --num_countries 10 --sort_by gdp
```

### Parameters
- `--num_countries`: Number of countries to include (default: 15)
- `--sort_by`: Criterion to select top countries
  - `gdp`: By GDP (default)
  - `population`: By population
  - `gdp_per_capita`: By GDP per capita
  - `trade`: By trade openness
  - `exports`: By total exports

## Data Sources

All data fetched from real APIs:
- **World Bank API**: Economic indicators via pandas_datareader
- **Nominatim**: Geographic coordinates via geopy
- **REST Countries API**: Languages
- **country_converter**: UN region classifications

## Features Generated

### Country-Level (Modular) Features

#### Economic Size & Growth
- `gdp_billions`: GDP in billions USD
- `gdp_per_capita`: GDP per capita
- `gdp_growth_pct`: Annual GDP growth rate

#### Population & Demographics
- `population_millions`: Total population
- `population_density`: People per sq km
- `urban_population_pct`: Urban population percentage

#### Trade & Openness
- `trade_openness_pct`: Trade as % of GDP
- `exports_billions`: Total exports
- `imports_billions`: Total imports
- `tariff_rate`: Average tariff rate

#### Business Environment
- `fdi_inflow_billions`: Foreign direct investment inflows
- `inflation_pct`: Consumer price inflation

#### Infrastructure & Technology
- `internet_users_pct`: Internet penetration
- `mobile_per_100`: Mobile subscriptions per 100 people
- `electricity_access_pct`: Access to electricity

#### Human Capital
- `literacy_rate`: Adult literacy rate
- `unemployment_pct`: Unemployment rate

#### Market Characteristics
- `household_consumption_pct`: Household consumption as % GDP

#### Language Dummies
- Binary indicators for each language (eng, zho, spa, etc.)

#### Region Dummies
- Binary indicators for each region (Europe, Asia, Americas, Oceania, Africa)

### Pairwise (Quadratic) Features

- **`distances.csv`**: Geographic distance in km (computed via geodesic)
- **`common_language.csv`**: Binary indicator for shared official language
- **`common_region.csv`**: Binary indicator for same continent/region
- **`contiguity.csv`**: Binary indicator for shared border (distance < 100km)
- **`timezone_difference.csv`**: Time zone difference in hours
- **`colonial_ties.csv`**: Indicator for historical colonial relationships (same language, different region)
- **`legal_origin_similarity.csv`**: Binary indicator for similar legal systems (common law vs civil law)

## Output Files

All files saved to `datasets/`:
- `country_features.csv`: All country-level features
- `distances.csv`: Pairwise distances
- `common_language.csv`: Language similarity matrix
- `common_region.csv`: Regional similarity matrix
- `contiguity.csv`: Border sharing matrix
- `timezone_difference.csv`: Time zone differences
- `colonial_ties.csv`: Historical ties matrix
- `legal_origin_similarity.csv`: Legal system similarity matrix

## Examples

### Top 10 countries by GDP
```bash
python generate_data.py --num_countries 10 --sort_by gdp
```

### Top 20 countries by population
```bash
python generate_data.py --num_countries 20 --sort_by population
```

### Top 15 most trade-open countries
```bash
python generate_data.py --num_countries 15 --sort_by trade
```
