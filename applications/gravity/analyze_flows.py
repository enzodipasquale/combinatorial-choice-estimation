"""Analyze simulated trade flows"""
import numpy as np
import pandas as pd
import sys

# Load results
npz_file = 'datasets/simulated_choices.npz'
try:
    sim = np.load(npz_file)
except FileNotFoundError:
    print(f"Error: {npz_file} not found. Run simulate_simple.py first.")
    sys.exit(1)
bundles = sim['bundles']
bilateral = sim['bilateral_flows']
home_countries = sim['home_countries']
countries = sim['country_names']

features = pd.read_csv('datasets/country_features.csv', index_col=0)
distances = pd.read_csv('datasets/distances.csv', index_col=0)

print('='*70)
print('TRADE FLOW ANALYSIS: 10,000 Firms × 50 Countries')
print('='*70)

# GDP correlation
gdp = features['gdp_billions'].fillna(0).values
inflows = bundles.sum(axis=0)

print(f'\n📊 Do large economies receive more exports?')
corr = np.corrcoef(gdp, inflows)[0, 1]
print(f'   Correlation(GDP, Inflows): {corr:.3f}')

top5_gdp_idx = np.argsort(gdp)[::-1][:5]
print(f'\n   Top 5 by GDP → Export inflows:')
for idx in top5_gdp_idx:
    print(f'     {countries[idx]}: ${gdp[idx]:.0f}B GDP → {inflows[idx]}/10000 exporters ({inflows[idx]/100:.0f}%)')

# Home market
print(f'\n🏠 Home Market Bias:')
home_exports = sum([bundles[i, home_countries[i]] for i in range(len(home_countries))])
print(f'   {home_exports}/{len(home_countries)} firms export to home ({home_exports/len(home_countries)*100:.1f}%)')

# Firm heterogeneity
destinations_per_firm = bundles.sum(axis=1)
print(f'\n📦 Export Intensity:')
print(f'   Mean destinations: {destinations_per_firm.mean():.1f}')
print(f'   Std: {destinations_per_firm.std():.1f}')
print(f'   Range: {destinations_per_firm.min():.0f} - {destinations_per_firm.max():.0f}')

# Distance decay
print(f'\n🌍 Distance Patterns (USA-based firms):')
usa_idx = list(countries).index('USA')
usa_firm_indices = np.where(home_countries == usa_idx)[0]
print(f'   {len(usa_firm_indices)} US firms')

# Where do US firms export?
usa_destinations = bundles[usa_firm_indices, :].sum(axis=0)
top_usa_dest = np.argsort(usa_destinations)[::-1][:10]
print(f'\n   Top destinations for US firms:')
for dest_idx in top_usa_dest:
    dist = distances.iloc[usa_idx, dest_idx]
    count = usa_destinations[dest_idx]
    print(f'     → {countries[dest_idx]}: {count} exporters, {dist:.0f} km')

# Regional patterns
print(f'\n🌎 Regional Integration:')
region_mat = pd.read_csv('datasets/common_region.csv', index_col=0).values
region_exports = 0
total_exports = 0

for i in range(len(home_countries)):
    home_region = np.where(region_mat[home_countries[i], :] == 1)[0]
    exports_in_region = bundles[i, home_region].sum()
    total_exp = bundles[i, :].sum()
    region_exports += exports_in_region
    total_exports += total_exp

pct_regional = region_exports / total_exports * 100
print(f'   {pct_regional:.1f}% of exports stay within home region')

print(f'\n' + '='*70)
