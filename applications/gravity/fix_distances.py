"""Regenerate distances for existing countries"""
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import time

features = pd.read_csv('datasets/country_features.csv', index_col=0)
countries = features.index.tolist()

print(f"Regenerating distances for {len(countries)} countries...")

# Get coordinates via geocoding
geolocator = Nominatim(user_agent="gravity_fix")
coords = {}

for country in countries:
    try:
        location = geolocator.geocode(country, timeout=10)
        if location:
            coords[country] = (location.latitude, location.longitude)
        else:
            print(f"  ! Could not geocode {country}")
            coords[country] = None
    except Exception as e:
        print(f"  ! Error geocoding {country}: {e}")
        coords[country] = None
    
    time.sleep(1)  # Rate limiting

# Compute distances
n = len(countries)
distances = np.zeros((n, n))

for i, c1 in enumerate(countries):
    for j, c2 in enumerate(countries):
        if i != j and coords[c1] and coords[c2]:
            distances[i, j] = geodesic(coords[c1], coords[c2]).kilometers

# Save
df = pd.DataFrame(distances, index=countries, columns=countries)
df.to_csv('datasets/distances.csv')

print(f"\n✓ Saved distances: {df.shape}")
print(f"  Range: {distances[distances>0].min():.0f} - {distances.max():.0f} km")
