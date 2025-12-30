#!/usr/bin/env python3
"""
Fetch official airline hub data from Wikipedia and other sources.

This script scrapes Wikipedia's airline hub information to get official
hub designations rather than inferring them from route patterns.

Usage:
    python fetch_official_hubs.py

Output:
    - data/airline_hubs_official.csv: Official hub cities for airlines
"""

import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Set
import time

# Optional dependencies for Wikipedia scraping
try:
    import requests
    from bs4 import BeautifulSoup
    HAS_SCRAPING_DEPS = True
except ImportError:
    HAS_SCRAPING_DEPS = False
    print("Warning: beautifulsoup4 and requests not installed.")
    print("Install with: pip install beautifulsoup4 requests")
    print("Will only use curated hub data.")


def get_wikipedia_hubs(airline_name: str) -> List[str]:
    """
    Scrape Wikipedia for official hubs of an airline.
    
    Returns list of hub city names.
    Requires: pip install beautifulsoup4 requests
    """
    if not HAS_SCRAPING_DEPS:
        return []
    
    # Clean airline name for URL
    url_name = airline_name.replace(' ', '_').replace('/', '_')
    search_url = f"https://en.wikipedia.org/wiki/{url_name}"
    
    hubs = []
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(search_url, timeout=15, headers=headers)
        
        if response.status_code != 200:
            # Try alternative URL formats
            url_name_alt = airline_name.replace(' ', '_').replace('-', '_')
            search_url = f"https://en.wikipedia.org/wiki/{url_name_alt}"
            response = requests.get(search_url, timeout=15, headers=headers)
            if response.status_code != 200:
                return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check infobox first (most reliable)
        infobox = soup.find('table', class_='infobox')
        if infobox:
            for row in infobox.find_all('tr'):
                header = row.find('th')
                if header:
                    header_text = header.get_text().strip().lower()
                    if any(keyword in header_text for keyword in ['hub', 'base', 'focus city', 'headquarters']):
                        data = row.find('td')
                        if data:
                            # Extract links (cities are usually linked)
                            for link in data.find_all('a'):
                                city = link.get_text().strip()
                                # Filter out non-city links
                                if (city and len(city) > 2 and len(city) < 40 and 
                                    not any(skip in city.lower() for skip in ['airport', 'international', 'air', 'airlines', 'airline', 'see', 'list'])):
                                    hubs.append(city)
                            # Also try plain text if no links
                            if not hubs:
                                text = data.get_text()
                                # Look for city patterns
                                cities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text)
                                hubs.extend([c for c in cities if len(c) > 2 and len(c) < 40])
        
        # Also check for "Hubs" section in article
        for heading in soup.find_all(['h2', 'h3']):
            heading_text = heading.get_text().strip().lower()
            if any(keyword in heading_text for keyword in ['hub', 'base', 'operations']):
                content = heading.find_next_sibling()
                count = 0
                while content and content.name not in ['h2', 'h3'] and count < 5:
                    if content.name == 'ul':
                        for li in content.find_all('li'):
                            # Get first link (usually the city)
                            link = li.find('a')
                            if link:
                                city = link.get_text().strip()
                                if city and len(city) > 2 and len(city) < 40:
                                    hubs.append(city)
                    content = content.find_next_sibling()
                    count += 1
        
        # Clean and deduplicate
        hubs = list(set([h.strip() for h in hubs if len(h.strip()) > 2]))
        return hubs[:20]  # Limit to reasonable number
        
    except requests.exceptions.RequestException:
        return []
    except Exception:
        return []


def get_known_official_hubs() -> Dict[str, List[str]]:
    """
    Manually curated list of official hubs for major airlines.
    This is more reliable than scraping for well-known airlines.
    """
    return {
        'American Airlines': [
            'Dallas', 'Charlotte', 'Chicago', 'Miami', 'Philadelphia',
            'Phoenix', 'Washington', 'New York', 'Los Angeles'
        ],
        'United Airlines': [
            'Chicago', 'Denver', 'Guam', 'Houston', 'Los Angeles',
            'Newark', 'San Francisco', 'Washington'
        ],
        'Delta Air Lines': [
            'Atlanta', 'Boston', 'Detroit', 'Los Angeles', 'Minneapolis',
            'New York', 'Salt Lake City', 'Seattle'
        ],
        'Southwest Airlines': [
            'Dallas', 'Denver', 'Chicago', 'Las Vegas', 'Phoenix',
            'Baltimore', 'Houston', 'Oakland', 'Los Angeles'
        ],
        'JetBlue Airways': [
            'New York', 'Boston', 'Fort Lauderdale', 'Orlando',
            'Los Angeles', 'San Juan'
        ],
        'Alaska Airlines': [
            'Seattle', 'Portland', 'Anchorage', 'Los Angeles', 'San Francisco'
        ],
        'Spirit Airlines': [
            'Fort Lauderdale', 'Detroit', 'Las Vegas', 'Dallas', 'Chicago'
        ],
        'Frontier Airlines': [
            'Denver', 'Las Vegas', 'Orlando', 'Phoenix', 'Trenton'
        ],
        'Hawaiian Airlines': [
            'Honolulu', 'Kahului', 'Kona', 'Lihue'
        ],
        'British Airways': [
            'London', 'Gatwick', 'Heathrow'
        ],
        'Lufthansa': [
            'Frankfurt', 'Munich', 'Vienna', 'Zurich', 'Brussels'
        ],
        'Air France': [
            'Paris', 'Lyon', 'Nice', 'Toulouse'
        ],
        'KLM Royal Dutch Airlines': [
            'Amsterdam'
        ],
        'Turkish Airlines': [
            'Istanbul'
        ],
        'Emirates': [
            'Dubai'
        ],
        'Qatar Airways': [
            'Doha'
        ],
        'Singapore Airlines': [
            'Singapore'
        ],
        'Cathay Pacific': [
            'Hong Kong'
        ],
        'Japan Airlines': [
            'Tokyo', 'Osaka'
        ],
        'All Nippon Airways': [
            'Tokyo', 'Osaka'
        ],
        'Ryanair': [
            'Dublin', 'London', 'Milan', 'Rome', 'Madrid', 'Barcelona'
        ],
        'EasyJet': [
            'London', 'Milan', 'Berlin', 'Geneva', 'Amsterdam'
        ],
    }


def match_city_names(wikipedia_city: str, route_cities: Set[str]) -> str:
    """
    Match Wikipedia city name to actual city name in route data.
    """
    wikipedia_city_lower = wikipedia_city.lower()
    
    # Direct match
    for city in route_cities:
        if city.lower() == wikipedia_city_lower:
            return city
    
    # Partial match
    for city in route_cities:
        if wikipedia_city_lower in city.lower() or city.lower() in wikipedia_city_lower:
            return city
    
    # Try matching airport codes or common variations
    return None


def main():
    """Main function to fetch official airline hub data."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("FETCHING OFFICIAL AIRLINE HUB DATA")
    print("=" * 70)
    
    # Load existing routes to get airline list and city names
    routes_path = data_dir / 'airline_routes_real.csv'
    if not routes_path.exists():
        print(f"\nError: {routes_path} not found.")
        print("Please run fetch_airline_routes.py first to get route data.")
        return
    
    routes_df = pd.read_csv(routes_path)
    airlines = sorted(routes_df['airline'].unique())
    all_cities = set(routes_df['origin'].unique()) | set(routes_df['destination'].unique())
    
    print(f"\nFound {len(airlines)} airlines in route data")
    print(f"Found {len(all_cities)} unique cities")
    
    # Get known official hubs
    known_hubs = get_known_official_hubs()
    print(f"\nUsing {len(known_hubs)} manually curated hub lists")
    
    # Build hub database
    hub_data = []
    scraped_count = 0
    total_airlines = len(airlines)
    
    print(f"\nProcessing {total_airlines} airlines...")
    print("(This may take a while - scraping Wikipedia for official hub data)\n")
    
    for idx, airline in enumerate(airlines, 1):
        hubs = []
        
        # Progress indicator
        if idx % 50 == 0 or idx == total_airlines:
            print(f"Progress: {idx}/{total_airlines} airlines processed ({scraped_count} scraped)")
        
        # First try known official hubs
        if airline in known_hubs:
            for hub_city in known_hubs[airline]:
                matched = match_city_names(hub_city, all_cities)
                if matched:
                    hubs.append(matched)
            if idx <= 25:  # Only print first 25 for curated ones
                print(f"  [{idx}/{total_airlines}] ✓ {airline}: {len(hubs)} official hubs (curated)")
        else:
            # Scrape Wikipedia for airlines not in curated list
            if HAS_SCRAPING_DEPS:
                if idx <= 100 or idx % 10 == 0:  # Print progress for first 100, then every 10th
                    print(f"  [{idx}/{total_airlines}] Scraping {airline}...", end=' ')
                wikipedia_hubs = get_wikipedia_hubs(airline)
                for hub_city in wikipedia_hubs:
                    matched = match_city_names(hub_city, all_cities)
                    if matched:
                        hubs.append(matched)
                scraped_count += 1
                if idx <= 100 or idx % 10 == 0:
                    if hubs:
                        print(f"✓ {len(hubs)} hubs")
                    else:
                        print("✗ no hubs")
                time.sleep(0.3)  # Rate limiting to be respectful to Wikipedia
            else:
                # Without scraping deps, skip
                pass
        
        # Deduplicate
        hubs = list(set(hubs))
        
        # Only add if we found at least one hub
        if hubs:
            for hub_city in hubs:
                hub_data.append({
                    'airline': airline,
                    'hub_city': hub_city,
                })
        else:
            print(f"  Warning: No hubs found for {airline}")
    
    # Create DataFrame
    hubs_df = pd.DataFrame(hub_data)
    
    # Save
    hubs_path = data_dir / 'airline_hubs_official.csv'
    hubs_df.to_csv(hubs_path, index=False)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Saved official hubs to: {hubs_path}")
    print(f"  {len(hubs_df)} hub assignments")
    print(f"  {hubs_df['airline'].nunique()} airlines with official hub data")
    print(f"\nTop 10 airlines by official hubs:")
    print(hubs_df['airline'].value_counts().head(10).to_string())
    
    print(f"\nNote: Only airlines with known official hub data are included.")
    print(f"      To add more airlines, expand the get_known_official_hubs() function")
    print(f"      with data from airline websites or Wikipedia.")


if __name__ == "__main__":
    main()

