"""
Script to generate bidder_data.csv from FCC Form 175 data.

This script replicates the data generation process from the original 
retreive_data.ipynb notebook (commit 76d07914d47d77c2440ad206755d46a5e0fda500).

The bidder_data.csv combines:
1. biddercblk_03_28_2004_pln.csv from 114402-V1 (bidder_num_fox, bidder_num, co_name, pops_eligible)
2. FCC Form 175 data (City, State, Applicant_Status, Legal_Classification)
3. BTA mapping from county/state using btacnty1990.txt

EXTERNAL DATA REQUIREMENTS (not in repo):
- FCC Form 175 .txt files (pipe-delimited APP records)
- btacnty1990.txt (FCC BTA-county mapping file)

Usage:
    python retrieve_bidder_data.py --form175-dir /path/to/form175_data --bta-mapping /path/to/btacnty1990.txt
    python retrieve_bidder_data.py --verify  # Only verify existing bidder_data.csv
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "114402-V1" / "Replication-Fox-and-Bajari" / "data"
OUTPUT_FILE = SCRIPT_DIR / "bidder_data.csv"


def process_app_records(file_path: str) -> pd.DataFrame:
    """Process FCC Form 175 APP records from a pipe-delimited file."""
    df = pd.read_csv(file_path, delimiter="|", header=None, dtype=str)
    app_records = df[df[0] == "APP"].reset_index(drop=True)

    app_columns = [
        "Record_Type", "Applicant_FRN", "Auction_ID", "File_Number", "Applicant_Name",
        "First_Name", "Middle_Name", "Last_Name", "Suffix", "State_Or_Citizenship",
        "Applicant_Status", "Legal_Classification", "Reserved1", "Reserved2", "Reserved3",
        "Noncommercial_Status", "Address1", "Address2", "City", "State", "Zip_Code",
        "Country", "Bidding_Option", "Bidding_Credit", "New_Entrant_Credit",
        "Gross_Revenue_Lower", "Gross_Revenue_Upper", "Closed_Bidding_Eligibility",
        "Bidding_Credit_Percentage", "Certifier_First_Name", "Certifier_Middle_Initial",
        "Certifier_Last_Name", "Certifier_Suffix", "Certifier_Title", "Prior_Defaulter",
        "Financial_Statement_Type", "Gross_Revenue_Most_Recent", "Recent_Year_End",
        "Gross_Revenue_One_Year_Prior", "One_Year_Prior_End", "Gross_Revenue_Two_Years_Prior",
        "Two_Years_Prior_End", "Total_Assets", "Aggregate_Gross_Revenues",
        "Aggregate_Gross_Revenues_1", "Aggregate_Gross_Revenues_2", "Aggregate_Total_Assets",
        "Aggregate_Total_Assets_1", "Aggregate_Total_Assets_2", "Financial_Statement",
        "Aggregate_Financial_Statement", "Aggregate_Credits"
    ]

    app_records.columns = app_columns
    return app_records


def load_form175_data(folder_path: str) -> pd.DataFrame:
    """Load and concatenate all Form 175 .txt files from a folder."""
    all_dataframes = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            df = process_app_records(file_path)
            all_dataframes.append(df)
    return pd.concat(all_dataframes, ignore_index=True)


def get_county_and_state(zip_code: str):
    """Get county and state from zip code using pgeocode."""
    import pgeocode
    nomi = pgeocode.Nominatim('us')
    location = nomi.query_postal_code(str(zip_code))
    if pd.isna(location.county_name) or pd.isna(location.state_code):
        return None, None
    return location.county_name, location.state_code


# Manual name corrections: Form 175 names that don't match biddercblk names
NAME_CORRECTIONS = {
    0: 'Betty A. Gleaton',
    1: 'HARVEY LEONG',
    2: 'William Ingram',
    3: 'ELIZABETH R. GUEST',
    4: 'Mark M. Guest',
    5: 'Vincent  D. McBride',
    6: 'ADILIA M. AGUILAR',
    7: 'Shawn Capistrano',
    8: 'GLENN ISHIHARA',
    9: 'Harold L. Sudbury, Jr.'
}

# County name corrections for BTA mapping
COUNTY_CORRECTIONS = {
    'City and County of San Francisco': 'San Francisco',
    'Anchorage Municipality': 'Anchorage',
    'St. Louis (city)': 'St. Louis',
    'City of Alexandria': 'Alexandria City',
    'East Baton Rouge Parish': 'East Baton Rouge',
    'Waynesboro (city)': 'Waynesboro City',
    'Jefferson Parish': 'Jefferson',
    'Western Connecticut': 'Fairfield',
    'Capitol Region': 'Hartford',
}


def generate_bidder_data(form175_dir: str, bta_mapping_path: str) -> pd.DataFrame:
    """Generate bidder_data.csv from Form 175 data and 114402-V1 data."""
    import pgeocode  # noqa: F401 - verify pgeocode is available

    # Load Form 175 data
    print("Loading Form 175 data...")
    form175_df = load_form175_data(form175_dir)
    form175_df = form175_df.drop_duplicates().reset_index(drop=True)

    # Load bidder data from 114402-V1
    print("Loading biddercblk data from 114402-V1...")
    bidder_data = pd.read_csv(DATA_DIR / "biddercblk_03_28_2004_pln.csv")

    # Create co_name column and apply corrections
    form175_df = form175_df.copy()
    form175_df['co_name'] = form175_df['Applicant_Name']

    # Find names that don't match and apply corrections
    index_to_update = form175_df[~form175_df['Applicant_Name'].isin(bidder_data['co_name'])].index
    for i, idx in enumerate(index_to_update):
        if i in NAME_CORRECTIONS:
            form175_df.loc[idx, 'co_name'] = NAME_CORRECTIONS[i]

    # Get county from zip code
    print("Looking up counties from zip codes...")
    form175_df['Zip_Code_cleaned'] = form175_df['Zip_Code'].str[:5]
    counties, states = zip(*form175_df['Zip_Code_cleaned'].apply(get_county_and_state))
    form175_df['County'] = counties
    form175_df['State_from_zipcode'] = states

    # Apply county corrections
    for old_name, new_name in COUNTY_CORRECTIONS.items():
        form175_df.loc[form175_df['County'] == old_name, 'County'] = new_name

    # Keep relevant columns
    form175_df = form175_df.dropna(axis=1, how='all')
    relevant_columns = ['co_name', 'City', 'State', 'County', 'Applicant_Status', 'Legal_Classification']
    form175_df = form175_df[relevant_columns]

    # Load BTA mapping and merge
    print("Mapping counties to BTAs...")
    county_bta_mapping = pd.read_csv(bta_mapping_path, encoding='latin1')
    merged = form175_df.merge(
        county_bta_mapping[['County', 'State', 'BTA']],
        on=['County', 'State'],
        how='left'
    )
    form175_df['bta'] = merged['BTA']

    # Manual BTA assignments for GU and PR
    form175_df.loc[form175_df['State'] == 'GU', 'bta'] = 490
    form175_df.loc[form175_df['City'] == 'Mercedita', 'bta'] = 489
    form175_df.loc[form175_df['City'] == 'SAN JUAN', 'bta'] = 488
    form175_df['bta'] = form175_df['bta'].astype(int)

    # Merge with bidder data from 114402-V1
    print("Merging with bidder data...")
    final_df = bidder_data.merge(
        form175_df[['co_name', 'bta', 'City', 'State', 'Applicant_Status', 'Legal_Classification']],
        on='co_name',
        how='left'
    )

    return final_df


def verify_bidder_data() -> bool:
    """Verify that bidder_data.csv matches expected structure and content."""
    if not OUTPUT_FILE.exists():
        print(f"ERROR: {OUTPUT_FILE} does not exist")
        return False

    print(f"Verifying {OUTPUT_FILE}...")
    df = pd.read_csv(OUTPUT_FILE)

    # Check columns
    expected_columns = ['bidder_num_fox', 'bidder_num', 'co_name', 'pops_eligible',
                        'bta', 'City', 'State', 'Applicant_Status', 'Legal_Classification']
    if list(df.columns) != expected_columns:
        print(f"ERROR: Column mismatch. Expected {expected_columns}, got {list(df.columns)}")
        return False
    print(f"  ✓ Columns match: {expected_columns}")

    # Check row count
    expected_rows = 256  # 255 bidders + header
    if len(df) != 255:
        print(f"ERROR: Row count mismatch. Expected 255, got {len(df)}")
        return False
    print(f"  ✓ Row count: {len(df)}")

    # Check base data matches 114402-V1 (excluding dummy FCC row)
    base_data = pd.read_csv(DATA_DIR / "biddercblk_03_28_2004_pln.csv")
    # Filter out dummy FCC row (bidder_num_fox=256, co_name='FCC')
    base_data = base_data[base_data['co_name'] != 'FCC'].reset_index(drop=True)

    # Check non-numeric columns exactly
    for col in ['bidder_num_fox', 'bidder_num', 'co_name']:
        if not df[col].equals(base_data[col]):
            print(f"ERROR: Column '{col}' doesn't match biddercblk_03_28_2004_pln.csv")
            return False

    # Check pops_eligible (NaN in base may be 0 in df)
    base_pops = base_data['pops_eligible'].fillna(0)
    df_pops = df['pops_eligible'].fillna(0)
    if not np.allclose(df_pops.values, base_pops.values):
        print("ERROR: pops_eligible doesn't match biddercblk_03_28_2004_pln.csv")
        return False
    print(f"  ✓ Base columns match biddercblk_03_28_2004_pln.csv (excluding FCC dummy row)")

    # Check BTA values are valid (1-493 for continental US, plus territories)
    if df['bta'].isna().any():
        print("ERROR: Some BTA values are missing")
        return False
    if not ((df['bta'] >= 1) & (df['bta'] <= 493)).all():
        print("ERROR: BTA values out of expected range")
        return False
    print(f"  ✓ BTA values valid (range: {df['bta'].min()}-{df['bta'].max()})")

    # Check for expected territories
    gu_count = (df['State'] == 'GU').sum()
    pr_count = df['State'].isin(['PR']).sum() + (df['bta'].isin([488, 489])).sum()
    print(f"  ✓ Territory coverage: GU={gu_count}, PR-related BTAs present")

    print("\n✓ Verification PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Generate or verify bidder_data.csv")
    parser.add_argument('--form175-dir', type=str, help='Path to folder with Form 175 .txt files')
    parser.add_argument('--bta-mapping', type=str, help='Path to btacnty1990.txt BTA mapping file')
    parser.add_argument('--verify', action='store_true', help='Only verify existing bidder_data.csv')
    parser.add_argument('--output', type=str, default=str(OUTPUT_FILE), help='Output file path')

    args = parser.parse_args()

    if args.verify:
        success = verify_bidder_data()
        exit(0 if success else 1)

    if not args.form175_dir or not args.bta_mapping:
        print("ERROR: --form175-dir and --bta-mapping are required for generation")
        print("\nExternal data requirements:")
        print("  1. FCC Form 175 .txt files (pipe-delimited APP records)")
        print("  2. btacnty1990.txt (FCC BTA-county mapping file)")
        print("\nThese files are not included in the repository.")
        print("Use --verify to check existing bidder_data.csv instead.")
        exit(1)

    df = generate_bidder_data(args.form175_dir, args.bta_mapping)

    # Compare with existing if it exists
    if OUTPUT_FILE.exists():
        existing = pd.read_csv(OUTPUT_FILE)
        if df.equals(existing):
            print("\n✓ Generated data matches existing bidder_data.csv")
        else:
            print("\n⚠ Generated data differs from existing bidder_data.csv")
            print("Differences:")
            for col in df.columns:
                if not df[col].equals(existing[col]):
                    print(f"  - Column '{col}' differs")

    # Save
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
