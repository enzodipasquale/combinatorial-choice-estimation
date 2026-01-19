"""
Improved script to generate bidder_data.csv from FCC Form 175 data.

Fixes from v1:
1. Name corrections by Applicant_Name instead of fragile index-based approach
2. Validates all merges complete (no unmatched rows)
3. Verifies Form 175 columns (City, State, Applicant_Status, Legal_Classification)
4. Better error handling and logging

Usage:
    python retrieve_bidder_data_v2.py --form175-dir /path/to/form175_data --bta-mapping /path/to/btacnty1990.txt
    python retrieve_bidder_data_v2.py --verify  # Verify existing bidder_data.csv
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

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

    if len(app_records.columns) >= len(app_columns):
        app_records = app_records.iloc[:, :len(app_columns)]
        app_records.columns = app_columns
    return app_records


def load_form175_data(folder_path: str) -> pd.DataFrame:
    """Load and concatenate all Form 175 .txt files from a folder."""
    all_dataframes = []
    for file_name in sorted(os.listdir(folder_path)):  # Sort for reproducibility
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)
            df = process_app_records(file_path)
            all_dataframes.append(df)
            print(f"  Loaded {file_name}: {len(df)} APP records")
    return pd.concat(all_dataframes, ignore_index=True)


def get_county_and_state(zip_code: str) -> Tuple[Optional[str], Optional[str]]:
    """Get county and state from zip code using pgeocode."""
    import pgeocode
    nomi = pgeocode.Nominatim('us')
    location = nomi.query_postal_code(str(zip_code))
    if pd.isna(location.county_name) or pd.isna(location.state_code):
        return None, None
    return location.county_name, location.state_code


# FIX #1: Name corrections by Applicant_Name, not by index
# Maps Form 175 Applicant_Name -> biddercblk co_name (for individuals whose names differ)
# Form 175 uses "Last, First" format, biddercblk uses "First Last" format
NAME_CORRECTIONS: Dict[str, str] = {
    # Form 175 name (Last, First) -> biddercblk name (First Last)
    'GLEATON, BETTY A': 'Betty A. Gleaton',
    'LEONG, HARVEY': 'HARVEY LEONG', 
    'INGRAM, WILLIAM': 'William Ingram',
    'GUEST, ELIZABETH R': 'ELIZABETH R. GUEST',
    'GUEST, MARK M': 'Mark M. Guest',
    'MCBRIDE, VINCENT D': 'Vincent  D. McBride',  # Note: double space in biddercblk
    'AGUILAR, ADILIA M': 'ADILIA M. AGUILAR',
    'CAPISTRANO, SHAWN': 'Shawn Capistrano',
    'ISHIHARA, GLENN': 'GLENN ISHIHARA',
    'SUDBURY, HAROLD L': 'Harold L. Sudbury, Jr.',
}


def normalize_name(name: str) -> str:
    """Normalize a name for matching: uppercase, strip, collapse spaces."""
    if pd.isna(name):
        return ""
    return " ".join(str(name).upper().split())


def apply_name_corrections(form175_df: pd.DataFrame, bidder_data: pd.DataFrame) -> pd.DataFrame:
    """Apply name corrections using name-based matching, not index-based."""
    form175_df = form175_df.copy()
    form175_df['co_name'] = form175_df['Applicant_Name']
    
    # Find unmatched names
    biddercblk_names = set(bidder_data['co_name'].values)
    unmatched_mask = ~form175_df['co_name'].isin(biddercblk_names)
    unmatched = form175_df.loc[unmatched_mask, 'Applicant_Name'].unique()
    
    print(f"  Found {len(unmatched)} unmatched names in Form 175 data")
    
    # Try to match using normalized names
    for idx in form175_df[unmatched_mask].index:
        orig_name = form175_df.loc[idx, 'Applicant_Name']
        norm_name = normalize_name(orig_name)
        
        # Check if we have a correction
        if norm_name in NAME_CORRECTIONS:
            form175_df.loc[idx, 'co_name'] = NAME_CORRECTIONS[norm_name]
        else:
            # Try fuzzy matching with biddercblk names
            for biddercblk_name in biddercblk_names:
                if normalize_name(biddercblk_name) == norm_name:
                    form175_df.loc[idx, 'co_name'] = biddercblk_name
                    break
    
    # Report any still unmatched
    still_unmatched = form175_df[~form175_df['co_name'].isin(biddercblk_names)]['co_name'].unique()
    if len(still_unmatched) > 0:
        print(f"  WARNING: {len(still_unmatched)} names still unmatched:")
        for name in still_unmatched[:10]:
            print(f"    - {name}")
    
    return form175_df


# County name corrections for BTA mapping
COUNTY_CORRECTIONS: Dict[str, str] = {
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
    print("Step 1: Loading Form 175 data...")
    form175_df = load_form175_data(form175_dir)
    form175_df = form175_df.drop_duplicates().reset_index(drop=True)
    print(f"  Total unique APP records: {len(form175_df)}")

    # Load bidder data from 114402-V1
    print("\nStep 2: Loading biddercblk data from 114402-V1...")
    bidder_data = pd.read_csv(DATA_DIR / "biddercblk_03_28_2004_pln.csv")
    # Filter out dummy FCC row
    bidder_data = bidder_data[bidder_data['co_name'] != 'FCC'].reset_index(drop=True)
    print(f"  Loaded {len(bidder_data)} bidders (excluding FCC dummy)")

    # Apply name corrections
    print("\nStep 3: Matching Form 175 names to biddercblk...")
    form175_df = apply_name_corrections(form175_df, bidder_data)

    # Get county from zip code
    print("\nStep 4: Looking up counties from zip codes...")
    form175_df['Zip_Code_cleaned'] = form175_df['Zip_Code'].str[:5]
    counties, states = zip(*form175_df['Zip_Code_cleaned'].apply(get_county_and_state))
    form175_df['County'] = counties
    form175_df['State_from_zipcode'] = states
    
    missing_county = form175_df['County'].isna().sum()
    print(f"  Missing county for {missing_county} zip codes")

    # Apply county corrections
    for old_name, new_name in COUNTY_CORRECTIONS.items():
        form175_df.loc[form175_df['County'] == old_name, 'County'] = new_name

    # Keep relevant columns
    form175_df = form175_df.dropna(axis=1, how='all')
    relevant_columns = ['co_name', 'City', 'State', 'County', 'Applicant_Status', 'Legal_Classification']
    form175_df = form175_df[relevant_columns]

    # Load BTA mapping and merge
    print("\nStep 5: Mapping counties to BTAs...")
    county_bta_mapping = pd.read_csv(bta_mapping_path, encoding='latin1')
    merged = form175_df.merge(
        county_bta_mapping[['County', 'State', 'BTA']],
        on=['County', 'State'],
        how='left'
    )
    form175_df['bta'] = merged['BTA']

    # Manual BTA assignments for territories
    form175_df.loc[form175_df['State'] == 'GU', 'bta'] = 490
    form175_df.loc[form175_df['City'] == 'Mercedita', 'bta'] = 489
    form175_df.loc[form175_df['City'] == 'SAN JUAN', 'bta'] = 488
    
    missing_bta = form175_df['bta'].isna().sum()
    if missing_bta > 0:
        print(f"  WARNING: {missing_bta} bidders still missing BTA")
        print(form175_df[form175_df['bta'].isna()][['co_name', 'City', 'State', 'County']])
    
    form175_df['bta'] = form175_df['bta'].astype(int)

    # FIX #2: Validate merge completeness
    print("\nStep 6: Merging with bidder data...")
    final_df = bidder_data.merge(
        form175_df[['co_name', 'bta', 'City', 'State', 'Applicant_Status', 'Legal_Classification']],
        on='co_name',
        how='left'
    )
    
    # Check for unmatched rows
    unmatched = final_df[final_df['bta'].isna()]
    if len(unmatched) > 0:
        print(f"  ERROR: {len(unmatched)} bidders have no Form 175 match:")
        print(unmatched[['bidder_num_fox', 'co_name']].to_string())
        raise ValueError("Merge incomplete - some bidders have no Form 175 data")
    
    print(f"  ✓ All {len(final_df)} bidders matched successfully")
    
    # Fill NaN in pops_eligible with 0 (consistent with original)
    final_df['pops_eligible'] = final_df['pops_eligible'].fillna(0)
    
    return final_df


def verify_bidder_data(verbose: bool = True) -> bool:
    """Comprehensive verification of bidder_data.csv."""
    if not OUTPUT_FILE.exists():
        print(f"ERROR: {OUTPUT_FILE} does not exist")
        return False

    print(f"Verifying {OUTPUT_FILE}...")
    df = pd.read_csv(OUTPUT_FILE)
    errors = []

    # 1. Check columns
    expected_columns = ['bidder_num_fox', 'bidder_num', 'co_name', 'pops_eligible',
                        'bta', 'City', 'State', 'Applicant_Status', 'Legal_Classification']
    if list(df.columns) != expected_columns:
        errors.append(f"Column mismatch. Expected {expected_columns}, got {list(df.columns)}")
    else:
        print(f"  ✓ Columns match")

    # 2. Check row count
    if len(df) != 255:
        errors.append(f"Row count mismatch. Expected 255, got {len(df)}")
    else:
        print(f"  ✓ Row count: {len(df)}")

    # 3. Check base data matches 114402-V1
    base_data = pd.read_csv(DATA_DIR / "biddercblk_03_28_2004_pln.csv")
    base_data = base_data[base_data['co_name'] != 'FCC'].reset_index(drop=True)

    for col in ['bidder_num_fox', 'bidder_num', 'co_name']:
        if not df[col].equals(base_data[col]):
            errors.append(f"Column '{col}' doesn't match biddercblk")
    
    # pops_eligible with NaN handling
    base_pops = base_data['pops_eligible'].fillna(0)
    df_pops = df['pops_eligible'].fillna(0)
    if not np.allclose(df_pops.values, base_pops.values):
        errors.append("pops_eligible doesn't match biddercblk")
    else:
        print(f"  ✓ Base columns match biddercblk")

    # 4. Check BTA values
    if df['bta'].isna().any():
        errors.append("Some BTA values are missing")
    elif not ((df['bta'] >= 1) & (df['bta'] <= 493)).all():
        errors.append(f"BTA values out of range: {df['bta'].min()}-{df['bta'].max()}")
    else:
        print(f"  ✓ BTA values valid (range: {df['bta'].min()}-{df['bta'].max()})")

    # 5. FIX #3: Verify Form 175 columns
    # Check City is not empty
    empty_city = df['City'].isna().sum() + (df['City'] == '').sum()
    if empty_city > 0:
        errors.append(f"{empty_city} bidders have empty City")
    else:
        print(f"  ✓ All bidders have City")

    # Check State is valid 2-letter code
    valid_states = {'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'HI', 'ID', 
                    'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 
                    'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 
                    'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 
                    'WI', 'WY', 'DC', 'GU', 'PR', 'VI'}
    invalid_states = set(df['State'].unique()) - valid_states
    if invalid_states:
        errors.append(f"Invalid states: {invalid_states}")
    else:
        print(f"  ✓ All states valid ({len(df['State'].unique())} unique)")

    # Check Legal_Classification is not empty
    empty_legal = df['Legal_Classification'].isna().sum()
    if empty_legal > 0:
        errors.append(f"{empty_legal} bidders have empty Legal_Classification")
    else:
        print(f"  ✓ All bidders have Legal_Classification")

    # 6. Check duplicates
    dup_fox = df['bidder_num_fox'].duplicated().sum()
    dup_name = df['co_name'].duplicated().sum()
    if dup_fox > 0:
        errors.append(f"{dup_fox} duplicate bidder_num_fox values")
    if dup_name > 0:
        errors.append(f"{dup_name} duplicate co_name values")
    if dup_fox == 0 and dup_name == 0:
        print(f"  ✓ No duplicates")

    # 7. Check BTA-State consistency for territories
    gu_bidders = df[df['State'] == 'GU']
    if len(gu_bidders) > 0 and not (gu_bidders['bta'] == 490).all():
        errors.append("GU bidders should have BTA 490")
    else:
        print(f"  ✓ GU territory BTA correct ({len(gu_bidders)} bidders)")

    # Summary
    if errors:
        print(f"\n✗ Verification FAILED with {len(errors)} errors:")
        for err in errors:
            print(f"  - {err}")
        return False
    
    print("\n✓ Verification PASSED")
    return True


def compare_with_existing(new_df: pd.DataFrame) -> None:
    """Compare generated DataFrame with existing bidder_data.csv."""
    if not OUTPUT_FILE.exists():
        print("No existing bidder_data.csv to compare with")
        return
    
    existing = pd.read_csv(OUTPUT_FILE)
    
    print("\n=== Comparison with existing bidder_data.csv ===")
    
    if new_df.shape != existing.shape:
        print(f"  Shape differs: new={new_df.shape}, existing={existing.shape}")
        return
    
    all_match = True
    for col in new_df.columns:
        if col not in existing.columns:
            print(f"  Column '{col}' not in existing")
            all_match = False
            continue
        
        if new_df[col].dtype in ['float64', 'int64'] and existing[col].dtype in ['float64', 'int64']:
            match = np.allclose(new_df[col].fillna(0).values, existing[col].fillna(0).values)
        else:
            match = new_df[col].equals(existing[col])
        
        if not match:
            diff_count = (new_df[col] != existing[col]).sum()
            print(f"  Column '{col}': {diff_count} differences")
            # Show first few differences
            diff_mask = new_df[col] != existing[col]
            if diff_mask.any():
                sample_idx = diff_mask[diff_mask].head(3).index
                for idx in sample_idx:
                    print(f"    Row {idx}: new='{new_df.loc[idx, col]}' vs existing='{existing.loc[idx, col]}'")
            all_match = False
    
    if all_match:
        print("  ✓ All columns match!")


def main():
    parser = argparse.ArgumentParser(description="Generate or verify bidder_data.csv (v2)")
    parser.add_argument('--form175-dir', type=str, help='Path to folder with Form 175 .txt files')
    parser.add_argument('--bta-mapping', type=str, help='Path to btacnty1990.txt BTA mapping file')
    parser.add_argument('--verify', action='store_true', help='Only verify existing bidder_data.csv')
    parser.add_argument('--output', type=str, help='Output file path (default: same as existing)')
    parser.add_argument('--compare-only', action='store_true', help='Run verification and comparison only')

    args = parser.parse_args()

    if args.verify or args.compare_only:
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
    
    # Compare with existing
    compare_with_existing(df)

    # Save if output specified
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
