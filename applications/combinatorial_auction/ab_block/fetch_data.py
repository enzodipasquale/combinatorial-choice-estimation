#!/usr/bin/env python3
"""Download and process FCC Auction 4 (Broadband PCS A/B block) data.

Source: Penn State Center for Auctions, Procurements and Competition Policy
        https://capcp.la.psu.edu/data-and-software/fcc-spectrum-auction-data/

Auction 4 ran Dec 5 1994 – Mar 13 1995.  99 MTA-level licenses (51 MTAs × 2
frequency blocks, minus 3 A-block pioneer's-preference awards in M001, M002,
M010).  18 winners out of 31 qualified bidders.  $7.0B gross revenue.

Outputs (in data/):
    winning_bids.csv  – 99 rows: license, mta, block, winner, price, population
    bidders.csv       – 31 rows: fcc_acct, name, eligibility (in population)
"""

import io, zipfile, urllib.request
from pathlib import Path

import pandas as pd

URL = ("https://capcp.la.psu.edu/wp-content/uploads/sites/11/"
       "fcc-spectrum-auction-data/FCC_Auction04e.zip")

OUT_DIR = Path(__file__).parent / "data"


def fetch_excel():
    """Download zip → return openpyxl Workbook (read-only)."""
    print(f"Downloading {URL} ...")
    resp = urllib.request.urlopen(URL)
    with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
        xlsx_name = [n for n in zf.namelist() if n.endswith(".xlsx")][0]
        return pd.ExcelFile(io.BytesIO(zf.read(xlsx_name)), engine="openpyxl")


def build_winning_bids(xls):
    wb = xls.parse("WinningBids")
    mk = xls.parse("Markets")

    pop = mk.set_index("LICENSE_NAME")["POPULATION"]
    mta_num = {f"M{i:03d}": i for i in range(1, 52)}

    df = pd.DataFrame({
        "license":    wb["LICENSE_NAME"].values,
        "mta":        wb["MARKET"].values,
        "block":      wb["FREQ_BLOCK"].values,
        "fcc_acct":   wb["FCC_ACCT"].astype(str).str.lstrip("0").astype(int).values,
        "winner":     wb["BIDDER_NAME"].values,
        "price":      wb["BID_AMNT"].astype(int).values,
        "population": wb["LICENSE_NAME"].map(pop).values,
        "mta_num":    wb["MARKET"].map(mta_num).values,
    })
    return df.sort_values("mta_num")


def build_bidders(xls):
    bd = xls.parse("Bidders")
    return pd.DataFrame({
        "fcc_acct":    bd["FCC_ACCT"].values,
        "name":        bd["BIDDER_NAME"].values,
        "eligibility": bd["BID_UNITS"].astype(int).values,
    })


def main():
    xls = fetch_excel()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    wb = build_winning_bids(xls)
    wb.to_csv(OUT_DIR / "winning_bids.csv", index=False)
    print(f"  winning_bids.csv: {len(wb)} licenses, "
          f"{wb['mta'].nunique()} MTAs, {wb['winner'].nunique()} winners")

    bd = build_bidders(xls)
    bd.to_csv(OUT_DIR / "bidders.csv", index=False)
    print(f"  bidders.csv: {len(bd)} bidders")


if __name__ == "__main__":
    main()
