# Application: NAIC Schedule D — US Life Insurance Corporate Bond Portfolios

## One-line summary

Structurally estimate US life insurers' joint inclusion + weight decisions over corporate bonds, using NAIC Schedule D holdings data and NAIC Risk-Based Capital (RBC) designations as the regulatory constraint.

## Scope of this spec

This document covers **only the data assembly and descriptive sanity-check phase**. No modeling, no estimation, no oracles. Modeling will be specified in a subsequent SPEC after the data is reviewed and the application's economic preconditions are verified.

The numerical scenario at `paper/numerical_experiments/scenarios/portfolio/` is the reference for combest conventions, the $(s, w)$ encoding, and the framework-edit context. Modeling code in this application will follow the same patterns when it is written.

## Economic setting (for context, not for implementation in this phase)

- **Agents:** US life insurers, one observation per insurer-year.
- **Items:** corporate bonds (CUSIPs) in a defined universe.
- **Choice:** $(s_i, w_i)$ with $s_{ij} \in \{0,1\}$ inclusion and $w_{ij} \in \mathbb{R}_+$ portfolio share.
- **Constraint:** RBC capital charge $\sum_j w_{ij} \cdot \text{rbc}(\text{NAIC}_j) \leq C_i$, plus state-law concentration limits (NY §1405 and analogs).
- **Why combinatorial:** insurers run buy-and-hold portfolios with real per-CUSIP underwriting, monitoring, and accounting costs. Holdings are sparse relative to the universe. The NAIC-2 / NAIC-3 (IG / HY) cliff in RBC creates a discrete regulatory bucket.

References for the audience: Becker & Ivashina (2015, JF), Ellul, Jotikasthira & Lundblad (2011, JFE), Ge & Weisbach (2021, JFE), Murray & Nikolova (2022, JF).

## Data sources

- **NAIC Schedule D Part 1** (annual): CUSIP-level fixed-income holdings per insurer per year. Access via WRDS (NAIC database) or S&P Global Market Intelligence.
- **Mergent FISD** (or TRACE): bond-level characteristics — coupon, maturity, issue size, callable indicator, sector, rating history.
- **NAIC designation file**: NAIC 1–6 designation per CUSIP per year (in Schedule D itself, or via NAIC SVO files).
- **Insurer-level financials**: from NAIC Annual Statement (admitted assets, capital, surplus). Available on WRDS.

## Sample restrictions (Phase 1)

- **Years:** start with one year, 2019 (post-LCR-implementation noise, pre-COVID).
- **Insurer type:** US life insurers only (drop P&C, health, fraternal).
- **Insurer size:** general account admitted assets $\geq \$1\text{B}$ (filters to ~150–250 insurers).
- **Bond universe:** US-domiciled corporate bonds, single-sector restriction (start with industrial corporates, NAICS 2-digit code 31–33 or equivalent), investment-grade and below.
- **Aggregation:** consolidate by NAIC group code where multiple subsidiaries report (one observation per group, not per legal entity).

These restrictions are starting points and may be revised after seeing the data.

## Phase 1 deliverables

The agent produces a `data/` directory with:

### `prepare.py`

Pulls and assembles the sample. Produces a single output file (parquet or HDF5) with three tables:

1. **Holdings table:** one row per (insurer-group, CUSIP, year). Columns: `group_id`, `cusip`, `year`, `par_value`, `book_value`, `market_value`, `naic_designation`.
2. **Bond characteristics table:** one row per (CUSIP, year). Columns: `cusip`, `year`, `coupon`, `maturity_date`, `issue_size`, `callable`, `sector`, `naic_designation`, `yield_at_year_end`, `duration`.
3. **Insurer table:** one row per (group, year). Columns: `group_id`, `year`, `admitted_assets`, `capital_and_surplus`, `total_corporate_bond_holdings`.

### `descriptives.ipynb`

Notebook producing the following sanity checks. Each is a table or figure with a one-paragraph interpretation.

1. **Sample size:** N insurers, N CUSIPs in the universe, N (insurer, CUSIP) holdings.
2. **Portfolio sparsity:** distribution of $|S_i|$ (number of distinct CUSIPs held per insurer). Histogram and summary statistics. *Pass criterion:* mean $|S_i|$ should be substantially less than $M$ (universe size). If insurers hold most of the universe, the extensive margin is degenerate and the application is dead.
3. **Within-portfolio concentration:** distribution of largest position $\max_j w_{ij}$ and Herfindahl across $w_i$. *Read:* are weights highly skewed (a few large positions) or roughly equal?
4. **NAIC designation composition:** for each insurer, share of portfolio in each NAIC designation (1–6). Cross-tabulate by insurer size. *Read:* is there meaningful variation across insurers in designation mix? If everyone is 100% NAIC-1, the RBC constraint isn't binding.
5. **RBC constraint slack:** computed RBC charge from holdings vs. insurer's available capital. Distribution of slack across insurers. *Read:* is the constraint actually binding for a non-trivial share of insurers?
6. **Cross-insurer overlap:** for each pair of insurers, fraction of CUSIPs in common. *Read:* do insurers hold similar or different bonds? High overlap suggests common preferences or common constraints; low overlap suggests heterogeneity.
7. **Universe coverage:** for each CUSIP in the bond universe, number of insurers holding it. *Read:* is the universe well-covered or is there a long tail of bonds nobody holds?
8. **Data quality:** missing-data rates per column, CUSIP match rates between Schedule D and FISD, count of dropped observations and reason.

## Stop here

After Phase 1 is complete, do not proceed to modeling. Report findings and wait for review. The decision to write a Phase 2 modeling spec depends on what the descriptives show.

## Acceptance criterion for Phase 1

- `prepare.py` runs end-to-end on a single year and produces the three tables.
- `descriptives.ipynb` runs end-to-end and produces all eight sanity checks.
- A short written summary (in the notebook or a separate markdown) flags whether the four economic preconditions hold:
  1. Sparsity: portfolios are small relative to the universe.
  2. Constraint binding: RBC slack is small for a non-trivial share of insurers.
  3. Heterogeneity: insurers differ meaningfully in their portfolio composition.
  4. Universe coverage: the chosen universe is well-supported by the data.

If any precondition fails, document which and why. Do not attempt to fix it by silently changing sample restrictions.

## Constraints on execution

- Local execution only. No HPC, no SLURM.
- Use the existing combest virtualenv or a separate venv as needed for WRDS pulls (WRDS has its own Python tooling).
- Do not modify `combest/` or any other application directory.
- All code lives under `applications/portfolio/`.
- Reference `paper/numerical_experiments/scenarios/portfolio/` for combest conventions when the modeling phase begins; do not reference it during Phase 1.

## Files to create in Phase 1

```
applications/portfolio/
├── SPEC.md                    # this file
├── README.md                  # short overview, how to run prepare.py and the notebook
└── data/
    ├── prepare.py             # WRDS pull + sample construction
    ├── descriptives.ipynb     # eight sanity checks
    └── output/                # parquet/HDF5 files (gitignored)
```
