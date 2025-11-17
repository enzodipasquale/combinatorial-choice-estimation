# LaTeX Tables - Ready for Copy-Paste

## Combined Table
The main table file `tables_all.tex` contains all four experiments:
- Greedy (Gross Substitutes)
- Supermodular Network
- Linear Knapsack
- Quadratic Knapsack

## Individual Tables
Each experiment also has its own table in:
- `outputs/greedy_*/tables.tex`
- `outputs/supermod_*/tables.tex`
- `outputs/knapsack_*/tables.tex`
- `outputs/supermodknapsack_*/tables.tex`

## Usage
Simply copy the contents of `tables_all.tex` into your LaTeX document.
The tables use the `threeparttable` environment and require:
- `\usepackage{booktabs}`
- `\usepackage{threeparttable}`
