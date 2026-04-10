# Deterministic DGP Proposal

## Interface constraint

`generate_data.py` must return the same `(geo, firms, bundles, theta_true)` 
tuple with the same field names and shapes as today.  
`costs.py`, `solver.py`, `oracles.py`, `dc.py`, `milp.py` — **zero changes**.

Fields that `geo` must supply (from `costs.py`):

| field | shape | consumed by |
|---|---|---|
| `L1`, `L2`, `n_markets` | scalars | everywhere |
| `cont1` | (L1,) int | `compute_rev_factor`, `compute_facility_costs` |
| `cont2` | (L2,) int | same |
| `cont_m` | (N,) int | `build_firms` feasibility |
| `ln_d_12` | (L1, L2) | `compute_rev_factor` |
| `ln_d_2m` | (L2, N) | `compute_rev_factor` |
| `ln_d_hq1` | (3, L1) | `compute_facility_costs` |
| `ln_d_hq2` | (3, L2) | `compute_facility_costs` |
| `tau_12` | (L1, L2) | `compute_rev_factor` |
| `tau_2m` | (L2, N) | `compute_rev_factor` |
| `R_n` | (N,) | `build_firm_milp` |

Fields that each `firm` dict must supply (unchanged):
`hq_cont`, `n_models`, `cell_groups`, `n_platforms`, `platforms`,
`feasible`, `shares`, `ln_xi_1`, `ln_xi_2`.

---

## 1. Cell locations  (L1 = 12)

Continent indices: Am=0, As=1, Eu=2.  
`cont1[l]` gives the continent of cell l.

| idx | name | lat | lon | continent |
|---|---|---|---|---|
| 0 | AM_C0 Detroit, MI | 42.33 | −83.05 | Am (0) |
| 1 | AM_C1 Monterrey, MX | 25.67 | −100.32 | Am (0) |
| 2 | AM_C2 Greenville-Spartanburg, SC | 34.90 | −82.40 | Am (0) |
| 3 | AM_C3 São Paulo, BR | −23.55 | −46.63 | Am (0) |
| 4 | AS_C0 Wuhan, CN | 30.59 | 114.30 | As (1) |
| 5 | AS_C1 Shanghai, CN | 31.23 | 121.47 | As (1) |
| 6 | AS_C2 Busan, KR | 35.10 | 129.04 | As (1) |
| 7 | AS_C3 Nagoya, JP | 35.18 | 136.91 | As (1) |
| 8 | AS_C4 Bangkok, TH | 13.75 | 100.52 | As (1) |
| 9 | EU_C0 Stuttgart, DE | 48.78 | 9.18 | Eu (2) |
| 10 | EU_C1 Katowice, PL | 50.26 | 19.02 | Eu (2) |
| 11 | EU_C2 Gothenburg, SE | 57.71 | 12.00 | Eu (2) |

Rationale: real EV/battery production hubs. Am=4, As=5, Eu=3 — matches
current `l1_per_continent: [4, 5, 3]`.

---

## 2. Assembly locations  (L2 = 15)

`cont2[l]` gives continent of assembly site l.

| idx | name | lat | lon | continent |
|---|---|---|---|---|
| 0 | AM_A0 Los Angeles, CA | 34.05 | −118.24 | Am (0) |
| 1 | AM_A1 Chicago, IL | 41.88 | −87.63 | Am (0) |
| 2 | AM_A2 Mexico City, MX | 19.43 | −99.13 | Am (0) |
| 3 | AM_A3 Houston, TX | 29.76 | −95.37 | Am (0) |
| 4 | AM_A4 Toronto, CA | 43.70 | −79.42 | Am (0) |
| 5 | AS_A0 Shenzhen, CN | 22.54 | 114.06 | As (1) |
| 6 | AS_A1 Tianjin, CN | 39.13 | 117.20 | As (1) |
| 7 | AS_A2 Incheon, KR | 37.46 | 126.71 | As (1) |
| 8 | AS_A3 Osaka, JP | 34.69 | 135.50 | As (1) |
| 9 | AS_A4 Ho Chi Minh City, VN | 10.82 | 106.63 | As (1) |
| 10 | EU_A0 Rotterdam, NL | 51.92 | 4.48 | Eu (2) |
| 11 | EU_A1 Munich, DE | 48.14 | 11.58 | Eu (2) |
| 12 | EU_A2 Poznan, PL | 52.41 | 16.93 | Eu (2) |
| 13 | EU_A3 Zaragoza, ES | 41.65 | −0.89 | Eu (2) |
| 14 | EU_A4 Turku, FI | 60.45 | 22.27 | Eu (2) |

5 per continent → `l2_per_continent: [5, 5, 5]` (unchanged).

---

## 3. Markets  (N = 24)

`cont_m[n]` gives continent, Am=0..7, As=8..15, Eu=16..23.

### Expenditure weights
5 "top" markets (1 Am + 2 As + 2 Eu) each receive **14%** of total.  
19 "rest" markets each receive **30%/19 ≈ 1.58%**.  
Total sums to 1.0 (normalized; scale up in code if desired).

Top markets: Am_M0 (New York), As_M0 (Shanghai), As_M1 (Tokyo),
Eu_M0 (London), Eu_M1 (Paris).

| idx | name | lat | lon | cont | R weight |
|---|---|---|---|---|---|
| 0 | New York, US | 40.71 | −74.01 | Am | **0.1400** |
| 1 | Los Angeles, US | 34.05 | −118.24 | Am | 0.0158 |
| 2 | Chicago, US | 41.88 | −87.63 | Am | 0.0158 |
| 3 | Houston, US | 29.76 | −95.37 | Am | 0.0158 |
| 4 | Mexico City, MX | 19.43 | −99.13 | Am | 0.0158 |
| 5 | Toronto, CA | 43.70 | −79.42 | Am | 0.0158 |
| 6 | São Paulo, BR | −23.55 | −46.63 | Am | 0.0158 |
| 7 | Miami, US | 25.77 | −80.19 | Am | 0.0158 |
| 8 | Shanghai, CN | 31.23 | 121.47 | As | **0.1400** |
| 9 | Tokyo, JP | 35.68 | 139.69 | As | **0.1400** |
| 10 | Beijing, CN | 39.91 | 116.39 | As | 0.0158 |
| 11 | Seoul, KR | 37.57 | 126.98 | As | 0.0158 |
| 12 | Delhi, IN | 28.64 | 77.22 | As | 0.0158 |
| 13 | Bangkok, TH | 13.75 | 100.52 | As | 0.0158 |
| 14 | Singapore, SG | 1.35 | 103.82 | As | 0.0158 |
| 15 | Jakarta, ID | −6.21 | 106.85 | As | 0.0158 |
| 16 | London, GB | 51.51 | −0.13 | Eu | **0.1400** |
| 17 | Paris, FR | 48.86 | 2.35 | Eu | **0.1400** |
| 18 | Berlin, DE | 52.52 | 13.41 | Eu | 0.0158 |
| 19 | Madrid, ES | 40.42 | −3.70 | Eu | 0.0158 |
| 20 | Milan, IT | 45.47 | 9.19 | Eu | 0.0158 |
| 21 | Stockholm, SE | 59.33 | 18.07 | Eu | 0.0158 |
| 22 | Warsaw, PL | 52.23 | 21.01 | Eu | 0.0158 |
| 23 | Amsterdam, NL | 52.37 | 4.90 | Eu | 0.0158 |

---

## 4. HQ reference points  (3 points, one per continent)

Used to compute `ln_d_hq1[c, l]` and `ln_d_hq2[c, l2]` in `costs.py`.  
Placed at an iconic site in each continent — not at a cell or assembly node.

| cont | name | lat | lon |
|---|---|---|---|
| 0 Am | Detroit, MI (same as AM_C0) | 42.33 | −83.05 |
| 1 As | Wuhan, CN (same as AS_C0) | 30.59 | 114.30 |
| 2 Eu | Stuttgart, DE (same as EU_C0) | 48.78 | 9.18 |

(Re-using cell index 0 of each continent as the HQ anchor is clean and
makes `rho_HQ_*` identifiable: distance from the Eu HQ to Katowice or
Gothenburg is clearly non-zero, while Stuttgart→Stuttgart = 0.)

---

## 5. Distance computation

```
haversine_km(a, b):
    φ1, λ1 = radians(a.lat), radians(a.lon)
    φ2, λ2 = radians(b.lat), radians(b.lon)
    dφ = φ2 − φ1;  dλ = λ2 − λ1
    h = sin²(dφ/2) + cos(φ1)·cos(φ2)·sin²(dλ/2)
    return 2 · 6371 · arcsin(√h)          # km

ln_d_12[l1, l2] = log(1 + haversine_km(cell[l1], asm[l2]) / 1000)
ln_d_2m[l2, n]  = log(1 + haversine_km(asm[l2], mkt[n])  / 1000)
ln_d_hq1[c, l1] = log(1 + haversine_km(hq[c],   cell[l1]) / 1000)
ln_d_hq2[c, l2] = log(1 + haversine_km(hq[c],   asm[l2])  / 1000)
```

Dividing by 1000 keeps units in "thousands of km"; log(1+x) is well-behaved
at x=0.  Typical intercontinental distance: LA↔Shanghai ≈ 9,100 km → 2.32.

---

## 6. Tariff rules

```
TARIFF_MATRIX[from_cont, to_cont]:
    (Am, Am) = 0.00   (As, As) = 0.00   (Eu, Eu) = 0.00
    (As, Am) = 0.25   (As, Eu) = 0.25   # Asia → West: high
    (Am, As) = 0.10   (Eu, As) = 0.10   # West → Asia: moderate
    (Am, Eu) = 0.10   (Eu, Am) = 0.10   # transatlantic: moderate

tau_12[l1, l2] = TARIFF_MATRIX[cont1[l1], cont2[l2]]
tau_2m[l2, n]  = TARIFF_MATRIX[cont2[l2], cont_m[n]]
```

No randomness. Every seed sees the same tariff structure.

---

## 7. Firm structure  (15 firms: 6 big + 9 small)

### Big firms  (2 per continent, indices 0–5)

| firm idx | hq_cont | HQ marker |
|---|---|---|
| 0 | Am (0) | HQ[0] = Detroit |
| 1 | Am (0) | HQ[0] = Detroit |
| 2 | As (1) | HQ[1] = Wuhan |
| 3 | As (1) | HQ[1] = Wuhan |
| 4 | Eu (2) | HQ[2] = Stuttgart |
| 5 | Eu (2) | HQ[2] = Stuttgart |

- `n_models = 12`
- `n_platforms = 4`
- `models→platform`: round-robin [0,1,2,3, 0,1,2,3, 0,1,2,3]
- `models→cell_group`: first 6 in group 0, last 6 in group 1
  → `cell_groups = [0,0,0,0,0,0, 1,1,1,1,1,1]`
- `feasible[m, n] = True` for all n (sells in all 24 markets)

### Small firms  (3 per continent, indices 6–14)

| firm indices | hq_cont |
|---|---|
| 6, 7, 8 | Am (0) |
| 9, 10, 11 | As (1) |
| 12, 13, 14 | Eu (2) |

- `n_models = 5`
- `n_platforms = 2`
- `models→platform`: [0,0,1,1,0]
- `models→cell_group`: [0,0,1,1,0]  (3 in group 0, 2 in group 1)
- `feasible[m, n] = True` only if `cont_m[n] == hq_cont`  
  (sells only in home continent's 8 markets)

---

## 8. Quality index ξ  (deterministic)

```
ln_xi_1[firm, g]:         # shape: (n_groups_cells,) = (2,)
    size_bonus = +0.5 if big else −0.5
    group_bonus = +0.2 if g == 0 else −0.2
    ln_xi_1[g] = size_bonus + group_bonus

    Big firm:  [+0.7, +0.3]
    Small firm: [−0.3, −0.7]

ln_xi_2[firm, p]:         # shape: (n_platforms,) = (4,) or (2,)
    Big firm (4 platforms):  [+0.6, +0.2, −0.2, −0.6]
    Small firm (2 platforms): [+0.2, −0.2]
```

---

## 9. Market shares  s°_{m,n}  (deterministic)

EV penetration: `s_ICE = 0.85`, so EV share budget per market = 0.15.

Step 1 — unnormalized weight per (model m of firm i, market n):
```
weight[i, m, n] = exp(home_bonus · 1[cont_i == cont_m[n]]
                      + size_bonus · log(firm_size_i))
    home_bonus = 1.5
    size_bonus = 0.5
    firm_size  = n_models  (12 for big, 5 for small)
```

Step 2 — normalize across all (firm, model) pairs active in market n:
```
W[n] = Σ_{i,m: feasible[i,m,n]} weight[i, m, n]
shares[i, m, n] = (1 − s_ICE) · weight[i, m, n] / W[n]
               = 0.15 · weight[i, m, n] / W[n]
```

Shares are zero for infeasible (m, n) pairs.

### Sanity check (approximate, for home Am market)

Active pairs in a home-Am market:
- 2 big Am firms × 12 models: weight = exp(1.5 + 0.5·ln 12) = exp(1.5+1.24) ≈ 15.5 each → 2·12·15.5 = 372
- 2 big As firms × 12 models: weight = exp(0 + 0.5·ln 12) = exp(1.24) ≈ 3.46 → 2·12·3.46 = 83
- 2 big Eu firms × 12 models: same as As → 83
- 3 small Am firms × 5 models: weight = exp(1.5 + 0.5·ln 5) = exp(1.5+0.80) ≈ 10.0 → 3·5·10.0 = 150
- Small As/Eu firms: infeasible → 0
- Total W ≈ 372 + 83 + 83 + 150 = 688

Market shares (× 0.15):
- Big Am firm total (12 models): 0.15 × 12·15.5/688 ≈ 4.0% per firm
- Big As/Eu firm total: 0.15 × 12·3.46/688 ≈ 0.9% per firm
- Small Am firm total (5 models): 0.15 × 5·10.0/688 ≈ 1.1% per firm
- Grand total: 2×4.0 + 4×0.9 + 3×1.1 = 8 + 3.6 + 3.3 = 14.9% ≈ 15% ✓

---

## 10. Errors  (only random element, cross-seed variation)

Unchanged from current code:
```python
phi1[i, :ng, :L1] ~ N(0, sigma['phi_1'])
phi2[i, :P,  :L2] ~ N(0, sigma['phi_2'])
nu[i, :nm, :N, :L1, :L2] ~ N(0, sigma['nu'])
```
Seeded with `(err_seed, agent_global_id)` — one RNG per agent, 
fresh from DGP errors (current correct implementation).

---

## 11. config.yaml changes

The `dgp:` block becomes metadata only (new generate_data.py ignores it 
for geography/firms, uses hardcoded tables). Fields to keep for 
documentation purposes:

```yaml
dgp:
  n_firms: 15            # 6 big + 9 small
  n_markets: 24          # 8 per continent
  n_continents: 3
  l1_per_continent: [4, 5, 3]
  l2_per_continent: [5, 5, 5]
  n_groups_cells: 2
  n_platforms: 4         # was 8; big=4, small=2, max=4
  models_range: [12, 5]  # big=12, small=5  (no longer a range, just docs)
```

`sigma`, `sourcing_coefs`, `theta_true`, `estimation`, `monte_carlo` — unchanged.

---

## 12. What this fixes

| problem | before | after |
|---|---|---|
| Geography varies by seed | random scatter | fixed tables |
| Tariffs vary by seed | uniform random [0.05,0.25] | rule-based matrix |
| Shares vary by seed | Dirichlet + random EV fraction | deterministic weight rule |
| Firm structure varies by seed | random n_models, cell groups, platforms | fixed per tier |
| Quality ξ varies by seed | N(0,0.5) draws | deterministic formula |
| Identification degeneracies | random orthogonality failures | designed-in variation |

---

## Open questions for review

1. **home_bonus=1.5, size_bonus=0.5**: are these reasonable, or would you prefer
   a different parameterization of the share rule?

2. **HQ = cell-0 of each continent**: should big firms within the same continent
   have distinct HQ coordinates (e.g., firm 0 → Detroit, firm 1 → another Am cell)?
   Currently `ln_d_hq1[c, :]` is continent-level, so both Am big firms share one
   HQ distance vector. If you want firm-level HQ, we'd need to expand
   `ln_d_hq1` to shape (n_firms, L1) — which requires a small change in
   `compute_facility_costs`. Flag this before coding.

3. **São Paulo as Am cell**: it is geographically in South America. If the
   experiment is meant to be "North America", swap it for, say, 
   Atlanta, GA (33.75, −84.39) or Kansas City, MO (39.10, −94.58).

4. **n_platforms in config.yaml**: currently 8, new DGP uses 4. This affects
   array sizing in `run_experiment.py` (`P_max`). Fine to change, just flagging.
