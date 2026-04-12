# Airline / GS — Agent Brief

Read `SPEC.md` first. It is the source of truth for the economics and math. This document is how you build.

## Working principles

**Build incrementally.** Every change is verified at tiny scale before being used at larger scale. If a step can't be checked at tiny scale, that's a sign the step is too big — break it up.

**No HPC.** You do not have cluster access. Everything you run must fit on a single machine in reasonable time. The user launches HPC jobs separately.

**No silent edits to `combest/` core.** All scenario code lives in `scenarios/airline/`. If you believe `combest/` needs changes, raise it as a question; do not edit.

**Gurobi timeouts.** Any subproblem solver you touch must have a wall-clock timeout (`TimeLimit` parameter) during development and debugging. Default to 30s. A solve that takes longer than 30s at small scale is a bug, not a slow computer.

**Determinism.** Every random draw is seeded explicitly. DGP seed, simulation error seed, and any search seed are separate and logged.

## Build order

Do these in order. Do not proceed to step N+1 until step N passes its check.

### Step 1 — DGP skeleton
Implement geography, populations, hubs, covariate construction (both `fe_mode: "none"` and `fe_mode: "origin"`).

**Check:** run `python generate_data.py` with $K=4$ (so $M=12$), $N=3$. Print the covariate matrix and hub assignments. Inspect by eye: do covariates look sane, are hubs in $[K]$, does the covariate vector length match the advertised dimension under each `fe_mode`?

### Step 2 — Utility oracle (no greedy yet)
Implement the full utility function $V_i^\theta(b)$ as a pure function of $(b, \theta, \text{covariates}, \text{errors}, \text{hubs})$.

**Check:** at $K=4$, $N=3$, compute $V$ on three hand-picked bundles (empty, full, one arbitrary). Verify the congestion term by hand for the arbitrary bundle.

### Step 3 — Brute-force demand
Implement brute-force demand: enumerate all $2^M$ bundles, return the maximizer.

**Check:** at $K=3$ (so $M=6$, $2^M=64$), compute brute-force demand for 3 airlines. Print the result. Hand-verify at least one.

### Step 4 — Greedy demand, naive
Implement greedy with the existing naive oracle-based path (`_naive_greedy_solve` in `combest/subproblems/registry/greedy.py`).

**Check:** at $M \leq 5$, run greedy AND brute-force on 10 random airlines × 10 random $\theta$s × random errors. Assert greedy utility $\geq$ brute-force utility $- 10^{-8}$. This test must pass and must live in the test suite.

### Step 5 — Greedy demand, custom `find_best_item`
Implement the $O(M)$ `find_best_item` described in SPEC.md (maintains per-hub counts, constant-time marginal update).

**Check:** at $M \leq 5$, run both naive greedy AND `find_best_item`-greedy on the same 10 random instances from step 4. Bundles must be identical (or utilities equal within $10^{-8}$ if ties differ).

Additionally: at $M = 100$, time both naive and custom `find_best_item`. The custom version should be at least 10× faster. If not, the implementation is wrong.

### Step 6 — Healthy-DGP $\theta$ search
Implement the search procedure from SPEC.md.

**Check:** at $K=5$ ($M=20$), $N=30$, run the search under both `fe_mode` settings. The search must return a $\theta^*$ that passes all three healthy-DGP criteria. Log the candidates tried and the winning one.

### Step 7 — End-to-end pilot run
Wire up the `combest` estimator with the greedy solver, feed the healthy-DGP data, run one replication.

**Check (small):** $K=5$ ($M=20$), $N=30$. Runtime target: under 60 seconds. Estimated $\theta$ componentwise within 30% of true $\theta^*$ (this is a smoke test, not a scientific claim — it just has to work).

**Check (medium):** $K=10$ ($M=90$), $N=30$. Runtime target: under 10 minutes. Estimated $\theta$ within 30%.

Do NOT attempt $K \geq 30$ without explicit approval from the user. Large runs are for HPC.

### Step 8 — Write `result.json`
Produce the deliverable JSON per SPEC.md. Include everything listed there. Nothing more.

## File layout

```
scenarios/airline/
├── SPEC.md                  # source of truth (do not edit)
├── AGENT_BRIEF.md           # this file (do not edit)
├── config.yaml              # parameters (K, N, fe_mode, seeds, bounds)
├── generate_data.py         # geography, populations, hubs, covariates, healthy-θ search
├── oracle.py                # find_best_item + brute-force reference
├── run.py                   # end-to-end: DGP → estimation → result.json
├── tests/
│   ├── test_greedy_vs_brute.py   # step 4/5 verification, mandatory
│   └── test_find_best_item.py    # step 5 speed + equivalence
└── results/
    └── result.json          # deliverable
```

## Rules on failure modes

- A Gurobi solve exceeding its timeout: stop, report, do not increase the timeout to "make it work."
- Greedy disagreeing with brute force at $M \leq 5$: stop, debug, do not proceed.
- Healthy-DGP search failing to find a valid $\theta^*$ after, say, 100 candidates: stop, report search diagnostics, do not weaken the criteria.
- Estimation returning $\hat\theta$ far from $\theta^*$ at step 7 small-check: stop, report. Do not ship a failing estimator.

## Reporting

At each step, produce a short stdout log of what you did and what the check returned. If a check fails, halt and report. Do not silently retry with different parameters.

At the end, the deliverable is `result.json` plus passing tests. Not a narrative. Not a writeup.
