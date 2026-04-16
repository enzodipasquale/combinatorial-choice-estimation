# Unit demand probit benchmark

Benchmarks the combinatorial estimator against simulated probit MLE (GHK)
for three simulation counts: S ∈ {1, √N, N}.

## Running on HPC

```bash
sbatch run_S1.slurm      # S=1, lightweight (1 node)
sbatch run_Ssqrt.slurm   # S=√N, moderate (1 node, 32 ranks)
sbatch run_SN.slurm      # S=N, heavy (16 nodes, 512 ranks)
```

Each script is a SLURM array job over N ∈ {200, 500, 1000}, running both
J=2 and J=10 per task.  Results land in `results/raw/{one,sqrt_n,match_n}/`.

## Generating tables and figures

After all jobs finish:

```bash
python analyze_results.py
```

Produces `results/table.tex` (consolidated three-S table) and
`results/efficiency.pdf`.

## Local testing

```bash
python run_hpc.py --s-mode one --N 200       # fast: S=1
python run_hpc.py --s-mode sqrt_n --N 200    # moderate
python run_hpc.py --s-mode match_n --N 200   # heavy
```
