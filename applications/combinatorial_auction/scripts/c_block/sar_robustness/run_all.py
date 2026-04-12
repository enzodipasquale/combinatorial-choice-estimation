#!/usr/bin/env python3
"""
Run all four SAR bootstrap estimations sequentially in a single MPI job.

Usage:
    mpiexec --bind-to none <python-wrapper> \
        python /path/to/sar_robustness/run_all.py
"""
import gc, sys, traceback
from pathlib import Path

SAR_DIR   = Path(__file__).parent
REPO_ROOT = SAR_DIR.parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from mpi4py import MPI
    comm, rank = MPI.COMM_WORLD, MPI.COMM_WORLD.Get_rank()
except ImportError:
    comm, rank = None, 0

from applications.combinatorial_auction.scripts.c_block.sar_robustness.run import main

CONFIGS = [
    (SAR_DIR / "configs" / f"config_sar_rho{suffix}.yaml", rho_str)
    for suffix, rho_str in [("00", "0.0"), ("02", "0.2"), ("04", "0.4"), ("06", "0.6")]
]

completed = []
failed    = []

for config_path, rho_str in CONFIGS:
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Starting rho={rho_str}  [{config_path.name}]")
        print(f"{'='*70}\n", flush=True)

    try:
        main(str(config_path))
        completed.append(rho_str)
    except Exception:
        if rank == 0:
            print(f"\n[ERROR] rho={rho_str} failed with exception:", flush=True)
            traceback.print_exc()
        failed.append(rho_str)

    # GC pass on all ranks before barrier
    gc.collect()

    # All ranks sync before the next estimation
    if comm is not None:
        comm.Barrier()

    if rank == 0:
        # Current RSS from /proc (Linux); more accurate than ru_maxrss which is peak-only
        try:
            rss_mb = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1024
            print(f"[rank 0] current RSS after rho={rho_str}: {rss_mb:.0f} MB", flush=True)
        except Exception:
            pass
        status = "Finished" if rho_str in completed else "FAILED"
        print(f"\n{'='*70}")
        print(f"{status} rho={rho_str}")
        print(f"{'='*70}\n", flush=True)

if rank == 0:
    print("\n" + "="*70)
    print("SAR run_all summary")
    print("="*70)
    if completed:
        print(f"  Completed: rho in {{{', '.join(completed)}}}")
    if failed:
        print(f"  FAILED:    rho in {{{', '.join(failed)}}}")
    print("="*70, flush=True)
