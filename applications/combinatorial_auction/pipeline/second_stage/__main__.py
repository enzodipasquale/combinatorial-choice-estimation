"""Entry point: python -m applications.combinatorial_auction.pipeline.second_stage SPEC..."""
import sys
from .tables import run

if __name__ == "__main__":
    specs = sys.argv[1:]
    if not specs:
        print("usage: python -m ...second_stage SPEC [SPEC ...]", file=sys.stderr)
        sys.exit(2)
    run(specs)
