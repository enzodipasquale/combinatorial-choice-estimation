#!/usr/bin/env python3
import argparse, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from .tables import run
from .compute import PREFERRED

parser = argparse.ArgumentParser()
parser.add_argument("specs", nargs="*", default=PREFERRED)
run(parser.parse_args().specs)
