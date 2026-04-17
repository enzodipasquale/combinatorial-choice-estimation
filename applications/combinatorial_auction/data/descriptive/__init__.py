"""Descriptive figures and tables.

Outputs land in ``slides/artifacts/{figures,tables}/`` so they flow straight
into the Overleaf-linked slides repo on git push.
"""
from pathlib import Path

_REPO = Path(__file__).resolve().parents[4]
OUT_FIG = _REPO / "slides" / "artifacts" / "figures"
OUT_TAB = _REPO / "slides" / "artifacts" / "tables"
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)
