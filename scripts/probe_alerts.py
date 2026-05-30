"""One-off probe: see what /alerts returns under different query shapes."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import sybilion_client as sc, config

spec = config.active_ticker()

probes = [
    ("baseline TTF, no filters",
     dict(title=spec.metadata_title, description=spec.metadata_description,
          keywords=list(spec.keywords))),
    ("baseline TTF, EU + Energy + Commodities",
     dict(title=spec.metadata_title, description=spec.metadata_description,
          keywords=list(spec.keywords), regions=[3], categories=[25, 46])),
    ("Hormuz shock title, broad",
     dict(title=spec.metadata_title + " -- Hormuz oil and LNG supply disruption scenario",
          description=spec.metadata_description + " Shock context: Strait of Hormuz tanker disruption.",
          keywords=list(spec.keywords) + ["Hormuz", "oil supply shock", "tanker", "Middle East"])),
    ("Hormuz shock title, EU + Energy",
     dict(title=spec.metadata_title + " -- Hormuz oil and LNG supply disruption scenario",
          description=spec.metadata_description + " Shock context: Strait of Hormuz tanker disruption.",
          keywords=list(spec.keywords) + ["Hormuz", "oil supply shock", "tanker", "Middle East"],
          regions=[3, 1], categories=[25, 46, 17])),
]

for label, kwargs in probes:
    alerts = sc.get_alerts(context_enriched=True, limit=15, **kwargs)
    print(f"\n=== {label} ===  -> {len(alerts)} alerts")
    for a in alerts[:8]:
        pct = a.get("pct_change")
        try:
            pct_s = f"{float(pct):+6.2f}%"
        except Exception:
            pct_s = "    n/a"
        print(f"  {pct_s}  trending={bool(a.get('trending'))}  {a.get('name')}")
