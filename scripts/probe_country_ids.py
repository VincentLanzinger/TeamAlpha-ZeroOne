"""One-off: list Sybilion region ids for the European countries we care about."""
from __future__ import annotations
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import sybilion_client as sc

WANT = {
    "Germany", "France", "Italy", "Spain", "Netherlands", "Belgium",
    "United Kingdom", "Austria", "Poland", "Norway", "Denmark", "Portugal",
    "Sweden", "Finland", "Ireland", "Greece", "Czech Republic", "Czechia",
    "Slovakia", "Hungary", "Romania", "Bulgaria",
    "European Union", "Europe", "World",
}

print("Fetching regions catalog (free, sync)...")
items = sc.list_regions()
print(f"got {len(items)} regions\n")

found = [r for r in items if r.get("name") in WANT]
found.sort(key=lambda r: r.get("name", ""))
print(f"matched {len(found)} of {len(WANT)} we wanted:\n")
print(f"  {'id':>5}  name")
print(f"  {'-'*5}  {'-'*30}")
for r in found:
    print(f"  {r['id']:>5}  {r['name']}")

missing = WANT - {r.get("name") for r in found}
if missing:
    print(f"\nmissing (not in catalog under these names): {sorted(missing)}")
