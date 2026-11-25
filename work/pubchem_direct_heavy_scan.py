import json
import sys
import time
import urllib.parse
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "work"))
from pubchem_pv_scan import BASE, collect_properties, get_json, screen_record


CACHE = ROOT / "work" / "pubchem_direct_c_br_i_cache.json"
OUTPUT = ROOT / "outputs" / "pubchem_direct_heavy_candidates.json"
SMARTS = "[C;X4]-[Br,I]"
MAX_RECORDS = 10000


session = requests.Session()
session.headers["User-Agent"] = "QM-PV-direct-heavy-screen/1.0"

if CACHE.exists():
    records = json.loads(CACHE.read_text())
else:
    encoded = urllib.parse.quote(SMARTS, safe="")
    url = (
        f"{BASE}/compound/fastsubstructure/smarts/{encoded}/cids/JSON"
        f"?MaxRecords={MAX_RECORDS}"
    )
    cids = get_json(session, url)["IdentifierList"]["CID"]
    records = collect_properties(session, cids)
    CACHE.write_text(json.dumps(records))

candidates = []
seen = set()
for record in records:
    item = screen_record(record)
    if item is None or item["smiles"] in seen:
        continue
    if not any(atom["distance"] == 1 for atom in item["target_atoms"]):
        continue
    seen.add(item["smiles"])
    candidates.append(item)

candidates.sort(
    key=lambda item: (
        -sum(atom["distance"] == 1 for atom in item["target_atoms"]),
        -item["screen_score"],
        item["heavy_atoms"],
    )
)
OUTPUT.write_text(json.dumps(candidates, indent=2))

print(f"PubChem records read: {len(records):,}")
print(f"Direct C*-Br/I candidates: {len(candidates):,}")
print(f"Saved: {OUTPUT}")
print("=" * 120)
for rank, item in enumerate(candidates[:100], 1):
    direct = sum(atom["distance"] == 1 for atom in item["target_atoms"])
    cip = ",".join(center["cip"] for center in item["chiral_centers"])
    atoms = ", ".join(
        f"{atom['element']}(d={atom['distance']})" for atom in item["target_atoms"]
    )
    print(
        f"{rank:3d}. CID {item['cid']:<10d} | direct={direct} | "
        f"score={item['screen_score']:.5f} | {item['formula']:<18s} | "
        f"CIP={cip:<9s} | {atoms:<22s} | HA={item['heavy_atoms']:2d} "
        f"RB={item['rotatable_bonds']} | {item['name']}"
    )
