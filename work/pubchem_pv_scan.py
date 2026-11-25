from __future__ import annotations

import json
import math
import time
from pathlib import Path

import requests
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "work" / "pubchem_pv_cache.json"
OUTPUT = ROOT / "outputs" / "pubchem_pv_candidates.json"
BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Organic/main-group systems suitable for a first all-electron DIRAC screen.
ALLOWED_Z = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
PV_TARGET_Z = {35, 53}  # First public-database pass: Br and I.
MAX_RECORDS_PER_ELEMENT = 2000


def get_json(session: requests.Session, url: str, attempts: int = 6) -> dict:
    for attempt in range(attempts):
        response = session.get(url, timeout=180)
        if response.status_code == 200:
            return response.json()
        if response.status_code in {429, 503, 504}:
            time.sleep(2 ** attempt)
            continue
        response.raise_for_status()
    raise RuntimeError(f"PubChem did not answer after {attempts} attempts: {url}")


def collect_cids(session: requests.Session, symbol: str) -> list[int]:
    smarts = {"Br": "%5BBr%5D", "I": "%5BI%5D"}[symbol]
    url = (
        f"{BASE}/compound/fastsubstructure/smarts/{smarts}/cids/JSON"
        f"?MaxRecords={MAX_RECORDS_PER_ELEMENT}"
    )
    payload = get_json(session, url)
    return payload["IdentifierList"]["CID"]


def collect_properties(session: requests.Session, cids: list[int]) -> list[dict]:
    fields = (
        "Title,MolecularFormula,MolecularWeight,IsomericSMILES,CanonicalSMILES,"
        "Charge,HeavyAtomCount,RotatableBondCount,DefinedAtomStereoCount,"
        "UndefinedAtomStereoCount"
    )
    properties: list[dict] = []
    for start in range(0, len(cids), 100):
        batch = cids[start : start + 100]
        cid_text = ",".join(map(str, batch))
        url = f"{BASE}/compound/cid/{cid_text}/property/{fields}/JSON"
        payload = get_json(session, url)
        properties.extend(payload["PropertyTable"]["Properties"])
        time.sleep(0.12)
    return properties


def graph_distance_matrix(mol: Chem.Mol):
    return Chem.GetDistanceMatrix(mol)


def has_distinct_mirror_image(mol: Chem.Mol) -> bool:
    """Reject meso/achiral structures even when local R/S centers are present."""
    original = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    mirror = Chem.Mol(mol)
    inverted = 0
    for atom in mirror.GetAtoms():
        if atom.GetChiralTag() in {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        }:
            atom.InvertChirality()
            inverted += 1
    if inverted == 0:
        return False
    Chem.AssignStereochemistry(mirror, cleanIt=True, force=True)
    reflected = Chem.MolToSmiles(mirror, canonical=True, isomericSmiles=True)
    return original != reflected


def screen_record(record: dict) -> dict | None:
    smiles = record.get("SMILES") or record.get("IsomericSMILES")
    if not smiles:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if 6 not in {atom.GetAtomicNum() for atom in mol.GetAtoms()}:
        return None
    if any(atom.GetAtomicNum() not in ALLOWED_Z for atom in mol.GetAtoms()):
        return None
    if Chem.GetFormalCharge(mol) != 0:
        return None

    heavy_atoms = mol.GetNumHeavyAtoms()
    rotors = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if not 4 <= heavy_atoms <= 25 or rotors > 5:
        return None

    chiral = Chem.FindMolChiralCenters(
        mol,
        includeUnassigned=False,
        includeCIP=True,
        useLegacyImplementation=False,
    )
    if not chiral or len(chiral) > 4:
        return None
    if not has_distinct_mirror_image(mol):
        return None

    target_atoms = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() in PV_TARGET_Z
    ]
    if not target_atoms:
        return None

    chiral_atoms = [idx for idx, _ in chiral]
    distances = graph_distance_matrix(mol)
    target_details = []
    atomic_signal = 0.0
    for atom_idx in target_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        z = atom.GetAtomicNum()
        d = int(min(distances[atom_idx, center] for center in chiral_atoms))
        # Ranking heuristic only: Z^5 weighting and exponential graph-distance decay.
        contribution = (z / 53.0) ** 5 * math.exp(-0.70 * d)
        atomic_signal += contribution
        target_details.append({"element": atom.GetSymbol(), "distance": d})

    pair_bonus = 1.0
    if len(target_atoms) >= 2:
        nearest_pair = min(
            distances[a, b]
            for i, a in enumerate(target_atoms)
            for b in target_atoms[i + 1 :]
        )
        pair_bonus += 0.25 * math.exp(-0.20 * float(nearest_pair))

    ring_atoms = sum(atom.IsInRing() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    ring_fraction = ring_atoms / heavy_atoms
    rigidity = (1.0 + 0.35 * ring_fraction) / (1.0 + 0.22 * rotors)
    size_penalty = math.exp(-0.035 * max(0, heavy_atoms - 8))
    score = atomic_signal * pair_bonus * rigidity * size_penalty

    return {
        "cid": int(record["CID"]),
        "name": record.get("Title", ""),
        "formula": record.get("MolecularFormula", ""),
        "molecular_weight": float(record.get("MolecularWeight", 0.0)),
        "smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "heavy_atoms": heavy_atoms,
        "rotatable_bonds": int(rotors),
        "chiral_centers": [{"atom": int(i), "cip": label} for i, label in chiral],
        "target_atoms": target_details,
        "screen_score": float(score),
    }


def main() -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "QM-PV-candidate-screen/1.0"

    if CACHE.exists():
        records = json.loads(CACHE.read_text())
    else:
        cids = sorted(set(collect_cids(session, "I") + collect_cids(session, "Br")))
        records = collect_properties(session, cids)
        CACHE.write_text(json.dumps(records))

    screened = []
    seen_smiles = set()
    for record in records:
        candidate = screen_record(record)
        if candidate is None or candidate["smiles"] in seen_smiles:
            continue
        seen_smiles.add(candidate["smiles"])
        screened.append(candidate)

    screened.sort(key=lambda item: item["screen_score"], reverse=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(screened, indent=2))

    print(f"PubChem records read: {len(records):,}")
    print(f"Candidates after chemical filters: {len(screened):,}")
    print(f"Saved: {OUTPUT}")
    print("\nTOP 30 PRELIMINARY CANDIDATES")
    print("=" * 110)
    for rank, item in enumerate(screened[:30], 1):
        targets = ", ".join(
            f"{x['element']}(d={x['distance']})" for x in item["target_atoms"]
        )
        configurations = ",".join(x["cip"] for x in item["chiral_centers"])
        print(
            f"{rank:2d}. CID {item['cid']:<9d} | score={item['screen_score']:.5f} | "
            f"{item['formula']:<18s} | CIP={configurations:<7s} | {targets:<20s} | "
            f"HA={item['heavy_atoms']:2d} RB={item['rotatable_bonds']} | {item['name']}"
        )


if __name__ == "__main__":
    main()
