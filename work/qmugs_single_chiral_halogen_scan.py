from __future__ import annotations

import json
import math
import time
from pathlib import Path

import duckdb
import requests
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, QED, rdMolDescriptors


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "qmugs_single_chiral_br_cl_candidates.json"
PARQUET_URLS = [
    f"https://huggingface.co/api/datasets/JuIm/QMugs_Summary/parquet/default/train/{i}.parquet"
    for i in range(3)
]

SUMMARY_COLUMNS = [
    "chembl_id",
    "conf_id",
    "smiles",
    "mw",
    "heavy_atoms",
    "rotatable_bonds",
    "stereocenters",
    "rings",
    "hbond_acceptors",
    "hbond_donors",
    "significant_negative_wavenumbers",
    "GFN2_TOTAL_ENERGY",
    "GFN2_DIPOLE_TOT",
    "GFN2_HOMO_ENERGY",
    "GFN2_LUMO_ENERGY",
    "GFN2_HOMO_LUMO_GAP",
    "GFN2_POLARIZABILITY_MOLECULAR",
    "DFT_DIPOLE_TOT",
    "DFT_HOMO_ENERGY",
    "DFT_LUMO_ENERGY",
    "DFT_HOMO_LUMO_GAP",
]


def finite_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def resolve_cdn_urls(urls):
    """Resolve each Hub API URL once so range reads go directly to the CDN."""
    resolved = []
    session = requests.Session()
    session.headers["User-Agent"] = "QMugs-PV-screen/1.0"
    for url in urls:
        for attempt in range(10):
            response = session.get(url, allow_redirects=True, stream=True, timeout=120)
            if response.status_code == 200:
                # response.url is the final signed CDN address after all redirects.
                resolved.append(response.url)
                response.close()
                break
            if response.status_code == 429:
                delay = min(int(response.headers.get("Retry-After", "30")), 60)
                response.close()
                print(f"QMugs rate-limit cooldown: {delay} seconds ...", flush=True)
                time.sleep(delay)
                continue
            response.raise_for_status()
        else:
            raise RuntimeError(f"Could not resolve QMugs file: {url}")
    return resolved


def examine_structure(row: dict) -> dict | None:
    smiles = row.get("smiles")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        return None
    if Chem.GetFormalCharge(mol) != 0:
        return None

    # Require exactly one specified tetrahedral centre and no additional
    # unspecified tetrahedral centres.
    all_centres = Chem.FindMolChiralCenters(
        mol,
        includeUnassigned=True,
        includeCIP=True,
        useLegacyImplementation=False,
    )
    assigned = [(idx, label) for idx, label in all_centres if label in {"R", "S"}]
    if len(all_centres) != 1 or len(assigned) != 1:
        return None

    centre_idx, cip = assigned[0]
    centre_atom = mol.GetAtomWithIdx(centre_idx)
    if centre_atom.GetAtomicNum() != 6 or centre_atom.GetHybridization() != Chem.HybridizationType.SP3:
        return None

    halogen_indices = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() in {17, 35}
    ]
    if not halogen_indices:
        return None

    distances = Chem.GetDistanceMatrix(mol)
    halogens = []
    for atom_idx in halogen_indices:
        atom = mol.GetAtomWithIdx(atom_idx)
        distance = int(distances[centre_idx, atom_idx])
        if distance <= 3:
            halogens.append({"element": atom.GetSymbol(), "distance": distance})
    if not halogens:
        return None

    heavy_atoms = mol.GetNumHeavyAtoms()
    rotors = rdMolDescriptors.CalcNumRotatableBonds(mol)
    molecular_weight = Descriptors.MolWt(mol)
    qed = QED.qed(mol)
    clogp = Crippen.MolLogP(mol)

    # No Rule-of-Five filter. This is only a generous computational-size cap
    # so that subsequent all-electron DIRAC calculations remain practical.
    if not (4 <= heavy_atoms <= 45) or rotors > 15:
        return None

    atomic_signal = sum(
        (35.0 if atom["element"] == "Br" else 17.0) ** 5
        * math.exp(-0.80 * (atom["distance"] - 1))
        for atom in halogens
    ) / (35.0**5)
    ring_fraction = sum(a.IsInRing() for a in mol.GetAtoms() if a.GetAtomicNum() > 1) / heavy_atoms
    rigidity = (1.0 + 0.30 * ring_fraction) / (1.0 + 0.18 * rotors)
    size_penalty = math.exp(-0.025 * max(0, heavy_atoms - 12))
    screen_score = atomic_signal * rigidity * size_penalty

    return {
        "chembl_id": row["chembl_id"],
        "conf_id": row["conf_id"],
        "smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": molecular_weight,
        "heavy_atoms": heavy_atoms,
        "rotatable_bonds": rotors,
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "qed": qed,
        "clogp": clogp,
        "chiral_center": {"atom": centre_idx, "cip": cip},
        "nearby_halogens": halogens,
        "minimum_halogen_distance": min(atom["distance"] for atom in halogens),
        "screen_score": screen_score,
        "significant_negative_wavenumbers": bool(row.get("significant_negative_wavenumbers")),
        "GFN2_TOTAL_ENERGY": finite_float(row.get("GFN2_TOTAL_ENERGY")),
        "GFN2_DIPOLE_TOT": finite_float(row.get("GFN2_DIPOLE_TOT")),
        "GFN2_HOMO_ENERGY": finite_float(row.get("GFN2_HOMO_ENERGY")),
        "GFN2_LUMO_ENERGY": finite_float(row.get("GFN2_LUMO_ENERGY")),
        "GFN2_HOMO_LUMO_GAP": finite_float(row.get("GFN2_HOMO_LUMO_GAP")),
        "GFN2_POLARIZABILITY_MOLECULAR": finite_float(row.get("GFN2_POLARIZABILITY_MOLECULAR")),
        "DFT_DIPOLE_TOT": finite_float(row.get("DFT_DIPOLE_TOT")),
        "DFT_HOMO_ENERGY": finite_float(row.get("DFT_HOMO_ENERGY")),
        "DFT_LUMO_ENERGY": finite_float(row.get("DFT_LUMO_ENERGY")),
        "DFT_HOMO_LUMO_GAP": finite_float(row.get("DFT_HOMO_LUMO_GAP")),
    }


def main():
    # Keep the lowest-GFN2-energy conformer for each ChEMBL molecule.
    best_by_id = {}
    total_rows = 0
    prefiltered_rows = 0

    print("Querying the three QMugs Parquet shards ...", flush=True)
    connection = duckdb.connect()
    connection.execute("SET allow_asterisks_in_http_paths = true")
    cdn_urls = resolve_cdn_urls(PARQUET_URLS)
    parquet_list = ", ".join(repr(url) for url in cdn_urls)
    selected_columns = ", ".join(f'"{column}"' for column in SUMMARY_COLUMNS)
    query = f"""
        SELECT {selected_columns}
        FROM read_parquet([{parquet_list}])
        WHERE stereocenters = 1
          AND (contains(smiles, 'Br') OR contains(smiles, 'Cl'))
    """
    reader = connection.execute(query).to_arrow_reader(batch_size=50000)

    for batch in reader:
        total_rows += batch.num_rows
        prefiltered_rows += batch.num_rows
        for row in batch.to_pylist():
            candidate = examine_structure(row)
            if candidate is None:
                continue
            key = candidate["chembl_id"]
            energy = candidate["GFN2_TOTAL_ENERGY"]
            previous = best_by_id.get(key)
            previous_energy = previous["GFN2_TOTAL_ENERGY"] if previous else None
            if previous is None or (
                energy is not None
                and (previous_energy is None or energy < previous_energy)
            ):
                best_by_id[key] = candidate

    candidates = list(best_by_id.values())
    candidates.sort(
        key=lambda item: (
            item["minimum_halogen_distance"],
            -item["screen_score"],
            item["heavy_atoms"],
        )
    )
    OUTPUT.write_text(json.dumps(candidates, indent=2))

    print(f"QMugs rows transferred after server-side filtering: {total_rows:,}")
    print(f"Rows with stereocenters=1 and Br/Cl: {prefiltered_rows:,}")
    print(f"Unique filtered molecules: {len(candidates):,}")
    print(f"Saved: {OUTPUT}")
    print("=" * 125)
    for rank, item in enumerate(candidates[:100], 1):
        halogens = ", ".join(
            f"{atom['element']}(d={atom['distance']})"
            for atom in item["nearby_halogens"]
        )
        print(
            f"{rank:3d}. {item['chembl_id']:<14s} | score={item['screen_score']:.5f} | "
            f"{item['formula']:<20s} | {item['chiral_center']['cip']} | {halogens:<28s} | "
            f"HA={item['heavy_atoms']:2d} RB={item['rotatable_bonds']} QED={item['qed']:.3f} | "
            f"DFT gap={item['DFT_HOMO_LUMO_GAP']}"
        )


if __name__ == "__main__":
    main()
