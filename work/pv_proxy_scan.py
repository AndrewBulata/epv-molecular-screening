#!/usr/bin/env python3
"""Fast, deliberately approximate PV prioritisation for local candidate lists.

This is a ranking surrogate, not a replacement for a relativistic E_PV calculation.
It combines Z^5, graph distance to the sole stereocentre, rigidity, nearby polar
groups and (for QMugs) precomputed GFN2 descriptors.  It deliberately does not
generate or optimise geometries: QMugs conformers are already optimised.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
ANCHOR_ID = "CHEMBL1723373"
ANCHOR_EPV_EH = 1.753952279838e-18
Z = {"Cl": 17, "Br": 35, "I": 53}
IDEAL_TETRA_DET = 4.0 / (3.0 * math.sqrt(3.0))


def tetrahedral_quality(mol_h: Chem.Mol, center: int, conf_id: int) -> float:
    """0..1 measure of how non-planar/tetrahedral the nominated carbon is."""
    atom = mol_h.GetAtomWithIdx(center)
    nbrs = [n.GetIdx() for n in atom.GetNeighbors()]
    if len(nbrs) != 4 or atom.GetSymbol() != "C":
        return 0.0
    conf = mol_h.GetConformer(conf_id)
    c = np.array(conf.GetAtomPosition(center))
    vecs = []
    for idx in nbrs:
        v = np.array(conf.GetAtomPosition(idx)) - c
        norm = np.linalg.norm(v)
        if norm < 1e-8:
            return 0.0
        vecs.append(v / norm)
    dets = []
    for i in range(4):
        tri = [vecs[j] for j in range(4) if j != i]
        dets.append(abs(float(np.linalg.det(np.stack(tri)))))
    return float(np.clip(np.mean(dets) / IDEAL_TETRA_DET, 0.0, 1.0))


def embed_qualities(mol: Chem.Mol, center: int, nconf: int = 3) -> tuple[float, float, int]:
    mh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 20261117
    params.pruneRmsThresh = 0.35
    params.numThreads = 1
    ids = list(AllChem.EmbedMultipleConfs(mh, numConfs=nconf, params=params))
    if not ids:
        return 0.0, 1.0, 0
    qs = []
    for cid in ids:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mh):
                AllChem.MMFFOptimizeMolecule(mh, confId=cid, maxIters=200)
            else:
                AllChem.UFFOptimizeMolecule(mh, confId=cid, maxIters=200)
        except Exception:
            pass
        qs.append(tetrahedral_quality(mh, center, cid))
    mean = float(np.mean(qs))
    cv = float(np.std(qs) / max(mean, 1e-8))
    return mean, cv, len(ids)


def exact_one_tetrahedral_carbon(mol: Chem.Mol):
    Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
    # Count potential but unassigned stereocentres too.  Otherwise a molecule
    # with one labelled R/S centre and several unspecified centres would slip
    # through as an apparent single-centre candidate.
    centers = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    carbon = [
        (idx, cip)
        for idx, cip in centers
        if cip != "?" and mol.GetAtomWithIdx(idx).GetSymbol() == "C"
    ]
    return carbon[0] if len(centers) == 1 and len(carbon) == 1 else None


def structural_components(mol: Chem.Mol, center: int, rotors: int, heavy_atoms: int):
    dmat = Chem.GetDistanceMatrix(mol)
    heavy = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in Z:
            distance = int(round(dmat[center, atom.GetIdx()]))
            term = (Z[symbol] / 35.0) ** 5 * math.exp(-0.80 * max(0, distance - 1))
            heavy.append({"element": symbol, "atom": atom.GetIdx(), "distance": distance, "z5_term": term})
    if not heavy:
        return None

    # Multiple equivalent heavy atoms are more likely to cancel electronically.
    groups = {}
    for h in heavy:
        groups.setdefault((h["element"], h["distance"]), 0)
        groups[(h["element"], h["distance"])] += 1
    symmetry_penalty = 1.0
    for count in groups.values():
        if count > 1:
            symmetry_penalty *= 1.0 / math.sqrt(count)

    hetero_near = 0
    for atom in mol.GetAtoms():
        if atom.GetSymbol() in {"N", "O", "F", "S", "P"} and dmat[center, atom.GetIdx()] <= 2.01:
            hetero_near += 1
    polar_factor = 1.0 + 0.12 * min(hetero_near, 4)
    patterns = ["[CX3]=[OX1]", "[C]#N", "[SX4](=[OX1])(=[OX1])", "[PX4](=[OX1])"]
    motif_hits = sum(bool(mol.HasSubstructMatch(Chem.MolFromSmarts(s))) for s in patterns)
    polar_factor *= 1.0 + 0.18 * min(motif_hits, 2)

    rigidity = math.exp(-0.16 * max(0, rotors))
    compactness = float(np.clip((10.0 / max(heavy_atoms, 1)) ** 0.25, 0.72, 1.25))
    nuclear = sum(h["z5_term"] for h in heavy) * symmetry_penalty
    return heavy, nuclear, polar_factor, rigidity, compactness, symmetry_penalty


def scan_record(record: dict, source: str):
    smiles = record.get("smiles")
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    center_info = exact_one_tetrahedral_carbon(mol)
    if center_info is None:
        return None
    center, cip = center_info
    rotors = int(record.get("rotatable_bonds", 0))
    heavy_atoms = int(record.get("heavy_atoms", mol.GetNumHeavyAtoms()))
    comp = structural_components(mol, center, rotors, heavy_atoms)
    if comp is None:
        return None
    heavy, nuclear, polar, rigidity, compact, symmetry_penalty = comp
    # The selected records already contain optimised QMugs conformers.  The
    # summary JSON does not carry their coordinates, so do not replace them by
    # an RDKit force-field geometry.  A specified tetrahedral carbon is given a
    # neutral geometry factor of 1.0; conformer sensitivity is left for a later
    # coordinate-aware pass over the original QMugs structure archive.
    qmean, qcv, nconf = 1.0, 0.0, 1
    raw_structural = nuclear * polar * rigidity * compact

    electronic_factor = 1.0
    if source == "QMugs":
        pol = record.get("GFN2_POLARIZABILITY_MOLECULAR")
        dip = record.get("GFN2_DIPOLE_TOT")
        gap = record.get("GFN2_HOMO_LUMO_GAP")
        if isinstance(pol, (int, float)) and pol > 0:
            electronic_factor *= float(np.clip((pol / 82.760509) ** 0.30, 0.65, 1.65))
        if isinstance(dip, (int, float)):
            electronic_factor *= float(np.clip(1.0 + 0.055 * (dip - 4.214), 0.78, 1.22))
        if isinstance(gap, (int, float)) and gap > 0:
            electronic_factor *= float(np.clip((0.1142364 / gap) ** 0.12, 0.80, 1.25))

    return {
        "source": source,
        "id": record.get("chembl_id") or (f"CID{record.get('cid')}" if record.get("cid") else "unknown"),
        "name": record.get("name"),
        "smiles": smiles,
        "formula": record.get("formula"),
        "cip": cip,
        "center_atom": center,
        "heavy_near_chirality": heavy,
        "tetrahedral_quality": qmean,
        "conformer_cv": qcv,
        "conformers": nconf,
        "nuclear_factor": nuclear,
        "polar_factor": polar,
        "rigidity_factor": rigidity,
        "compactness_factor": compact,
        "symmetry_penalty": symmetry_penalty,
        "electronic_factor": electronic_factor,
        "raw_score": raw_structural * electronic_factor,
    }


def load_and_scan(path: Path, source: str):
    rows = json.loads(path.read_text())
    results = []
    for i, row in enumerate(rows, 1):
        result = scan_record(row, source)
        if result:
            results.append(result)
        if i % 500 == 0:
            print(f"{source}: {i}/{len(rows)} citite; {len(results)} valide", flush=True)
    return results


def main():
    qmugs = load_and_scan(OUT / "qmugs_single_chiral_br_cl_candidates.json", "QMugs")
    pubchem_a = load_and_scan(OUT / "pubchem_pv_candidates.json", "PubChem")
    pubchem_b = load_and_scan(OUT / "pubchem_direct_heavy_candidates.json", "PubChem")

    # Deduplicate PubChem records present in both local lists.
    dedup = {}
    for r in pubchem_a + pubchem_b:
        dedup[(r["id"], r["smiles"])] = r
    pubchem = list(dedup.values())

    anchor = next((r for r in qmugs if r["id"] == ANCHOR_ID), None)
    if anchor is None or anchor["raw_score"] <= 0:
        raise RuntimeError("DIRAC calibration anchor was not found")
    anchor_raw = anchor["raw_score"]
    for r in qmugs + pubchem:
        r["relative_to_br_anchor"] = r["raw_score"] / anchor_raw
        r["rough_epv_eh"] = ANCHOR_EPV_EH * r["relative_to_br_anchor"]
        r["rough_rs_hz"] = 2.0 * r["rough_epv_eh"] * 6.5796839204999e15

    qmugs.sort(key=lambda x: x["raw_score"], reverse=True)
    pubchem.sort(key=lambda x: x["raw_score"], reverse=True)
    combined = sorted(qmugs + pubchem, key=lambda x: x["raw_score"], reverse=True)
    payload = {
        "warning": "Rough ranking surrogate only; rough_epv_eh is anchored to one DIRAC calculation and is not a computed PV energy.",
        "calibration": {"id": ANCHOR_ID, "epv_eh": ANCHOR_EPV_EH, "raw_score": anchor_raw},
        "counts": {"qmugs": len(qmugs), "pubchem": len(pubchem)},
        "qmugs": qmugs,
        "pubchem": pubchem,
        "combined": combined,
    }
    json_path = OUT / "pv_proxy_ranked_candidates.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    lines = []
    lines.append("PV PROXY SCAN — RELATIVE PRIORITISATION, NOT DIRAC ENERGIES")
    lines.append(f"Valid QMugs: {len(qmugs)} | valid unique PubChem: {len(pubchem)}")
    lines.append(f"Anchor: {ANCHOR_ID}, E_PV={ANCHOR_EPV_EH:.6e} Eh, raw={anchor_raw:.6g}")
    for title, rows in [("TOP 20 QMUGS", qmugs[:20]), ("TOP 20 PUBCHEM", pubchem[:20])]:
        lines.extend(["", "=" * 112, title, "=" * 112])
        for rank, r in enumerate(rows, 1):
            hs = ", ".join(f"{h['element']}@d{h['distance']}" for h in r["heavy_near_chirality"])
            lines.append(
                f"{rank:2d}. {r['id']:<18} rel={r['relative_to_br_anchor']:8.2f}  "
                f"rough|Epv|={r['rough_epv_eh']:.2e} Eh  rough R/S={r['rough_rs_hz']:.3g} Hz"
            )
            if r.get("name"):
                lines.append(f"    {r['name']}")
            lines.append(
                f"    {r['formula']} | {r['cip']} | {hs} | "
                f"rotor factor={r['rigidity_factor']:.3f} | symmetry={r['symmetry_penalty']:.3f}"
            )
            lines.append(f"    {r['smiles']}")
    report_path = OUT / "pv_proxy_top_candidates.txt"
    report_path.write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved JSON: {json_path}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
