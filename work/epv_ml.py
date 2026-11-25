"""Exploratory SMILES-to-|EPV| regression, using existing labels only."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from pv_proxy_scan import structural_components, exact_one_tetrahedral_carbon

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
ALPHA = 4.0
LIBRARY_FILES = (
    ("QMugs/ChEMBL", "qmugs_single_chiral_br_cl_candidates.json"),
    ("PubChem", "pubchem_pv_candidates.json"),
    ("PubChem", "pubchem_direct_heavy_candidates.json"),
)


def describe(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"Expected one valid molecular component: {smiles}")
    return describe_molecule(mol)


def describe_molecule(mol):
    if Chem.GetFormalCharge(mol) or not {a.GetSymbol() for a in mol.GetAtoms()} <= ALLOWED:
        raise ValueError("Expected a neutral organic structure within the supported elements")
    centre = exact_one_tetrahedral_carbon(mol)
    if centre is None or centre[1] not in {"R", "S"}:
        raise ValueError("Expected exactly one specified carbon stereocentre")
    components = structural_components(
        mol, centre[0], rdMolDescriptors.CalcNumRotatableBonds(mol), mol.GetNumHeavyAtoms()
    )
    if components is None:
        raise ValueError("Expected at least one Cl, Br or I")
    halogens, nuclear, polar, rigidity, compactness, _ = components
    baseline = np.log10(nuclear * polar * rigidity * compactness)
    features = np.log10([nuclear, polar, rigidity, compactness])
    group = Chem.MolToSmiles(mol, isomericSmiles=False)
    return group, baseline, features, {item["element"] for item in halogens}


def group_labels(records):
    labels = defaultdict(list)
    for row in records:
        group, baseline, features, elements = describe(row["smiles"])
        magnitude = abs(float(row["epv_eh"]))
        if not np.isfinite(magnitude) or magnitude <= 0:
            raise ValueError(f"Expected a nonzero, finite EPV label for {row['id']}")
        labels[group].append((baseline, features, np.log10(magnitude), elements))
    # One target per connectivity. Mirror calculations share a group.
    return [
        (group, rows[0][0], rows[0][1], np.mean([row[2] for row in rows]), rows[0][3])
        for group, rows in labels.items()
    ]


def fit(records):
    features = np.array([row[2] for row in records])
    residual = np.array([row[3] - row[1] for row in records])
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1.0
    design = (features - center) / scale
    intercept = residual.mean()
    weights = np.linalg.solve(
        design.T @ design + ALPHA * np.eye(design.shape[1]),
        design.T @ (residual - intercept),
    )
    return center, scale, intercept, weights, features.min(axis=0), features.max(axis=0)


def predict(baseline, features, model):
    center, scale, intercept, weights, lower, upper = model
    # Extrapolate only the declared physics heuristic, not the fitted correction.
    bounded = np.clip(features, lower, upper)
    return baseline + intercept + ((bounded - center) / scale) @ weights


def score_library(model, seen_groups, seen_elements, output_path):
    predictions = {}
    rejected = Counter()
    input_count = 0
    duplicates = 0
    for source, filename in LIBRARY_FILES:
        records = json.loads((ROOT / "outputs" / filename).read_text())
        input_count += len(records)
        for record in records:
            smiles = record.get("smiles") or ""
            identifier = record.get("chembl_id") or f"CID{record['cid']}"
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or len(Chem.GetMolFrags(mol)) != 1:
                rejected["invalid or disconnected SMILES"] += 1
                continue
            try:
                group, baseline, features, elements = describe_molecule(mol)
            except ValueError as error:
                rejected[str(error)] += 1
                continue
            stereo_key = Chem.MolToSmiles(mol, isomericSmiles=True)
            record_source = {"source": source, "id": identifier}
            if stereo_key in predictions:
                duplicates += 1
                aliases = predictions[stereo_key]["records"]
                if record_source not in aliases:
                    aliases.append(record_source)
                continue
            predictions[stereo_key] = {
                "id": identifier,
                "name": record.get("name") or identifier,
                "smiles": stereo_key,
                "records": [record_source],
                "halogens": sorted(elements),
                "in_training": group in seen_groups,
                "unseen_halogens": sorted(elements - seen_elements),
                "outside_descriptor_range": bool(
                    np.any(features < model[4] - 1e-12) or np.any(features > model[5] + 1e-12)
                ),
                "ridge_abs_epv_eh": float(10 ** predict(baseline, features, model)),
                "heuristic_abs_epv_eh": float(10 ** (baseline + model[2])),
            }
    ranked = sorted(predictions.values(), key=lambda row: row["ridge_abs_epv_eh"], reverse=True)
    summary = {
        "training_connectivities": len(seen_groups),
        "input_records": input_count,
        "duplicates": duplicates,
        "rejected": dict(rejected),
        "scored_stereostructures": len(ranked),
        "source_counts": dict(Counter(source for row in ranked for source in {r["source"] for r in row["records"]})),
        "containing_iodine": sum("I" in row["halogens"] for row in ranked),
        "outside_descriptor_range": sum(row["outside_descriptor_range"] for row in ranked),
    }
    print("\nLibrary summary:", json.dumps(summary, ensure_ascii=False))
    for title, selected in (
        ("Overall ridge top 10", ranked),
        ("QMugs/ChEMBL ridge top 5", [
            row for row in ranked if any(r["source"] == "QMugs/ChEMBL" for r in row["records"])
        ][:5]),
        ("Overall calibrated heuristic top 5", sorted(
            ranked, key=lambda row: row["heuristic_abs_epv_eh"], reverse=True
        )[:5]),
    ):
        print(f"\n{title} (Eh)")
        for row in selected[:10]:
            flags = []
            if row["in_training"]:
                flags.append("in training")
            if row["outside_descriptor_range"]:
                flags.append("outside descriptor range")
            if row["unseen_halogens"]:
                flags.append("unseen halogen")
            flag_text = f" [{', '.join(flags)}]" if flags else ""
            print(f"  {row['id']}: ridge {row['ridge_abs_epv_eh']:.3e}, "
                  f"heuristic {row['heuristic_abs_epv_eh']:.3e}{flag_text}")
    output_path.write_text(json.dumps({"summary": summary, "predictions": ranked}, indent=2) + "\n")
    print(f"\nFull estimates: {output_path}")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--labels", type=Path, default=ROOT / "results" / "epv_ml_labels.json")
    cli.add_argument("--literature", action="store_true", help="Add curated literature training labels and assess the external aromatic series.")
    cli.add_argument("--smiles", action="append", default=[], help="Repeat for multiple structures.")
    cli.add_argument("--smi-file", type=Path, help="Whitespace-delimited SMILES and optional name.")
    cli.add_argument("--library-output", type=Path, help="Score the bundled QMugs and PubChem candidate snapshots and write ranked JSON.")
    args = cli.parse_args()
    records = json.loads(args.labels.read_text())
    literature = (
        json.loads((ROOT / "results" / "epv_literature.json").read_text())
        if args.literature else []
    )
    labels = group_labels(records + [row for row in literature if row["role"] == "train"])
    if len(labels) < 2:
        cli.error("Training requires at least two distinct connectivities.")
    print(f"Training connectivities: {len(labels)}")
    print("Exploratory estimates. Reference labels mix computational protocols.")
    if len(labels) == 2:
        print("Two-connectivity diagnostic: each held-out fit has one training molecule and learns only the intercept.")
    errors = []
    baseline_errors = []
    for held_out in labels:
        group, baseline, features, target, _ = held_out
        remaining = [row for row in labels if row[0] != group]
        model = fit(remaining)
        errors.append(abs(predict(baseline, features, model) - target))
        baseline_errors.append(abs(baseline + model[2] - target))
    print(f"Leave-one-connectivity-out MAE (log10 Eh): ridge {np.mean(errors):.3f}, baseline {np.mean(baseline_errors):.3f}")

    model = fit(labels)
    seen_groups = {row[0] for row in labels}
    seen_elements = set().union(*(row[4] for row in labels))
    external_rows = [row for row in literature if row["role"] == "external_test"]
    if external_rows:
        print("External scaffold test (approximate Figure 1 reference values):")
        external_errors = []
        external_baseline_errors = []
        for row in external_rows:
            group, baseline, features, _ = describe(row["smiles"])
            if group in seen_groups:
                cli.error(f"External test connectivity overlaps training: {row['id']}")
            target = np.log10(abs(row["epv_eh"]))
            estimated = predict(baseline, features, model)
            heuristic = baseline + model[2]
            external_errors.append(abs(estimated - target))
            external_baseline_errors.append(abs(heuristic - target))
            print(f"  {row['id']}: reference {10**target:.2e}, ridge {10**estimated:.2e}, baseline {10**heuristic:.2e} Eh")
        print(f"External MAE (log10 Eh): ridge {np.mean(external_errors):.3f}, baseline {np.mean(external_baseline_errors):.3f}")

    if args.library_output:
        score_library(model, seen_groups, seen_elements, args.library_output)

    queries = [(smiles, smiles) for smiles in args.smiles]
    if args.smi_file:
        for line in args.smi_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                smiles, *name = line.split(maxsplit=1)
                queries.append((smiles, name[0] if name else smiles))

    estimates = []
    for smiles, name in queries:
        try:
            group, baseline, features, elements = describe(smiles)
        except ValueError as error:
            print(f"Skipped {name}: {error}")
            continue
        flags = ["in training" if group in seen_groups else "unseen connectivity"]
        if elements - seen_elements:
            flags.append("unseen halogen: " + ", ".join(sorted(elements - seen_elements)))
        if np.any(features < model[4] - 1e-12) or np.any(features > model[5] + 1e-12):
            flags.append("outside training descriptor range")
        estimates.append((predict(baseline, features, model), name, flags))
    for estimate, name, flags in sorted(estimates, reverse=True):
        print(f"{name}: |EPV| estimate {10 ** estimate:.3e} Eh [{'; '.join(flags)}]")


if __name__ == "__main__":
    main()
