"""Exploratory SMILES-to-|EPV| regression, using existing labels only."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from pv_proxy_scan import structural_components, exact_one_tetrahedral_carbon

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {"H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}
ALPHA = 4.0


def describe(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError(f"Expected one valid molecular component: {smiles}")
    if Chem.GetFormalCharge(mol) or not {a.GetSymbol() for a in mol.GetAtoms()} <= ALLOWED:
        raise ValueError(f"Expected a neutral organic structure within the supported elements: {smiles}")
    centre = exact_one_tetrahedral_carbon(mol)
    if centre is None or centre[1] not in {"R", "S"}:
        raise ValueError(f"Expected exactly one specified carbon stereocentre: {smiles}")
    components = structural_components(
        mol, centre[0], rdMolDescriptors.CalcNumRotatableBonds(mol), mol.GetNumHeavyAtoms()
    )
    if components is None:
        raise ValueError(f"Expected at least one Cl, Br or I: {smiles}")
    halogens, nuclear, polar, rigidity, compactness, _ = components
    baseline = np.log10(nuclear * polar * rigidity * compactness)
    features = np.log10([nuclear, polar, rigidity, compactness])
    group = Chem.MolToSmiles(mol, isomericSmiles=False)
    return group, baseline, features, {item["element"] for item in halogens}


def load_labels(path):
    labels = defaultdict(list)
    for row in json.loads(path.read_text()):
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


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--labels", type=Path, default=ROOT / "results" / "epv_ml_labels.json")
    cli.add_argument("--smiles", action="append", default=[], help="Repeat for multiple structures.")
    cli.add_argument("--smi-file", type=Path, help="Whitespace-delimited SMILES and optional name.")
    args = cli.parse_args()
    labels = load_labels(args.labels)
    if len(labels) < 2:
        cli.error("Training requires at least two distinct connectivities.")
    print(f"Training connectivities: {len(labels)}")
    print("Exploratory estimates. Archived labels mix two basis-set protocols.")
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
