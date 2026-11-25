"""Prepare a DIRAC input for the stereospecified PubChem 3D record."""
from collections import Counter
from pathlib import Path
from tempfile import mkdtemp

import requests
from rdkit import Chem

root = Path(__file__).resolve().parent
url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/5288605/record/SDF?record_type=3d"
response = requests.get(url, timeout=60)
response.raise_for_status()
molecule = Chem.MolFromMolBlock(response.text.split("$$$$")[0], removeHs=False)
if molecule is None or not molecule.GetNumConformers() or not molecule.GetConformer().Is3D():
    raise ValueError("PubChem did not return a readable 3D molecule")
if Counter(atom.GetSymbol() for atom in molecule.GetAtoms()) != {"C": 4, "H": 5, "I": 1, "N": 2, "O": 2}:
    raise ValueError("Unexpected composition or missing explicit H coordinates")
if Chem.GetFormalCharge(molecule) != 0:
    raise ValueError("Expected a neutral conformer")
Chem.AssignStereochemistryFrom3D(molecule)
centres = Chem.FindMolChiralCenters(molecule, includeUnassigned=True, useLegacyImplementation=False)
if len(centres) != 1 or centres[0][1] != "S" or molecule.GetAtomWithIdx(centres[0][0]).GetAtomicNum() != 6:
    raise ValueError(f"Expected one S carbon centre, found {centres}")
electrons = sum(atom.GetAtomicNum() for atom in molecule.GetAtoms())
template = (root / "CHEMBL1723373" / "CHEMBL1723373_4c_B3LYP_PVC.inp").read_text()
input_text = template.replace("CHEMBL1723373 R", "CID5288605 S").replace(".CLOSED SHELL\n 86", f".CLOSED SHELL\n {electrons}")
runs = root.parent / "runs"
runs.mkdir(exist_ok=True)
job = Path(mkdtemp(prefix="CID5288605_S.", dir=runs))
(job / "CID5288605_S.sdf").write_text(response.text)
(job / "CID5288605_S.xyz").write_text(Chem.MolToXYZBlock(molecule))
(job / "CID5288605_S.inp").write_text(input_text)
print(f"Prepared {electrons}-electron DIRAC input in {job}")
print('From that directory, launch with:')
print('OMP_NUM_THREADS=8 OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 "$DIRAC_ROOT/pam" --inp CID5288605_S.inp --mol CID5288605_S.xyz --dirac "$DIRAC_ROOT/dirac.x" --mw=512')
