# Molecular parity-violation screening

Exploratory workflow for finding small, metal-free chiral molecules for relativistic calculations of the electroweak parity-violating energy, $E_{\mathrm{PV}}$. The workflow uses a low-cost heuristic to shortlist molecules, followed by four-component DIRAC calculations on selected geometries.

## Calculated result

CHEMBL1723373 (C4H5BrN2O) was evaluated at DIRAC 26.1, four-component Dirac–Coulomb/B3LYP, dyall.v2z, with the nuclear spin-independent `.PVC` operator. The R structure is an RDKit ETKDG/MMFF94s geometry. The S structure is its coordinate reflection. These fixed geometries were used as single-point inputs.

| Enantiomer | Total SCF energy (Eh) | $E_{\mathrm{PV}}$ (Eh) | Br contribution (Eh) |
| --- | ---: | ---: | ---: |
| R | -2947.0869526405450 | +1.753952279838e-18 | +1.887670462594e-18 |
| S | -2947.0869526404003 | -1.754008978620e-18 | -1.887727092867e-18 |

The two EPV values reverse sign and agree in magnitude to about 0.0032%. Their difference is $3.5080\times10^{-18}$ Eh, equivalent to about 0.0231 Hz when divided by Planck's constant. This refers to the computed fixed-geometry ground-state difference. For a spectroscopic line the relevant observable is `[(EPV_upper − EPV_lower)_R − (EPV_upper − EPV_lower)_S]/h`, with the appropriate state averages.

- [Exact inputs and XYZ geometries](dirac/CHEMBL1723373)
- Output excerpts for [R](results/CHEMBL1723373_R_pvc.txt) and [S](results/CHEMBL1723373_S_pvc.txt)

### Other completed calculations

Bromochlorofluoromethane (CHFClBr) was also evaluated with 4c Dirac–Coulomb/B3LYP and cc-pVDZ at an ORCA B3LYP-D4/def2-TZVP geometry and its exact mirror. The [optimized S geometry](dirac/CHFClBr/S_optimized) gives `−3.873610964673e−19 Eh`; the [mirrored R geometry](dirac/CHFClBr/R_mirror) gives `+3.873552877906e−19 Eh`. The original local directory labels were reversed relative to the CIP assignments obtained from the coordinates. The archived inputs preserve the original job contents.

The wider Dirac archive also contains displaced CHFClBr geometries, a CH3PbFClBr mirror pair (approximately ±1.411e−15 Eh), and a Pt alaninate calculation (+1.485e−15 Eh for the archived geometry). The displaced geometries share a connectivity with the CHFClBr benchmark. The Pb and Pt compounds lie outside the metal-free model domain.

## Reproduce the shortlist

Use Python 3.10+ from the repository root.

```sh
python -m pip install -r requirements.txt
python work/pv_proxy_scan.py
```

The scanner reads the three frozen JSON candidate lists in `outputs/` and writes a complete ranked JSON and a text report there. The archived [top candidates report](results/pv_proxy_top_candidates.txt) contains separate rankings for QMugs (2,668 records) and PubChem (117 retained ID/SMILES records). These counts refer to the project snapshots. The upstream query scripts under `work/` can refresh those lists from the live services and overwrite the corresponding files.

The screen requires one RDKit-perceived tetrahedral stereocentre with a specified carbon configuration and includes unassigned centres in that count. For each Cl/Br/I atom it computes `(Z/35)^5 × exp(-0.80 × max(0, d−1))`, where `d` is the bond distance to the stereocentre. The summed terms are modified by heuristic weights for repeated element/distance groups, nearby heteroatoms and polar motifs, rotatable bonds, and molecular size. For QMugs only, the score includes additional factors from the archived GFN2 dipole, polarizability and HOMO–LUMO gap. PubChem uses a neutral multiplier.

This ranking uses SMILES and stored descriptors. Its repeated-group penalty is a heuristic and cannot identify physical cancellation. The electronic weighting differs between sources. The report's hartree and Hz columns are one-point rescalings from CHEMBL1723373 R and have no validated predictive accuracy. The score favours I by construction. The leading PubChem candidate is [CID5288605](https://pubchem.ncbi.nlm.nih.gov/compound/5288605), with a score of 7.73 relative to the Br reference. Its DIRAC EPV remains unverified. [Candidate structures as SMILES](results/shortlist.smi) and an [earlier PubChem 3D visualization set](results/pubchem_pv_top10.sdf) are included. The earlier SDF contains model geometries from a different shortlist.

## Experimental ML baseline

```sh
python work/epv_ml.py --smi-file results/shortlist.smi
python work/epv_ml.py --smiles 'O=C1NC[C@H](I)C(=O)N1'
```

The script fits a ridge regression correction to the structural heuristic in log10(|EPV|/Eh), using its nuclear, polar, rigidity and compactness terms. It reads the [archived reference labels](results/epv_ml_labels.json). Training and inference use RDKit and NumPy. Both run without DIRAC or conformer generation. The correction is bounded to the observed descriptor ranges. The heuristic still extrapolates outside those ranges, and the script flags unseen halogens.

The two R/S pairs contribute **two distinct connectivities**. Each pair is grouped and its log magnitudes are averaged. Leave-one-connectivity-out evaluation gives 0.584 log10(Eh) MAE for both ridge and the fold-calibrated heuristic. Each fold has one training structure and therefore fits only an intercept. The result does not establish a predictive advantage for ML. The labels also mix basis sets and geometry protocols. The model targets magnitude and has no iodine reference.

## Reproduce the DIRAC benchmark

Install [DIRAC](https://www.diracprogram.org/doku.php?id=start) separately and set `DIRAC_ROOT` to the directory containing `pam` and `dirac.x`. The original R job was serial. The S job was launched with eight OpenMP threads. The following commands use the archived inputs unchanged and write each job to a fresh working directory.

```sh
export DIRAC_ROOT=/absolute/path/to/dirac
sh dirac/run_benchmark.sh R 1
sh dirac/run_benchmark.sh S 8
```

The runner prints the PVC table and signed `Total EPV` from the output. The [iododihydrouracil SDF, XYZ and prepared input](dirac/CID5288605) are included. To recreate them, run `python dirac/prepare_cid5288605.py`. The script fetches the PubChem 3D conformer and checks the 112-electron neutral composition and S assignment before converting it to XYZ.

## Sources

- Isert et al., [QMugs, quantum mechanical properties of drug-like molecules](https://doi.org/10.1038/s41597-022-01390-7). The screening used the [QMugs summary mirror](https://huggingface.co/datasets/JuIm/QMugs_Summary).
- [PubChem PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) for compound records.
- [DIRAC property documentation](https://diracprogram.org/doc/release-26/manual/properties.html#pvc) for the electroweak operator.
- Laerdahl and Schwerdtfeger, [Relativistic effects in parity-violating energy differences](https://doi.org/10.1103/PhysRevA.60.4439).
