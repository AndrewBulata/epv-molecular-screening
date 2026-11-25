#!/bin/sh
set -eu
: "${DIRAC_ROOT:?Set DIRAC_ROOT to an absolute DIRAC installation path}"
case "${1:-}" in
    R) input=CHEMBL1723373_4c_B3LYP_PVC; ;;
    S) input=CHEMBL1723373_S_4c_B3LYP_PVC; ;;
    *) echo "Usage: $0 R|S [OpenMP threads]" >&2; exit 2 ;;
esac
threads=${2:-1}
case "$threads" in
    ''|*[!0-9]*|0) echo "Specify a positive thread count" >&2; exit 2 ;;
esac
script_dir=$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)
geometry=CHEMBL1723373_$1
mkdir -p "$script_dir/../runs"
run_dir=$(mktemp -d "$script_dir/../runs/${geometry}.XXXXXXXX")
cp "$script_dir/CHEMBL1723373/$input.inp" "$script_dir/CHEMBL1723373/$geometry.xyz" "$run_dir/"
cd "$run_dir"
echo "Run directory: $run_dir"
export OMP_NUM_THREADS="$threads" OMP_DYNAMIC=FALSE OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$DIRAC_ROOT/pam" --inp "$input.inp" --mol "$geometry.xyz" --dirac "$DIRAC_ROOT/dirac.x" --mw=512
output="${input}_${geometry}.out"
awk '/Parity violation - chirality E\(PV\):/{show=1} show{print} /Total EPV[[:space:]]*=/{found=1; show=0} END{if (!found) exit 1}' "$output"
