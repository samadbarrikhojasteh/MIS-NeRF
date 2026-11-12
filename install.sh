
#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="MIS-NeRF"
TCNN_COMMIT="c91138bcd4c6877c8d5e60e483c0581aafc70cce"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "========================================"
echo " Installing MIS-NeRF"
echo "========================================"

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: Conda was not found."
    exit 1
fi

set +u
eval "$(conda shell.bash hook)"

if conda env list | awk 'NR>2 {print $1}' | grep -Fxq "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists."
else
    echo "Creating Conda environment..."
    conda env create -f environment-train.yml
fi

conda activate "$ENV_NAME"
set -u

echo "Python:"
python --version

echo "Installing tiny-cuda-nn..."
export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${LIBRARY_PATH:+:$LIBRARY_PATH}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

git clone https://github.com/NVlabs/tiny-cuda-nn.git "$TMP_DIR/tiny-cuda-nn"

git -C "$TMP_DIR/tiny-cuda-nn" checkout "$TCNN_COMMIT"

git -C "$TMP_DIR/tiny-cuda-nn" submodule update --init --recursive

export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export CPATH="/usr/include${CPATH:+:$CPATH}"

python -m pip install --no-build-isolation \
    "$TMP_DIR/tiny-cuda-nn/bindings/torch"

echo "Installing modified Nerfstudio..."

python -m pip install -e "$ROOT_DIR/MIS_NeRF"

echo "========================================"
echo " MIS-NeRF installation completed!"
echo "========================================"

echo "Activate later using:"
echo "conda activate $ENV_NAME"
