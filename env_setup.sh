#!/usr/bin/env bash
set -euo pipefail

# 安装基础图形依赖
sudo apt-get update
sudo apt-get install -y python3-opengl xvfb freeglut3-dev libgl1-mesa-dev

ENV_NAME="duckie_clean"
REPO_URL="https://github.com/duckietown/gym-duckietown.git"
REPO_DIR="gym-duckietown"
CONSTRAINTS_FILE="duckie_constraints.txt"

# Ensure conda is available in non-interactive shells
source "$(conda info --base)/etc/profile.d/conda.sh"

# Create env only if missing
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
	echo "Conda env already exists: $ENV_NAME"
else
	conda create -y -n "$ENV_NAME" python=3.8
fi

conda activate "$ENV_NAME"

# Base toolchain + core pins via conda-forge (avoids old Gym build issues)
conda install -y -c conda-forge "pip<23.2" "setuptools<66" wheel "numpy=1.20.*" "gym=0.21.*"

# Clone repo only if missing
if [ -d "$REPO_DIR" ]; then
	echo "Repo already exists: $REPO_DIR"
else
	git clone "$REPO_URL" "$REPO_DIR"
fi

# Install Duckietown with pinned constraints
if [ ! -f "$CONSTRAINTS_FILE" ]; then
	echo "Missing constraints file: $CONSTRAINTS_FILE" >&2
	echo "Expected it at project root (next to env_setup.sh)." >&2
	exit 1
fi

python -m pip install -c "$CONSTRAINTS_FILE" Pillow==9.5.0 trimesh==3.9.43 pyglet==1.5.0 opencv-python
python -m pip install -c "$CONSTRAINTS_FILE" -e "$REPO_DIR"

python -m pip check

echo "Environment ready: conda activate $ENV_NAME"