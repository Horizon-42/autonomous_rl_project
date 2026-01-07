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

# Duckietown-world runtime deps
conda install -y -c conda-forge pyyaml "networkx<3"

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

# RL training deps (safe pins)
# - Use SB3 v1.x (Gym-based). Do NOT install shimmy/gymnasium here.
# - Install torch via conda to avoid pip pulling incompatible binaries.
# CPU-only (recommended baseline):
conda install -y -c pytorch cpuonly "pytorch==1.13.1"

# Workaround: some PyTorch builds expect optional iJIT profiling symbols at load time
# (iJIT_NotifyEvent/iJIT_IsProfilingActive/iJIT_GetNewMethodID). On some systems these
# come from Intel's libittnotify, but it may be unavailable. We provide a tiny stub
# library and preload it automatically when the env activates.
if command -v gcc >/dev/null 2>&1; then
	STUB_C="${CONDA_PREFIX}/lib/jitprofiling_stub.c"
	STUB_SO="${CONDA_PREFIX}/lib/libjitprofiling_stub.so"
	cat > "$STUB_C" <<'C'
#include <stdint.h>
__attribute__((visibility("default"))) uint32_t iJIT_GetNewMethodID(void){ static uint32_t id=0; return ++id; }
__attribute__((visibility("default"))) int iJIT_IsProfilingActive(void){ return 0; }
__attribute__((visibility("default"))) int iJIT_NotifyEvent(int eventType, void* eventData){ (void)eventType; (void)eventData; return 0; }
C
	gcc -shared -fPIC -O2 -o "$STUB_SO" "$STUB_C"
	mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"
	cat > "${CONDA_PREFIX}/etc/conda/activate.d/jitprofiling_stub.sh" <<'SH'
export _OLD_LD_PRELOAD_JITSTUB="${LD_PRELOAD-}"
_stub="$CONDA_PREFIX/lib/libjitprofiling_stub.so"
case ":${LD_PRELOAD-}:" in
  *":${_stub}:"*) ;;
  *) export LD_PRELOAD="${_stub}${LD_PRELOAD:+:${LD_PRELOAD}}" ;;
esac
SH
	cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/jitprofiling_stub.sh" <<'SH'
export LD_PRELOAD="${_OLD_LD_PRELOAD_JITSTUB-}"
unset _OLD_LD_PRELOAD_JITSTUB
SH
else
	echo "WARNING: gcc not found; skipping torch iJIT stub workaround. If 'import torch' fails, install gcc and re-run setup." >&2
fi

# If you want GPU instead, comment the cpuonly line above and use something like:
# conda install -y -c pytorch -c nvidia pytorch pytorch-cuda=12.1

# SB3 wants pandas; install without deps so it doesn't upgrade numpy.
python -m pip install --no-deps "pandas==1.3.5"
python -m pip install --no-deps "stable-baselines3==1.8.0"

python -m pip check

echo "Environment ready: conda activate $ENV_NAME"