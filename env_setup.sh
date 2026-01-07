#!/usr/bin/env bash
# 安装基础图形依赖
sudo apt-get update
sudo apt-get install -y python3-opengl xvfb freeglut3-dev libgl1-mesa-dev

# 创建并激活环境
conda create -n duckie_42 python=3.8 -y
conda activate duckie_42

# duckietown-gym-daffy 6.2.0 requires numpy<=1.20.0
pip install --upgrade pip wheel setuptools

# 克隆仓库
REPO_URL="https://github.com/duckietown/gym-duckietown.git"
REPO_DIR="gym-duckietown"

if [ -d "$REPO_DIR" ]; then
	echo "Repo already exists: $REPO_DIR"
else
	git clone "$REPO_URL" "$REPO_DIR"
fi

cd "$REPO_DIR"

# 安装依赖
pip install -e .

# RL libs (single-env option): use Stable-Baselines3 v1.x (Gym-based)
# Do NOT install shimmy/gymnasium here; they tend to pull newer NumPy.
pip uninstall -y stable-baselines3 shimmy gymnasium || true
# SB3 1.8.0 depends on gym==0.21 (pip build often fails with modern tooling), so install gym via conda.
conda install -y -c conda-forge "gym=0.21.*"

# Keep Duckietown compatible (duckietown-gym-daffy requires numpy<=1.20.0)
pip install --ignore-installed --no-deps --force-reinstall "numpy==1.20.0"
pip install --no-deps "pandas==1.3.5"
pip install --no-deps "stable-baselines3==1.8.0"