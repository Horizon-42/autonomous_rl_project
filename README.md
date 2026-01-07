
# Autonomous RL Project (Duckietown)

This workspace contains a Duckietown Gym setup and a lane-following PPO training script.

## Training

Activate the environment (example):

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate duckie_clean
```

Run training (logs go to `runs/centerline` by default):

```bash
python train_jetracer_centerline.py \
	--map-name loop_empty \
	--total-timesteps 200000 \
	--n-envs 1
```

Optional rendering (slower; must keep a single env):

```bash
python train_jetracer_centerline.py \
	--map-name loop_empty \
	--total-timesteps 200000 \
	--render \
	--n-envs 1
```

## Reward Curve (TensorBoard)

The training script writes TensorBoard event files under:

- `runs/centerline/`

It also records episode returns via SB3 `VecMonitor`:

- `runs/centerline/monitor.csv`

### 1) Install TensorBoard

```bash
python -m pip install tensorboard
```

### 2) Fix TensorBoard ↔ protobuf compatibility (if needed)

If you see an error like:

`TypeError: MessageToJson() got an unexpected keyword argument 'including_default_value_fields'`

pin protobuf to a compatible version:

```bash
python -m pip install --no-deps "protobuf==3.20.3"
```

### 3) Start TensorBoard

Use `python -m tensorboard.main ...` (more reliable than a `tensorboard` shell entrypoint):

```bash
python -m tensorboard.main --logdir runs/centerline --port 6007 --bind_all
```

Open in a browser:

- `http://<server-ip>:6007/`

Look for these scalars:

- `rollout/ep_rew_mean` (reward curve)
- `rollout/ep_len_mean`

### Remote (SSH port-forward)

If the server port is not directly reachable:

```bash
ssh -L 6007:localhost:6007 <user>@<server>
```

Then open:

- `http://localhost:6007/`

