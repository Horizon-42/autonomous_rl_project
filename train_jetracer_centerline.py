"""Train a Duckietown (sim) policy to follow the lane centerline.

This script is written to be easy to read and modify.

Goal
----
Learn a policy that keeps the car near the lane center and aligned with the lane direction.
This is a good proxy for a JetRacer-style lane following task:
- Observation: front RGB camera image
- Action: [velocity, steering] in [-1, 1]

Notes
-----
- This uses Stable-Baselines3 (SB3). For classic Gym (not Gymnasium) compatibility,
  use SB3 v1.x.
- Duckietown (duckietown-gym-daffy 6.2.0) requires numpy<=1.20.0. Some newer
  packages require newer NumPy; pin versions accordingly.

Example
-------
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate duckie_clean

# If SB3 isn't installed in this env yet:
# pip install --no-deps stable-baselines3==1.8.0
# (You also need torch installed; install via conda/pip as appropriate for your machine.)

python train_jetracer_centerline.py --total-timesteps 200000
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import gym
import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    """Weights for lane-following reward shaping."""

    # Positive term for moving forward along the lane direction
    w_forward: float = 1.0

    # Penalty for lateral deviation from lane center (meters)
    w_center: float = 2.0

    # Penalty for heading misalignment (radians)
    w_heading: float = 0.5

    # Extra penalty when the episode ends due to leaving the drivable area
    offroad_penalty: float = 10.0


class ResizeNormalizeObs(gym.ObservationWrapper):
    """Resize RGB observations and normalize to [0, 1].

    SB3's CNN policies typically work with channel-first images.
    """

    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)

        self._width = int(width)
        self._height = int(height)

        # Expect original observations to be HWC uint8 images
        h, w, c = self.observation_space.shape
        assert c == 3, f"Expected RGB observation, got shape {self.observation_space.shape}"

        # New observation is CHW float32
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self._height, self._width),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        import cv2

        resized = cv2.resize(observation, (self._width, self._height), interpolation=cv2.INTER_AREA)
        chw = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        return chw


class CenterLineRewardWrapper(gym.Wrapper):
    """Replace the environment reward with a lane-following shaped reward.

    We use the simulator's lane estimator:
    - dist: signed lateral distance to lane center (meters)
    - angle_rad: heading error relative to lane tangent (radians)
    - dot_dir: cosine-like alignment (1 is perfect forward, 0 sideways, <0 backwards)

    The wrapper still returns the original info dict and adds extra keys under
    info["centerline_reward"].
    """

    def __init__(self, env: gym.Env, cfg: RewardConfig = RewardConfig()):
        super().__init__(env)
        self.cfg = cfg

    def step(self, action):
        obs, _base_reward, done, info = self.env.step(action)

        lane = self._extract_lane_position(info)
        speed = self._extract_speed(info)
        done_why = self._extract_done_reason(info)

        shaped_reward = 0.0

        if lane is not None:
            dist = float(lane.get("dist", 0.0))
            angle_rad = float(lane.get("angle_rad", 0.0))
            dot_dir = float(lane.get("dot_dir", 0.0))

            forward = max(0.0, dot_dir) * float(speed)
            shaped_reward += self.cfg.w_forward * forward
            shaped_reward -= self.cfg.w_center * abs(dist)
            shaped_reward -= self.cfg.w_heading * abs(angle_rad)
        else:
            # If lane position is unavailable, give a small living penalty.
            shaped_reward -= 1.0

        if done and done_why in {"invalid-pose", "out-of-road", "out-of-track"}:
            shaped_reward -= self.cfg.offroad_penalty

        info = dict(info)  # avoid mutating upstream info
        info["centerline_reward"] = {
            "reward": float(shaped_reward),
            "speed": float(speed),
            "done_why": done_why,
            "lane": lane,
        }

        return obs, float(shaped_reward), done, info

    @staticmethod
    def _extract_lane_position(info: Dict) -> Optional[Dict]:
        # Duckietown Simulator puts values under info["Simulator"].
        sim = info.get("Simulator") if isinstance(info, dict) else None
        if isinstance(sim, dict) and isinstance(sim.get("lane_position"), dict):
            return sim["lane_position"]
        return None

    @staticmethod
    def _extract_speed(info: Dict) -> float:
        sim = info.get("Simulator") if isinstance(info, dict) else None
        if isinstance(sim, dict):
            speed = sim.get("robot_speed")
            if speed is not None:
                return float(speed)
        return 0.0

    @staticmethod
    def _extract_done_reason(info: Dict) -> Optional[str]:
        sim = info.get("Simulator") if isinstance(info, dict) else None
        if isinstance(sim, dict):
            msg = sim.get("msg")
            if isinstance(msg, str):
                return msg
        return None


class TrainingVizCallback:
    """Lightweight training visualization.

    - If rendering is enabled, calls env.render() each step.
    - Periodically prints lane metrics from info["centerline_reward"].

    Implemented as an SB3 callback (BaseCallback) to keep the main loop clean.
    """

    def __init__(self, render: bool, print_every_steps: int = 200):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Cb(BaseCallback):
            def __init__(self, render: bool, print_every_steps: int):
                super().__init__()
                self._render = bool(render)
                self._print_every = int(print_every_steps)

            def _on_step(self) -> bool:
                if self._render:
                    try:
                        # DummyVecEnv exposes `.envs`; render only the first env.
                        if hasattr(self.training_env, "envs") and self.training_env.envs:
                            self.training_env.envs[0].render()
                        else:
                            self.training_env.render()
                    except Exception:
                        # Rendering can fail on headless/GL setups; don't kill training.
                        pass

                if self._print_every > 0 and (self.num_timesteps % self._print_every == 0):
                    infos = self.locals.get("infos")
                    if isinstance(infos, (list, tuple)) and infos:
                        extra = infos[0].get("centerline_reward") if isinstance(infos[0], dict) else None
                        if isinstance(extra, dict):
                            lane = extra.get("lane") or {}
                            dist = lane.get("dist")
                            angle = lane.get("angle_rad")
                            speed = extra.get("speed")
                            reward = extra.get("reward")
                            if dist is not None and angle is not None:
                                print(
                                    f"step={self.num_timesteps} reward={reward:.3f} speed={speed:.2f} "
                                    f"dist={dist:.3f} angle={angle:.3f}"
                                )

                return True

        self._impl = _Cb(render=render, print_every_steps=print_every_steps)

    def sb3_callback(self):
        return self._impl


def make_duckietown_env(map_name: str, seed: int, headless: bool) -> gym.Env:
    """Create a DuckietownEnv with JetRacer-like (vel, steering) actions."""

    # Ensure pyglet headless mode is set BEFORE importing gym_duckietown.
    # (Duckietown sets pyglet options during import.)
    os.environ["PYGLET_HEADLESS"] = "true" if headless else "false"

    # Import locally so this file can be imported without Duckietown installed.
    from gym_duckietown.envs import DuckietownEnv

    env = DuckietownEnv(
        seed=seed,
        map_name=map_name,
        domain_rand=False,
        draw_curve=False,
        draw_bbox=False,
        max_steps=1000,
        camera_width=160,
        camera_height=120,
        # Makes lane_position available in info (and is generally useful for training)
        full_transparency=True,
    )

    # Rendering: for training, headless is typically faster.
    # Duckietown sets pyglet.options["headless"] in gym_duckietown/__init__.py.
    # If you want a preview window, pass --render during training and call env.render().
    _ = headless

    return env


def build_env_fn(args: argparse.Namespace) -> Callable[[], gym.Env]:
    def _thunk() -> gym.Env:
        env = make_duckietown_env(map_name=args.map_name, seed=args.seed, headless=not args.render)
        env = CenterLineRewardWrapper(env)
        env = ResizeNormalizeObs(env, width=args.obs_width, height=args.obs_height)
        return env

    return _thunk


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Duckietown centerline-following policy (JetRacer style).")

    parser.add_argument("--map-name", type=str, default="loop_empty", help="Duckietown map name")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--obs-width", type=int, default=84)
    parser.add_argument("--obs-height", type=int, default=84)

    parser.add_argument("--total-timesteps", type=int, default=200_000)
    # Default to 1 env to avoid multiprocessing + graphics/GL edge cases.
    # You can increase this once everything is stable.
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments (VecEnv)")

    parser.add_argument("--log-dir", type=str, default="runs/centerline")
    parser.add_argument("--save-path", type=str, default="models/centerline_ppo.zip")

    parser.add_argument("--render", action="store_true", help="Render a preview window (slower)")

    args = parser.parse_args()

    if args.render and args.n_envs != 1:
        raise RuntimeError("--render requires --n-envs 1 (rendering does not work with SubprocVecEnv).")

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.logger import configure
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    except ModuleNotFoundError as e:  # pragma: no cover
        missing = getattr(e, "name", None)
        if missing == "torch":
            raise RuntimeError(
                "Missing dependency: torch. Stable-Baselines3 requires PyTorch. "
                "Install torch in this conda env, then re-run."
            ) from e
        if missing == "stable_baselines3":
            raise RuntimeError(
                "Missing dependency: stable-baselines3. Install SB3 v1.x for classic Gym compatibility "
                "(e.g., stable-baselines3==1.8.0), then re-run."
            ) from e
        raise

    env_fns = [build_env_fn(args) for _ in range(args.n_envs)]
    vec_env = SubprocVecEnv(env_fns) if args.n_envs > 1 else DummyVecEnv(env_fns)

    # SB3 logger
    # TensorBoard is optional; fall back to stdout-only if not installed.
    format_strings = ["stdout"]
    try:
        from torch.utils.tensorboard import SummaryWriter as _SummaryWriter  # noqa: F401

        format_strings.append("tensorboard")
    except Exception:
        print("NOTE: tensorboard not installed; continuing with stdout logging only.")

    sb3_logger = configure(folder=args.log_dir, format_strings=format_strings)

    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        verbose=1,
        seed=args.seed,
        policy_kwargs={"normalize_images": False},
        n_steps=1024,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
    )
    model.set_logger(sb3_logger)

    viz_cb = TrainingVizCallback(render=args.render).sb3_callback()
    model.learn(total_timesteps=args.total_timesteps, callback=viz_cb)
    model.save(args.save_path)

    vec_env.close()
    print(f"Saved policy to: {args.save_path}")


if __name__ == "__main__":
    main()
