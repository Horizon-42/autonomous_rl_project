"""Run (enjoy) a trained Duckietown PPO policy with JetRacer-style actions.

This re-creates the same wrappers used in `train_jetracer_centerline.py`, loads the
saved SB3 model `.zip`, and runs episodes in the environment.

Examples
--------
Headless (no window; works on servers):

    python enjoy_jetracer_centerline.py --model-path models/centerline_ppo.zip

Render (needs a working display/GL):

    python enjoy_jetracer_centerline.py --model-path models/centerline_ppo.zip --render

"""

from __future__ import annotations

import argparse

import numpy as np


def build_env(
    map_name: str,
    seed: int,
    obs_width: int,
    obs_height: int,
    render: bool,
    *,
    domain_rand: bool,
    style: str,
    user_tile_start,
):
    # Import from the training script to ensure wrapper parity.
    from train_jetracer_centerline import (
        JetRacerRaceRewardWrapper,
        JetRacerWrapper,
        ResizeNormalizeObs,
        make_duckietown_env,
    )

    env = make_duckietown_env(
        map_name=map_name,
        seed=seed,
        headless=not render,
        domain_rand=domain_rand,
        style=style,
        user_tile_start=user_tile_start,
    )
    env = JetRacerWrapper(env)
    env = JetRacerRaceRewardWrapper(env)
    env = ResizeNormalizeObs(env, width=obs_width, height=obs_height)
    return env


def _reset_with_shadow_context(env):
    """Reset while the simulator's shadow GL context is current.

    Duckietown uses a hidden `shadow_window` for offscreen rendering (observations/FBOs).
    If `reset()` runs while the visible window context is current, lighting state can end up
    being applied to the visible window, making later episodes look uniformly darker.
    """

    base = env.unwrapped
    shadow = getattr(base, "shadow_window", None)
    if shadow is not None:
        try:
            shadow.switch_to()
        except Exception:
            pass
    return env.reset()


class _KeyboardController:
    """Minimal keyboard control using Duckietown's pyglet window.

    Keys
    ----
    - Arrow keys: manual throttle/steering (when in MANUAL mode)
    - M: toggle MODEL <-> MANUAL
    - SPACE: zero throttle/steering
    - R: reset episode
    - Q / ESC: quit
    """

    def __init__(self, window):
        from pyglet.window import key

        self._key = key
        self._keys = key.KeyStateHandler()
        window.push_handlers(self._keys)
        window.push_handlers(self)

        self.mode = "model"  # or "manual"
        self.throttle = 0.0
        self.steering = 0.0

        self.should_quit = False
        self.should_reset = False

    def on_key_press(self, symbol, modifiers):  # noqa: ARG002
        if symbol in (self._key.Q, self._key.ESCAPE):
            self.should_quit = True
        elif symbol == self._key.R:
            self.should_reset = True
        elif symbol == self._key.SPACE:
            self.throttle = 0.0
            self.steering = 0.0
        elif symbol == self._key.M:
            self.mode = "manual" if self.mode == "model" else "model"
            print(f"[keyboard] mode -> {self.mode}")

    def sample_action(self) -> np.ndarray:
        # Incremental control feels better than a fixed value.
        inc_throttle = 0.04
        inc_steer = 0.10

        if self._keys[self._key.UP]:
            self.throttle += inc_throttle
        if self._keys[self._key.DOWN]:
            self.throttle -= inc_throttle

        if self._keys[self._key.LEFT]:
            self.steering -= inc_steer
        if self._keys[self._key.RIGHT]:
            self.steering += inc_steer

        # Mild decay back to center when no key is pressed.
        if not (self._keys[self._key.UP] or self._keys[self._key.DOWN]):
            self.throttle *= 0.98
        if not (self._keys[self._key.LEFT] or self._keys[self._key.RIGHT]):
            self.steering *= 0.90

        self.throttle = float(np.clip(self.throttle, 0.0, 1.0))
        self.steering = float(np.clip(self.steering, -1.0, 1.0))

        return np.array([self.throttle, self.steering], dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and run a trained Duckietown PPO policy.")

    parser.add_argument("--model-path", type=str, default="models/centerline_ppo.zip")
    parser.add_argument("--map-name", type=str, default="loop_empty")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--style",
        type=str,
        default="photos",
        choices=["photos", "synthetic"],
        help="Tile texture style (synthetic is usually brighter).",
    )
    parser.add_argument(
        "--domain-rand",
        action="store_true",
        help="Enable domain randomization (includes lighting changes; can look brighter).",
    )

    parser.add_argument(
        "--user-tile-start",
        nargs=2,
        type=int,
        default=None,
        metavar=("I", "J"),
        help="Force the spawn tile coords (i j) so each episode starts on the same tile.",
    )

    parser.add_argument("--obs-width", type=int, default=84)
    parser.add_argument("--obs-height", type=int, default=84)

    parser.add_argument("--render", action="store_true", help="Show a preview window (requires display/GL)")
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="Enable keyboard control (requires --render). Press M to toggle model/manual.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (less exploration; good for evaluation)",
    )

    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps-per-episode", type=int, default=5000)

    args = parser.parse_args()

    if args.keyboard and not args.render:
        raise RuntimeError("--keyboard requires --render (needs a pyglet window for key events).")

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "stable-baselines3 is not available in this environment. "
            "Activate your duckie env and install SB3 v1.x."
        ) from e

    env = build_env(
        map_name=args.map_name,
        seed=args.seed,
        obs_width=args.obs_width,
        obs_height=args.obs_height,
        render=args.render,
        domain_rand=args.domain_rand,
        style=args.style,
        user_tile_start=tuple(args.user_tile_start) if args.user_tile_start is not None else None,
    )

    model = PPO.load(args.model_path)

    kb = None
    if args.render and args.keyboard:
        # Force window creation, then attach key handlers to Duckietown's pyglet window.
        try:
            env.render()
        except Exception:
            pass

        base = env.unwrapped
        win = getattr(base, "window", None)
        if win is None:
            raise RuntimeError("Could not access Duckietown pyglet window; keyboard control unavailable.")
        kb = _KeyboardController(win)

    for ep in range(1, args.episodes + 1):
        obs = _reset_with_shadow_context(env)
        ep_return = 0.0
        ep_len = 0
        done_why = None

        for _t in range(args.max_steps_per_episode):
            if kb is not None and kb.should_quit:
                break

            if kb is not None and kb.should_reset:
                kb.should_reset = False
                obs = _reset_with_shadow_context(env)
                ep_return = 0.0
                ep_len = 0
                done_why = None
                continue

            if kb is not None and kb.mode == "manual":
                action = kb.sample_action()
            else:
                action, _state = model.predict(obs, deterministic=args.deterministic)

            obs, reward, done, info = env.step(action)

            ep_return += float(reward)
            ep_len += 1

            extra = info.get("race_reward") if isinstance(info, dict) else None
            if isinstance(extra, dict):
                done_why = extra.get("done_why", done_why)

            if args.render:
                try:
                    env.render()
                except Exception:
                    # Rendering can fail on headless/GL setups.
                    pass

            if done:
                break

        if kb is not None and kb.should_quit:
            break

        print(f"episode={ep} len={ep_len} return={ep_return:.2f} done_why={done_why}")

    env.close()


if __name__ == "__main__":
    main()
