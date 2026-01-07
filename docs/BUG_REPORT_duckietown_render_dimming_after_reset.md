# Bug Report: Duckietown sim window becomes dim after first `reset()` / episode

**Date:** 2026-01-07  
**Project:** autonomous_rl_project (gym-duckietown / DuckietownEnv)

## Summary
When running Duckietown with a visible pyglet window (`env.render()`), the first episode renders at a normal/bright brightness, but after the first episode ends (i.e., after a subsequent `env.reset()`), the **entire window** becomes noticeably darker. The dimming persists for later episodes.

This is not just a different tile/texture looking darker; it appears as a global change in rendering/lighting state.

## Impact
- Makes qualitative evaluation (“enjoy” / demo) confusing because brightness changes between episodes.
- Can be mistaken for environment differences between training and evaluation.

## Environment
- Python: 3.8 (conda env `duckie_clean`)
- gym-duckietown: 6.2.0 (workspace source: `gym-duckietown/src`)
- Rendering: pyglet window (OpenGL)
- GPU/driver: NVIDIA (observed via Duckietown graphics info log)

## Reproduction steps
1. Run the enjoy script with rendering enabled:

   ```bash
   python enjoy_jetracer_centerline.py --model-path models/centerline_ppo.zip --render
   ```

2. Observe:
   - Episode 1 looks “normal/bright”.
   - After episode 1 ends and the environment resets, the window becomes globally darker.

Notes:
- The issue reproduces without `--keyboard` as well.
- The issue appears even when using the same `make_duckietown_env()` function and the same `style/domain_rand` settings.

## Expected behavior
Brightness/lighting should be consistent across episodes/resets for the same environment configuration.

## Actual behavior
Brightness changes after the first episode reset; subsequent episodes render darker.

## Root-cause hypothesis
Duckietown uses two pyglet windows/contexts:
- `window`: visible human-rendering window
- `shadow_window`: hidden window used for offscreen rendering (FBOs / observations)

During `Simulator.reset()` (called by `env.reset()`), Duckietown configures OpenGL lighting (`glLightfv`, `glEnable(GL_LIGHTING)`, etc.). If `reset()` happens while the **visible window context** is current, the lighting state can “stick” to the visible window context and produce a globally dim scene afterward.

In other words: **OpenGL state leaks into the visible window context across resets**.

## Workaround / Fix applied in this repo
We force `reset()` to run while the simulator’s `shadow_window` context is current.

Implemented in:
- `enjoy_jetracer_centerline.py`

Change:
- Before calling `env.reset()`, do:
  - `env.unwrapped.shadow_window.switch_to()` (best-effort)

This keeps lighting setup and offscreen rendering initialization in the hidden context, preventing the visible window from becoming dim after episode resets.

## Verification
After applying the workaround, brightness remains consistent across episodes when running:

```bash
python enjoy_jetracer_centerline.py --model-path models/centerline_ppo.zip --render
```

## Suggested upstream fix
In `gym_duckietown/simulator.py`:
- Ensure `reset()` (or the lighting setup inside it) executes under a known/expected OpenGL context, ideally the `shadow_window` context.
- Alternatively, explicitly reset relevant GL lighting state when rendering the visible window, so it does not depend on prior context state.

A minimal approach could be:
- Call `self.shadow_window.switch_to()` early in `reset()` before GL lighting configuration.

## Additional notes
This report focuses on visual consistency for the human-rendered window; it may also reduce hard-to-debug rendering differences between training/evaluation runs when resets happen at different times.
