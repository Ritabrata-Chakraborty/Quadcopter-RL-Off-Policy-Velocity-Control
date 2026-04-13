## Claude Skills Context

domain: robotics
description: Off-policy deep RL (TD3, DDPG, SAC) for quadcopter point-navigation with velocity control on 2D occupancy maps. Distributed training via Ray, prioritized experience replay, multi-critic decomposition, and full checkpoint/resume support.

primary-skills:
  - pytorch-lightning
  - stable-baselines3
  - optimize-for-gpu

---

## Project Overview

Quadcopter learns to navigate from a start pose to a goal on a 2D occupancy map using velocity commands `[linear_x, lateral_y, angular]` ∈ `[-1, 1]`. The physics simulation uses **Quadcopter_SimCon**. Training is distributed across Ray workers; each worker runs one episode and returns transitions to a central driver.

---

## Architecture

### Algorithms
- **TD3** — twin delayed DDPG with target policy smoothing (`POLICY_UPDATE_FREQUENCY=2`)
- **DDPG** — single Q-network, actor updated every step
- **SAC** — entropy-regularised, stochastic actor, no actor target, `log_alpha` auto-tuned
- Switchable via `EXPERIMENT_TYPE` in `parameter.py`

### Multi-Critic (`USE_MULTI_CRITIC`)
Decomposes reward into 3 sub-signals (forward vel / lateral vel / angular-yaw) fed to independent critics. Actor loss uses summed Q1 across all critics (`Q1_forward`).

### Replay Buffer
- **Uniform**: `ReplayBuffer` — pre-allocated numpy arrays, circular write pointer
- **PER**: `PrioritizedReplayBuffer` — SumTree with proportional sampling; IS weights applied to all critic losses. `frame` counter (beta annealing) is inside the pickled buffer object.

### State / Action
- **State** (45-dim): `[40 lidar bins | goal_dist_norm | goal_angle_norm | prev_linear_x | prev_lateral_y | prev_angular]`
- **Action** (3-dim): `[linear_x, lateral_y, angular]` ∈ `[-1, 1]`; converted to physical velocities in `env.step()`

### Stability Metric
Computed from the episode's velocity commands (actions) via jerk magnitude. Logged as `Perf/Stability` in TensorBoard. No cross-episode state — purely per-episode.

---

## File Structure

```
driver.py      — central training loop: Ray dispatch, gradient steps, checkpoint/buffer save
runner.py      — Ray remote wrapper (RLRunner) around episode worker
worker.py      — single episode execution; collects transitions and metrics
env.py         — QuadNavEnv: Gym-like interface wrapping Quadcopter_SimCon
agent.py       — action selection with OU noise (TD3/DDPG) or stochastic sampling (SAC)
model.py       — Actor, Critic, SACActor, DDPGCritic, MultiCritic, MultiDDPGCritic
utils.py       — ReplayBuffer, PrioritizedReplayBuffer, OUNoise, soft_update, visualization
parameter.py   — all hyperparameters and directory paths (single source of truth)
test.py        — evaluation: parallel Ray workers, CSV/JSON/GIF/PNG output
test_3d.py     — replay recorded CSV actions through SimCon for 3D visualization
```

---

## Checkpoint / Resume

Full resume state is stored in `<EXPERIMENT_DIR>/train/checkpoints/<episode>.pth`:

| Key | Contents |
|-----|----------|
| `actor_model`, `critic_model` | Network weights |
| `actor_target_model`, `critic_target_model` | Target network weights |
| `actor_optimizer`, `critic_optimizer` | AdamW full state (m/v/lr) |
| `log_alpha`, `alpha_optimizer` | SAC entropy coefficient (SAC only) |
| `episode`, `total_steps` | Training counters |
| `policy_start_episode`, `training_active` | OU noise decay continuity |
| `last_actor_loss` | Prevents 0.0 spike in first TB window after resume |
| `np_rng_state`, `torch_rng_state`, `cuda_rng_state` | RNG reproducibility |
| `experiment_type`, `multi_critic` | Config guard (mismatch raises `ValueError`) |

Replay buffer is pickled separately to `<EXPERIMENT_DIR>/train/buffer/<episode>.pkl` (includes PER SumTree, `frame`, `max_priority`, `write` pointer).

TensorBoard event files are **preserved** across restarts. `sync_logs()` validates continuity (gap > `SUMMARY_WINDOW` raises `RuntimeError`) but does not delete old files.

---

## Training

```bash
# Configure: set EXPERIMENT_NAME, EXPERIMENT_TYPE, LOAD_MODEL in parameter.py
python driver.py

# Evaluate latest checkpoint
python test.py

# 3D replay from eval CSV
python test_3d.py experiments/<name>/eval/<run>/steps/000_<map>.csv
```

---

## Key Parameters (`parameter.py`)

| Parameter | Default | Notes |
|-----------|---------|-------|
| `EXPERIMENT_TYPE` | `'TD3'` | `'TD3'`, `'DDPG'`, `'SAC'` |
| `USE_MULTI_CRITIC` | `False` | 3-critic decomposed reward |
| `USE_PER` | `True` | Prioritized experience replay |
| `LOAD_MODEL` | `True` | Resume from latest checkpoint |
| `NUM_META_AGENT` | `8` | Parallel Ray episode workers |
| `MINIMUM_BUFFER_SIZE` | `25_000` | Episodes before policy training starts |
| `CHECKPOINT_EVERY` | `5000` | Save frequency (episodes) |
| `SAVE_BUFFER_EVERY` | `5000` | Buffer save frequency (episodes) |
| `SUMMARY_WINDOW` | `10` | TensorBoard logging window |

---

## Common Pitfalls

- **Mismatch `USE_MULTI_CRITIC`** between checkpoint and current config raises `ValueError` on load.
- **Missing buffer file** at exact episode falls back to `latest.pkl` with a warning (may be out of sync if buffer and checkpoint save intervals differ).
- **TensorBoard gap** > `SUMMARY_WINDOW` at resume raises `RuntimeError` — restore log files or set `LOAD_MODEL=False`.
- **`SAVE_BUFFER_EVERY` ≠ `CHECKPOINT_EVERY`** means checkpoint and buffer may not be from the same episode. The loader warns if it falls back to `latest.pkl`.
