<div align="center">
  <h1>Quadcopter RL Off-Policy Velocity Control</h1>
  <h3>Multi-Critic Off-Policy Reinforcement Learning for Quadrotor Velocity Control in Unknown Indoor Environments</h3>
  
  [Ritabrata Chakraborty](https://github.com/caoyuhong001), Kaushal Kishore
  
  <sup>BITS Pilani, CSIR-CEERI</sup>
  
  [![Paper](https://img.shields.io/badge/Paper-TAI%202026-2b9348.svg?logo=arXiv)](https://arxiv.org/abs/your-paper-url)&nbsp;

</div>

## Introduction

This work implements a hierarchical quadcopter-based indoor exploration framework with a **multi-critic velocity controller** for autonomous navigation in unknown environments.

**Stage 1: Velocity-Based Point Navigation** involves a learned velocity controller that directly regulates forward, lateral, and angular velocities to reach goal positions. Details are provided in the [Control Architecture](#control-architecture-stage-1) section below.

![Navigation Demo](experiments/DDPG/train/plots/gifs/outdoor_1_ep40000_success.gif)

**Stage 2: Frontier-Based Exploration** uses CogniPlan for selecting exploration frontiers. We construct a path using A* search from the current location to the selected frontier and discretize it into waypoints at regular intervals. This approach enables robust navigation through maze-like environments and handles moving obstacles. The velocity controller from Stage 1 sequentially navigates between waypoints until reaching each frontier, then selects the next frontier and repeats.

## Control Architecture (Stage 1)

A **velocity-based controller** regulates three control axes: forward velocity ($\dot{x}$), lateral velocity ($\dot{y}$), and angular velocity ($\dot{\psi}$).

### Comparison Study

We compare three off-policy algorithms (`DDPG`, `TD3`, `SAC`) with two replay strategies:
- **Replay Method**: Uniform sampling vs. Prioritized Experience Replay (PER)
- **Critic Architecture**: Single critic vs. Multi-critic decomposition

### Single-Critic Baseline
A unified critic network outputs Q-values for all three actions jointly under a shared reward signal.

### Multi-Critic Decomposition (3 Critics)
Instead of a monolithic reward, we decompose the learning problem into three specialized critics:

| Critic | Controls | Reward Signal | Objective |
|--------|----------|---------------|-----------|
| **Critic 1** | Forward velocity ($\dot{x}$) | Progress-based reward | Maximize forward speed toward goal |
| **Critic 2** | Lateral velocity ($\dot{y}$) | Obstacle penalty | Maintain safe distance from walls |
| **Critic 3** | Angular velocity ($\dot{\psi}$) | Alignment reward | Align quadcopter heading with motion direction |

### Shared Reward Signals

All critics receive unified signals for episode termination:
- **Success**: +2500 (goal reached within threshold)
- **Crash**: −2000 (collision detected)
- **Timeout**: −100 (max episode steps exceeded)

Additionally, an **obstacle penalty** of −20 is applied at each step when obstacles are within a 1.0 m threshold, encouraging safe distance maintenance.

## Features

- **Algorithms**: Off-Policy TD3, DDPG, SAC
- **Rollouts**: Ray Multi-Worker
- **Replay**: Uniform or Prioritized Experience Replay (PER)
- **Logging**: TensorBoard, Weights & Biases
- **Critics**: Single critic or multi-critic decomposition (forward, lateral, angular velocity)
- **Tasks**: Point goal navigation, frontier-based exploration

## Setup

### Conda (recommended)

From the repository root:

```bash
conda env create -f environment.yml
conda activate DRLNav
```

## Configuration (**`parameter.py`**)

| Area | Examples |
|------|----------|
| Experiment | `EXPERIMENT_NAME`, `EXPERIMENT_TYPE` (`TD3` / `DDPG` / `SAC`) |
| Directories | `CHECKPOINT_DIR`, `TENSORBOARD_DIR`, `TRAIN_PLOTS_DIR`, `EVAL_DIR` |
| Checkpointing | `LOAD_MODEL`, `CHECKPOINT_EVERY`, `SAVE_IMG_GAP`, `SUMMARY_WINDOW` |
| WandB | `WANDB_ENABLED`, `WANDB_PROJECT`, `WANDB_ENTITY` |
| Network | `NUM_SCAN_SAMPLES`, `STATE_SIZE`, `ACTION_SIZE`, `HIDDEN_SIZE` |
| Training | `MAX_EPISODE_STEP`, `REPLAY_SIZE`, `BATCH_SIZE`, `NUM_META_AGENT`, `LR`, `GAMMA`, `TAU` |
| TD3 | `POLICY_NOISE`, `POLICY_NOISE_CLIP`, `POLICY_UPDATE_FREQUENCY` |
| DDPG | `POLICY_UPDATE_FREQUENCY` |
| SAC | `LOG_STD_MIN`, `LOG_STD_MAX`, `SAC_ALPHA_INIT`, `SAC_ALPHA_LR`, `SAC_TARGET_ENTROPY` |
| OU Noise | `OU_NOISE_MAX_SIGMA`, `OU_NOISE_MIN_SIGMA`, `OU_NOISE_DECAY_EPISODES` |
| Environment | `PHYSICS_TS`, `DRL_STEP_DURATION`, `EPISODE_TIMEOUT`, `GOAL_THRESHOLD`, `HOVER_ALTITUDE` |
| Map | `TRAIN_MAPS_DIR`, `TRAIN_GOALS_DIR`, `EVAL_MAPS_DIR`, `EVAL_GOALS_DIR`, `MAP_CELL_SIZE`, `MAP_PIXELS` |
| Sensing | `SENSOR_RANGE`, `COLLISION_RADIUS` |
| Reward | `REWARD_SUCCESS`, `REWARD_CRASH`, `REWARD_TIMEOUT`, `OBSTACLE_PENALTY`, `OBSTACLE_PENALTY_THRESHOLD` |
| Velocity | `SPEED_LINEAR_MAX`, `SPEED_LINEAR_Y_MAX`, `SPEED_ANGULAR_MAX` |
| PER | `USE_PER`, `PER_ALPHA`, `PER_BETA_START`, `PER_BETA_FRAMES`, `PER_EPSILON` |
| GPU | `USE_GPU`, `USE_GPU_GLOBAL`, `NUM_GPU` |


## Dataset

Download pre-generated maps and goals:

```bash
bash dataset/download.sh
```

Or generate custom environments:

```bash
bash scripts/generate_worlds.sh
```

## Training

```bash
python3 driver.py
```

## Evaluation

```bash
python3 test.py
```

Replay recorded actions from evaluation step CSVs:

```bash
python3 test_3d.py <path-to-steps-csv>
```

## Repository Layout

```
Quadrotor-RL-Off-Policy-Velocity-Control/
├── parameter.py              # Hyperparameters and Paths
├── driver.py                 # Training Loop and Logging
├── runner.py                 # Ray Remote Runner
├── worker.py                 # Episode Rollouts
├── model.py                  # Actor/Critic Networks
├── agent.py                  # Action Selection and OU Noise
├── env.py                    # Navigation Environment
├── utils.py                  # State, Lidar, Buffers
├── test.py                   # Evaluation Runner
├── test_3d.py                # 3D Trajectory Replay
├── environment.yml           # Conda Environment
├── README.md
├── dataset/                  # Maps, Goals, World Generation
│   ├── colors.json
│   ├── colors.py
│   ├── download.sh
│   ├── world_gen/            # Generation Helpers
│   │   ├── goals.py
│   │   ├── simple.py
│   │   └── utils.py
│   ├── maps_train/           # Training Maps and Goals
│   └── maps_eval/            # Evaluation Maps and Goals
├── experiments/              # Experiment Outputs
│   └── <EXPERIMENT_NAME>/
│       ├── config.yaml
│       └── train/
│           ├── checkpoints/  # Model Checkpoints
│           ├── tensorboard/  # TensorBoard Logs
│           └── plots/
│               └── gifs/
├── Quadcopter_SimCon/        # Quadcopter Controller
├── scripts/                  # Utility Scripts
│   └── generate_worlds.sh
└── wandb/                    # Weights & Biases Logs
```

## Acknowledgments

This work builds upon the following foundational implementations:

- **Quadcopter Dynamics and Control**: [Quadcopter_SimCon](https://github.com/bobzwik/Quadcopter_SimCon)
- **Frontier-Based Exploration Framework**: [CogniPlan](https://github.com/marmotlab/CogniPlan)