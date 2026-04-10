"""Hyperparameters and configuration for quadcopter point-navigation (TD3 / DDPG / SAC)."""

import math

# ------------------------------------------------------------------
# Experiment identity
# ------------------------------------------------------------------

EXPERIMENT_NAME = 'TEST'
EXPERIMENT_TYPE = 'DDPG'

# ------------------------------------------------------------------
# Directory structure
# ------------------------------------------------------------------

EXPERIMENT_DIR = f'experiments/{EXPERIMENT_NAME}'
CHECKPOINT_DIR = f'{EXPERIMENT_DIR}/train/checkpoints'
BUFFER_DIR = f'{EXPERIMENT_DIR}/train/buffer'
TENSORBOARD_DIR = f'{EXPERIMENT_DIR}/train/tensorboard'
TRAIN_PLOTS_DIR = f'{EXPERIMENT_DIR}/train/plots'
EVAL_DIR = f'{EXPERIMENT_DIR}/eval'

# ------------------------------------------------------------------
# Checkpointing and logging
# ------------------------------------------------------------------

SUMMARY_WINDOW = 10
LOAD_MODEL = True
SAVE_IMG_GAP = 5000
CHECKPOINT_EVERY = 2500
SAVE_BUFFER_EVERY = 2500        # save full replay buffer every N episodes (overwrites buffer_latest.pkl)

# ------------------------------------------------------------------
# Wandb
# ------------------------------------------------------------------

WANDB_ENABLED = True
WANDB_PROJECT = 'DRL-NAV'
WANDB_ENTITY = None

# ------------------------------------------------------------------
# Network
# ------------------------------------------------------------------

NUM_SCAN_SAMPLES = 40
STATE_SIZE = NUM_SCAN_SAMPLES + 5   # [40 lidar bins, goal_dist, goal_angle, prev_linear_x, prev_linear_y, prev_angular]
ACTION_SIZE = 3                     # linear_x (forward), linear_y (lateral), angular velocity
HIDDEN_SIZE = 512
LOG_STD_MIN = -20                   # SAC log-std clamp bounds (unconditional so model.py can import)
LOG_STD_MAX = 2

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------

MAX_EPISODE_STEP = 300              # 30 s at 0.1 s/step
REPLAY_SIZE = 200_000               # ~667 episodes of buffer at ~300 steps/ep
MINIMUM_BUFFER_SIZE = 25_000
BATCH_SIZE = 256                    # 45-dim state; larger batch not needed
GRADIENT_STEPS_PER_EPISODE = 2      # UTD ratio ≈ (2×256)/~150 trans/ep ≈ 3×
LR = 0.001                          # standard DDPG; avoids grad clipping every update
GAMMA = 0.99
TAU = 0.001                         # slower target tracking; more stable with lower LR
NUM_META_AGENT = 8

# ------------------------------------------------------------------
# TD3
# ------------------------------------------------------------------

if EXPERIMENT_TYPE == 'TD3':
    POLICY_NOISE = 0.2
    POLICY_NOISE_CLIP = 0.5
    POLICY_UPDATE_FREQUENCY = 2

# ------------------------------------------------------------------
# DDPG
# ------------------------------------------------------------------

if EXPERIMENT_TYPE == 'DDPG':
    POLICY_UPDATE_FREQUENCY = 1   # actor updates every step (no delay)

# ------------------------------------------------------------------
# SAC
# ------------------------------------------------------------------

if EXPERIMENT_TYPE == 'SAC':
    POLICY_UPDATE_FREQUENCY = 1
    SAC_ALPHA_INIT = 0.2
    SAC_ALPHA_LR = 0.004               # same as main LR
    SAC_TARGET_ENTROPY = -ACTION_SIZE  # = -3

# ------------------------------------------------------------------
# OU Noise
# ------------------------------------------------------------------

OU_NOISE_MAX_SIGMA = 0.2            # reduced from 0.3; environment not exploration-starved
OU_NOISE_MIN_SIGMA = 0.05
OU_NOISE_DECAY_EPISODES = 3_000     # reduced from 15_000; old DDPG converged at ~273 policy episodes

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------

PHYSICS_TS = 0.005
DRL_STEP_DURATION = 0.1
EPISODE_TIMEOUT = MAX_EPISODE_STEP * DRL_STEP_DURATION
GOAL_THRESHOLD = 0.8
HOVER_ALTITUDE = -1.0

# ------------------------------------------------------------------
# Map
# ------------------------------------------------------------------

TRAIN_MAPS_DIR = 'dataset/maps_train/simple'
TRAIN_GOALS_DIR = 'dataset/maps_train/simple_goals'
EVAL_MAPS_DIR = 'dataset/maps_eval/simple'
EVAL_GOALS_DIR = 'dataset/maps_eval/simple_goals'
MAP_CELL_SIZE = 0.2                                     # metres per cell
MAP_PIXELS = 250
MAP_SIDE_M = (MAP_PIXELS - 1) * MAP_CELL_SIZE           # metres
MAX_GOAL_DISTANCE = math.hypot(MAP_SIDE_M, MAP_SIDE_M)  # metres
COLLISION_RADIUS = 0.40                                 # metres; physical collision buffer around robot
DRONE_VIZ_SPAN_CELLS = 6                                # visual drone marker size in cells (~2.4 m at 0.2 m/cell)
FREE = 255
OCCUPIED = 1
UNKNOWN = 127

# ------------------------------------------------------------------
# Sensing
# ------------------------------------------------------------------

SENSOR_RANGE = 16.0

# ------------------------------------------------------------------
# Reward
# ------------------------------------------------------------------

REWARD_SUCCESS = 2500.0
REWARD_CRASH = -2000.0
REWARD_TIMEOUT = -100.0
OBSTACLE_PENALTY = -20.0
OBSTACLE_PENALTY_THRESHOLD = 0.80

# ------------------------------------------------------------------
# Velocity limits
# ------------------------------------------------------------------

SPEED_LINEAR_MAX = 3.0
SPEED_LINEAR_Y_MAX = 1.0   
SPEED_ANGULAR_MAX = math.radians(60)

# ------------------------------------------------------------------
# Multi-Critic
# ------------------------------------------------------------------

USE_MULTI_CRITIC = False   # True: 3 separate critics; False: single-critic baseline
NUM_CRITICS = 3            # only used when USE_MULTI_CRITIC is True

# ------------------------------------------------------------------
# Prioritized Experience Replay
# ------------------------------------------------------------------

USE_PER = False                    # set True to enable PER, False keeps uniform sampling
PER_ALPHA = 0.6                    # prioritization exponent (0=uniform, 1=full); only used if USE_PER=True
PER_BETA_START = 0.4               # initial IS exponent, anneals to 1.0
PER_BETA_FRAMES = 3_000 * 2        # 6K gradient steps; matches ~3000 policy episodes (post-fix convergence window)
PER_EPSILON = 1e-6                 # stability constant for priorities

# ------------------------------------------------------------------
# GPU
# ------------------------------------------------------------------

USE_GPU = False                    # per-worker GPU (Ray actors)
USE_GPU_GLOBAL = True              # central training device (driver.py)
NUM_GPU = 0                        # GPUs allocated per Ray worker


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def get_config_dict() -> dict:
    """Return all uppercase config values as a dict for wandb / yaml."""
    return {k: v for k, v in globals().items() if k.isupper() and not k.startswith('_')}
