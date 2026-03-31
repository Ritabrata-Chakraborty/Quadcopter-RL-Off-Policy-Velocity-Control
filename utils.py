"""Shared utilities for quadcopter point-navigation (TD3 / DDPG / SAC).

Provides state computation, lidar binning, OU noise, replay buffer, soft
update, and navigation visualization helpers used by driver.py and test.py.

When present in episode ``metrics`` (from goals JSON), ``start_corner_pixels`` and
``goal_corner_pixels`` align goal/start overlays with painted map squares.
"""

import math
import os
from typing import Any, Optional

import imageio
import numpy as np
from numpy import cos, pi, sin
import torch

from parameter import (
    DRONE_VIZ_SPAN_CELLS,
    DRL_STEP_DURATION,
    MAP_CELL_SIZE,
    MAX_GOAL_DISTANCE,
    NUM_SCAN_SAMPLES,
    OU_NOISE_DECAY_EPISODES,
    OU_NOISE_MAX_SIGMA,
    OU_NOISE_MIN_SIGMA,
    SPEED_ANGULAR_MAX,
    SPEED_LINEAR_MAX,
    SPEED_LINEAR_Y_MAX,
)


# ------------------------------------------------------------------
# Angle helpers
# ------------------------------------------------------------------

def normalize_angle(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


# ------------------------------------------------------------------
# Lidar binning
# ------------------------------------------------------------------

def bin_lidar(distances: np.ndarray, num_bins: int, distance_cap: float) -> np.ndarray:
    """Min-pool raw lidar distances into ``num_bins``, normalized to [0, 1].

    Args:
        distances: Raw ray distances in meters, shape ``(num_rays,)``.
        num_bins: Number of output bins (e.g. 40).
        distance_cap: Normalization cap in meters (e.g. 3.5).

    Returns:
        Binned distances, shape ``(num_bins,)``, clipped to [0, 1].
    """
    n = len(distances)
    bin_size = n / num_bins
    bins = np.ones(num_bins, dtype=np.float32)
    for i in range(num_bins):
        lo = int(i * bin_size)
        hi = int((i + 1) * bin_size)
        bins[i] = np.clip(np.min(distances[lo:hi]) / distance_cap, 0.0, 1.0)
    return bins


# ------------------------------------------------------------------
# State computation
# ------------------------------------------------------------------

def compute_state(
    lidar_bins: np.ndarray,
    quad: Any,
    goal_pos: np.ndarray,
    prev_action: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Build 45-dim state: [40 lidar bins | goal_dist | goal_angle | prev_linear_x | prev_linear_y | prev_angular].

    Returns:
        state: np.array of shape ``(45,)``, normalized.
        goal_dist: 2D distance to goal in meters.
    """
    diff_xy = goal_pos - quad.pos[0:2]
    goal_dist = float(np.linalg.norm(diff_xy))
    goal_angle = normalize_angle(np.arctan2(diff_xy[1], diff_xy[0]) - quad.euler[2])

    state = np.concatenate([
        lidar_bins,
        [np.clip(goal_dist / MAX_GOAL_DISTANCE, 0.0, 1.0)],
        [goal_angle / pi],
        [prev_action[0]],
        [prev_action[1]],
        [prev_action[2]],
    ]).astype(np.float32)

    return state, goal_dist


# ------------------------------------------------------------------
# OU Noise
# ------------------------------------------------------------------

class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise."""

    def __init__(
        self,
        action_space: int,
        mu: float = 0.0,
        theta: float = 0.15,
        max_sigma: float = OU_NOISE_MAX_SIGMA,
        min_sigma: float = OU_NOISE_MIN_SIGMA,
        decay_period: int = OU_NOISE_DECAY_EPISODES,
    ):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space
        self.reset()

    def reset(self) -> None:
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self) -> np.ndarray:
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_noise(self, t: int = 0) -> np.ndarray:
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, float(t) / self.decay_period)
        ou_state = self.evolve_state()
        return ou_state


# ------------------------------------------------------------------
# Replay buffer
# ------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity replay buffer using pre-allocated numpy arrays."""

    def __init__(self, capacity: int, reward_dim: int = 1):
        from parameter import STATE_SIZE, ACTION_SIZE
        self.capacity = capacity
        self.states = np.zeros((capacity, STATE_SIZE), dtype=np.float32)
        self.actions = np.zeros((capacity, ACTION_SIZE), dtype=np.float32)
        self.rewards = np.zeros((capacity, reward_dim), dtype=np.float32)
        self.next_states = np.zeros((capacity, STATE_SIZE), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.write = 0
        self.size = 0

    def add_sample(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> None:
        idx = self.write
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.write = (idx + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1

    def sample(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        indices = np.random.randint(0, self.size, size=min(batch_size, self.size))
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def get_length(self) -> int:
        return self.size


# ------------------------------------------------------------------
# Prioritized Experience Replay
# ------------------------------------------------------------------

class SumTree:
    """Binary tree where each leaf stores a priority value.

    Parent nodes store the sum of their children, enabling O(log n)
    proportional sampling and O(log n) priority updates.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self.propagate(parent, change)

    def retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = 2 * idx + 2
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self.retrieve(left, s)
        else:
            return self.retrieve(right, s - self.tree[left])

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        if idx != 0:
            self.propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, Any]:
        idx = self.retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Replay buffer with proportional prioritization via SumTree.

    Interface compatible with ReplayBuffer (add_sample, sample, get_length)
    plus update_priorities() for post-training-step priority correction.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 1e-6,
        reward_dim: int = 1,
    ):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.max_priority = 1.0
        self.frame = 0
        self.reward_dim = reward_dim

    def get_beta(self) -> float:
        return min(1.0, self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames)

    def add_sample(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> None:
        # max_priority already stores (|δ| + ε)^α; do not re-exponentiate.
        self.tree.add(self.max_priority, (state, action, reward, next_state, done))

    def sample(
        self, batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        from parameter import STATE_SIZE, ACTION_SIZE
        n = self.tree.n_entries
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        states = np.empty((batch_size, STATE_SIZE), dtype=np.float32)
        actions = np.empty((batch_size, ACTION_SIZE), dtype=np.float32)
        rewards = np.empty((batch_size, self.reward_dim), dtype=np.float32)
        next_states = np.empty((batch_size, STATE_SIZE), dtype=np.float32)
        dones = np.empty((batch_size, 1), dtype=np.float32)

        total = self.tree.total()
        segment = total / batch_size

        beta = self.get_beta()
        self.frame += 1

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s = np.random.uniform(lo, hi)
            idx, prio, data = self.tree.get(s)
            indices[i] = idx
            priorities[i] = prio
            states[i] = data[0]
            actions[i] = data[1]
            rewards[i] = data[2]
            next_states[i] = data[3]
            dones[i] = data[4]

        probs = priorities / total
        weights = (n * probs) ** (-beta)
        weights /= weights.max()

        return states, actions, rewards, next_states, dones, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, td_err in zip(indices, td_errors):
            priority = (abs(td_err) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), priority)
            self.max_priority = max(self.max_priority, priority)

    def get_length(self) -> int:
        return self.tree.n_entries


# ------------------------------------------------------------------
# Network helpers
# ------------------------------------------------------------------

def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """Polyak averaging: theta' = tau*theta + (1-tau)*theta'."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)


# ------------------------------------------------------------------
# Navigation visualization
# ------------------------------------------------------------------

TRAJECTORY_COLOR = '#1565C0'
LIDAR_HIT_COLOR  = '#CC00FF'
DRONE_ARM1_COLOR = '#E53935'
DRONE_ARM2_COLOR = '#1E88E5'
DRONE_TIP_COLOR  = 'white'
DRONE_TIP_EDGE   = '#212121'
GOAL_COLOR       = '#FF1744'
START_COLOR      = '#00E676'


def navigation_frame_stats(
    traj_xy: list,
    goal_pos_xy,
    step_idx: int,
    cumulative_rewards: Optional[list[float]],
    final_metrics: Optional[dict],
) -> tuple:
    """Cumulative reward, goal distance (m), travel distance (m) at ``step_idx``."""
    g = np.asarray(goal_pos_xy, dtype=np.float64).ravel()[:2]
    p = np.asarray(traj_xy[step_idx], dtype=np.float64).ravel()[:2]
    goal_dist = float(np.linalg.norm(g - p))
    if step_idx < 1:
        travel_dist = 0.0
    else:
        seg = np.asarray(traj_xy[: step_idx + 1], dtype=np.float64)
        travel_dist = float(np.sum(np.linalg.norm(np.diff(seg[:, :2], axis=0), axis=1)))
    if cumulative_rewards is not None and step_idx < len(cumulative_rewards):
        cum_r = float(cumulative_rewards[step_idx])
    elif final_metrics is not None:
        cum_r = float(final_metrics.get('total_reward', final_metrics.get('reward', 0.0)))
    else:
        cum_r = 0.0
    return cum_r, goal_dist, travel_dist


def plot_agent_marker(
    ax,
    world_x: float,
    world_y: float,
    yaw: float,
    zorder: int = 6,
    map_height: int = 250,
) -> None:
    """SimCon-style X cross with motor tips and center dot."""
    from env import world_to_cell

    span_m = DRONE_VIZ_SPAN_CELLS * MAP_CELL_SIZE
    arm_half = 0.5 * span_m
    c0, r0 = world_to_cell(world_x, world_y)
    r0 = map_height - 1 - r0

    ang1 = yaw + pi / 4
    ang2 = yaw - pi / 4

    # Arm 1 (red)
    c1_fwd, r1_fwd = world_to_cell(world_x + arm_half * cos(ang1), world_y + arm_half * sin(ang1))
    c1_back, r1_back = world_to_cell(world_x - arm_half * cos(ang1), world_y - arm_half * sin(ang1))
    r1_fwd = map_height - 1 - r1_fwd
    r1_back = map_height - 1 - r1_back
    ax.plot(
        [c1_back, c0, c1_fwd], [r1_back, r0, r1_fwd],
        color=DRONE_ARM1_COLOR, linewidth=3.5, solid_capstyle='round', zorder=zorder,
    )

    # Arm 2 (blue)
    c2_fwd, r2_fwd = world_to_cell(world_x + arm_half * cos(ang2), world_y + arm_half * sin(ang2))
    c2_back, r2_back = world_to_cell(world_x - arm_half * cos(ang2), world_y - arm_half * sin(ang2))
    r2_fwd = map_height - 1 - r2_fwd
    r2_back = map_height - 1 - r2_back
    ax.plot(
        [c2_back, c0, c2_fwd], [r2_back, r0, r2_fwd],
        color=DRONE_ARM2_COLOR, linewidth=3.5, solid_capstyle='round', zorder=zorder,
    )

    # Motor tips
    for c, r in [(c1_fwd, r1_fwd), (c1_back, r1_back), (c2_fwd, r2_fwd), (c2_back, r2_back)]:
        ax.plot(
            [c], [r], marker='o', markersize=5, markerfacecolor=DRONE_TIP_COLOR,
            markeredgecolor=DRONE_TIP_EDGE, markeredgewidth=0.5,
            linestyle='none', zorder=zorder + 1,
        )

    # Center dot
    ax.plot(
        [c0], [r0], marker='o', markersize=6, markerfacecolor=DRONE_ARM1_COLOR,
        markeredgecolor=DRONE_TIP_EDGE, markeredgewidth=0.5,
        linestyle='none', zorder=zorder + 1,
    )


def draw_map_marker_10x10(
    ax,
    col: float,
    row: float,
    facecolor: str,
    edgecolor: str,
    zorder: int = 5,
    *,
    size: int = 10,
) -> None:
    """Filled square marker on map at (col, row) in display coordinates."""
    from matplotlib.patches import Rectangle

    c0 = int(np.floor(float(col)))
    r0 = int(np.floor(float(row)))
    ax.add_patch(Rectangle(
        (c0, r0), size, size,
        facecolor=facecolor, edgecolor=edgecolor, linewidth=0.5, zorder=zorder,
    ))


def parse_corner_pixels_dict(corner: Optional[dict]) -> Optional[tuple[int, int, int, int]]:
    """Parse ``start_corner_pixels`` / ``goal_corner_pixels`` from goals JSON.

    Returns ``(top_left_row, top_left_col, height, width)`` in numpy image indexing
    (row 0 = top), or ``None`` if missing or invalid.
    """
    if corner is None or not isinstance(corner, dict):
        return None
    tl = corner.get('top_left')
    if not isinstance(tl, (list, tuple)) or len(tl) != 2:
        return None
    try:
        r0, c0 = int(tl[0]), int(tl[1])
        h = int(corner.get('height', 4))
        w = int(corner.get('width', 4))
    except (TypeError, ValueError):
        return None
    if h < 1 or w < 1:
        return None
    return r0, c0, h, w


def draw_pixel_patch_flipped(
    ax,
    top_left_row: int,
    top_left_col: int,
    height: int,
    width: int,
    map_height: int,
    facecolor: str,
    edgecolor: str,
    zorder: int = 5,
) -> None:
    """Draw a patch given numpy ``(row, col)`` top-left and size on ``imshow(flipud)`` axes."""
    from matplotlib.patches import Rectangle

    disp_row = map_height - top_left_row - height
    ax.add_patch(Rectangle(
        (top_left_col, disp_row), width, height,
        facecolor=facecolor, edgecolor=edgecolor, linewidth=0.5, zorder=zorder,
    ))


def draw_goal_start_markers(
    ax,
    map_height: int,
    metrics: dict,
    gc: int,
    gr_flipped: int,
    sc: int,
    sr_flipped: int,
) -> None:
    """Draw goal/start overlays: JSON corner pixels when present, else world→cell fallback."""
    g_rect = parse_corner_pixels_dict(metrics.get('goal_corner_pixels'))
    s_rect = parse_corner_pixels_dict(metrics.get('start_corner_pixels'))
    if g_rect is not None:
        r0, c0, h, w = g_rect
        draw_pixel_patch_flipped(
            ax, r0, c0, h, w, map_height, GOAL_COLOR, 'darkred', zorder=4,
        )
    else:
        draw_map_marker_10x10(ax, gc, gr_flipped, GOAL_COLOR, 'darkred', zorder=4, size=4)
    if s_rect is not None:
        r0, c0, h, w = s_rect
        draw_pixel_patch_flipped(
            ax, r0, c0, h, w, map_height, START_COLOR, 'darkgreen', zorder=4,
        )
    else:
        draw_map_marker_10x10(ax, sc, sr_flipped, START_COLOR, 'darkgreen', zorder=4, size=4)


def lidar_obstacle_points(
    lidar_endpoints,
    lidar_kinds: Optional[list[str]],
) -> list[tuple]:
    """Endpoints where the ray hit an obstacle (obstacle hits only, not free-space or boundary)."""
    if lidar_endpoints is None or lidar_kinds is None:
        return []
    if len(lidar_endpoints) != len(lidar_kinds):
        return []
    return [
        (float(ec), float(er))
        for (ec, er), k in zip(lidar_endpoints, lidar_kinds)
        if k == 'obstacle'
    ]


def render_navigation_frame(
    ground_truth: np.ndarray,
    belief: np.ndarray,
    traj_xy: list,
    traj_yaw: list,
    goal_pos,
    start_pos,
    lidar_endpoints,
    step_idx: int,
    total_steps: int,
    metrics: dict,
    episode: int,
    experiment_name: str = '',
    cumulative_rewards: Optional[list[float]] = None,
    lidar_kinds: Optional[list[str]] = None,
    checkpoint_ep=None,
    map_name: str = '',
    action_snapshots: Optional[list[np.ndarray]] = None,
    traj_roll: Optional[list[float]] = None,
    traj_pitch: Optional[list[float]] = None,
    trajectory: Optional[list] = None,
) -> 'matplotlib.figure.Figure':
    """Three-row figure: belief map + ground truth (top), time-series plots (bottom).

    Returns matplotlib Figure.
    """
    from env import world_to_cell
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(20, 9))
    gs = GridSpec(2, 6, figure=fig, height_ratios=[2.5, 1.5], hspace=0.45, wspace=0.4)
    ax1 = fig.add_subplot(gs[0, :3])   # belief map
    ax2 = fig.add_subplot(gs[0, 3:])   # ground truth
    ax3 = fig.add_subplot(gs[1, 0])    # x/y position
    ax4 = fig.add_subplot(gs[1, 1])    # roll
    ax5 = fig.add_subplot(gs[1, 2])    # pitch
    ax6 = fig.add_subplot(gs[1, 3])    # yaw
    ax7 = fig.add_subplot(gs[1, 4])    # linear velocity
    ax8 = fig.add_subplot(gs[1, 5])    # angular velocity

    traj_cols, traj_rows = [], []
    for x, y in traj_xy[: step_idx + 1]:
        c, r = world_to_cell(x, y)
        traj_cols.append(c)
        traj_rows.append(r)

    goal_pos = np.asarray(goal_pos, dtype=np.float64).ravel()[:2]
    start_pos = np.asarray(start_pos, dtype=np.float64).ravel()[:2]
    gc, gr = world_to_cell(goal_pos[0], goal_pos[1])
    sc, sr = world_to_cell(start_pos[0], start_pos[1])

    cur_x, cur_y = traj_xy[step_idx][0], traj_xy[step_idx][1]
    cur_yaw = traj_yaw[step_idx]

    cum_r, gd, td = navigation_frame_stats(
        traj_xy, goal_pos, step_idx, cumulative_rewards, metrics,
    )
    time_s = step_idx * DRL_STEP_DURATION

    # Flip maps vertically so North is displayed upward
    belief_display = np.flipud(belief)
    map_height = belief.shape[0]
    traj_rows_flipped = [map_height - 1 - r for r in traj_rows]
    gr_flipped = map_height - 1 - gr
    sr_flipped = map_height - 1 - sr

    # --- Belief panel ---
    ax1.imshow(belief_display, cmap='gray', vmin=0, vmax=255)
    ax1.set_title(f'Belief Map  ·  {NUM_SCAN_SAMPLES}-ray LiDAR', fontsize=9, pad=4)
    ax1.axis('off')
    if len(traj_cols) > 1:
        ax1.plot(traj_cols, traj_rows_flipped, color=TRAJECTORY_COLOR, linewidth=1.6, zorder=5)
    if lidar_endpoints is not None:
        hits = lidar_obstacle_points(lidar_endpoints, lidar_kinds)
        if hits:
            hx, hy = zip(*hits)
            hy_flipped = tuple(map_height - 1 - y for y in hy)
            ax1.plot(hx, hy_flipped, linestyle='none', marker='.', color=LIDAR_HIT_COLOR,
                     markersize=5, markeredgewidth=0, zorder=3)
    draw_goal_start_markers(ax1, map_height, metrics, gc, gr_flipped, sc, sr_flipped)
    plot_agent_marker(ax1, cur_x, cur_y, cur_yaw, zorder=6, map_height=map_height)

    # --- Ground truth panel ---
    ground_truth_display = np.flipud(ground_truth)
    ax2.imshow(ground_truth_display, cmap='gray', vmin=0, vmax=255)
    ax2.set_title(f'Ground Truth  ·  {map_name}', fontsize=9, pad=4)
    ax2.axis('off')
    if len(traj_cols) > 1:
        ax2.plot(traj_cols, traj_rows_flipped, color=TRAJECTORY_COLOR, linewidth=1.6, zorder=2)
    if lidar_endpoints is not None:
        hits = lidar_obstacle_points(lidar_endpoints, lidar_kinds)
        if hits:
            hx, hy = zip(*hits)
            hy_flipped = tuple(map_height - 1 - y for y in hy)
            ax2.plot(hx, hy_flipped, linestyle='none', marker='.', color=LIDAR_HIT_COLOR,
                     markersize=5, markeredgewidth=0, zorder=3)
    draw_goal_start_markers(ax2, map_height, metrics, gc, gr_flipped, sc, sr_flipped)
    plot_agent_marker(ax2, cur_x, cur_y, cur_yaw, zorder=6, map_height=map_height)

    # --- Time-series panels ---
    t_pos = np.arange(step_idx + 1) * DRL_STEP_DURATION
    xs = [traj_xy[k][0] for k in range(step_idx + 1)]
    ys = [traj_xy[k][1] for k in range(step_idx + 1)]
    zs = [trajectory[k][2] for k in range(step_idx + 1)] if trajectory else []
    rolls_deg  = [np.degrees(traj_roll[k])  for k in range(step_idx + 1)] if traj_roll  else []
    pitches_deg = [np.degrees(traj_pitch[k]) for k in range(step_idx + 1)] if traj_pitch else []
    yaws_deg   = [np.degrees(traj_yaw[k])   for k in range(step_idx + 1)]

    ax3.plot(t_pos, xs, color='#2196F3', linewidth=1.4, label='x')
    ax3.plot(t_pos, ys, color='#FF9800', linewidth=1.4, label='y')
    if zs:
        ax3.plot(t_pos, zs, color='#4CAF50', linewidth=1.4, label='z')
    ax3.set_xlabel('t (s)', fontsize=8)
    ax3.set_ylabel('m', fontsize=8)
    ax3.set_title('Position', fontsize=9, fontweight='bold')
    ax3.legend(fontsize=7, loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=7)

    if rolls_deg:
        ax4.plot(t_pos, rolls_deg, color='#E91E63', linewidth=1.4)
    ax4.set_xlabel('t (s)', fontsize=8)
    ax4.set_ylabel('deg', fontsize=8)
    ax4.set_title('Roll', fontsize=9, fontweight='bold')
    ax4.axhline(0, color='black', linewidth=0.6, alpha=0.4)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=7)

    if pitches_deg:
        ax5.plot(t_pos, pitches_deg, color='#FF5722', linewidth=1.4)
    ax5.set_xlabel('t (s)', fontsize=8)
    ax5.set_ylabel('deg', fontsize=8)
    ax5.set_title('Pitch', fontsize=9, fontweight='bold')
    ax5.axhline(0, color='black', linewidth=0.6, alpha=0.4)
    ax5.grid(True, alpha=0.3)
    ax5.tick_params(labelsize=7)

    ax6.plot(t_pos, yaws_deg, color='#9C27B0', linewidth=1.4)
    ax6.set_xlabel('t (s)', fontsize=8)
    ax6.set_ylabel('deg', fontsize=8)
    ax6.set_title('Yaw', fontsize=9, fontweight='bold')
    ax6.axhline(0, color='black', linewidth=0.6, alpha=0.4)
    ax6.grid(True, alpha=0.3)
    ax6.tick_params(labelsize=7)

    if action_snapshots and len(action_snapshots) > 0:
        n_act = min(step_idx, len(action_snapshots))
        t_act = np.arange(n_act) * DRL_STEP_DURATION
        lin_x_vels = [(action_snapshots[k][0] + 1.0) / 2.0 * SPEED_LINEAR_MAX for k in range(n_act)]
        lin_y_vels = [action_snapshots[k][1] * SPEED_LINEAR_Y_MAX for k in range(n_act)]
        ang_vels = [action_snapshots[k][2] * SPEED_ANGULAR_MAX for k in range(n_act)]
        ax7.plot(t_act, lin_x_vels, color='#4CAF50', linewidth=1.4, label='forward')
        ax7.plot(t_act, lin_y_vels, color='#2196F3', linewidth=1.4, label='lateral')
        ax7.legend(fontsize=7, loc='upper right')
        ax8.plot(t_act, ang_vels, color='#F44336', linewidth=1.4)

    ax7.set_xlabel('t (s)', fontsize=8)
    ax7.set_ylabel('m/s', fontsize=8)
    ax7.set_title('Linear Velocity', fontsize=9, fontweight='bold')
    ax7.set_ylim(-SPEED_LINEAR_Y_MAX * 1.05, SPEED_LINEAR_MAX * 1.05)
    ax7.axhline(0, color='black', linewidth=0.6, alpha=0.4)
    ax7.grid(True, alpha=0.3)
    ax7.tick_params(labelsize=7)

    ax8.set_xlabel('t (s)', fontsize=8)
    ax8.set_ylabel('rad/s', fontsize=8)
    ax8.set_title('Angular Velocity', fontsize=9, fontweight='bold')
    ax8.set_ylim(-SPEED_ANGULAR_MAX * 1.05, SPEED_ANGULAR_MAX * 1.05)
    ax8.axhline(0, color='black', linewidth=0.6, alpha=0.4)
    ax8.grid(True, alpha=0.3)
    ax8.tick_params(labelsize=7)

    # --- Title ---
    outcome = 'SUCCESS' if metrics.get('success') else ('CRASH' if metrics.get('crash') else 'TIMEOUT')
    if checkpoint_ep is not None:
        line1 = f'{experiment_name}  |  Checkpoint: {checkpoint_ep}  |  Map: {map_name}  |   Steps: {total_steps}  |  {outcome}'
    else:
        line1 = f'{experiment_name}  |  Train Episode: {episode}  |  Map: {map_name}  |  Total Steps: {total_steps}  |  {outcome}'
    line2 = (f'Step {step_idx}/{total_steps - 1}  |  t = {time_s:.1f} s  |  Reward: {cum_r:.1f}'
             f'  |  Goal Distance: {gd:.2f} m  |  Travel Distance: {td:.1f} m')
    fig.suptitle(f'{line1}\n{line2}', fontsize=10, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def save_navigation_episode_outputs(
    viz_data: dict,
    metrics: dict,
    episode: int,
    plots_dir: str,
    gifs_dir: str,
    experiment_name: str = '',
    checkpoint_ep=None,
) -> None:
    """Write 2-panel navigation GIF and final PNG for one episode."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    trajectory = viz_data['trajectory']
    trajectory_yaw = viz_data['trajectory_yaw']
    belief_snapshots = viz_data['belief_snapshots']
    lidar_snapshots = viz_data['lidar_snapshots']
    ground_truth = viz_data['ground_truth']
    map_name = viz_data.get('map_name', 'map')
    cumulative_rewards = viz_data.get('cumulative_rewards')
    action_snapshots = viz_data.get('action_snapshots')
    traj_roll = viz_data.get('trajectory_roll')
    traj_pitch = viz_data.get('trajectory_pitch')

    goal_pos = np.asarray(metrics['goal_pos'])
    start_pos = np.asarray(metrics['start_pos'])
    traj_xy = [(t[0], t[1]) for t in trajectory]
    n_steps = len(trajectory)

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(gifs_dir, exist_ok=True)

    outcome = (
        'SUCCESS' if metrics.get('success') else ('CRASH' if metrics.get('crash') else 'TIMEOUT')
    ).lower()
    stem = f'{map_name}_ep{episode}_{outcome}'

    gif_path = os.path.join(gifs_dir, f'{stem}.gif')
    with imageio.get_writer(gif_path, mode='I', duration=0.15) as gif_writer:
        for i in range(n_steps):
            snap = lidar_snapshots[i]
            if len(snap) == 3:
                _, endpoints, kinds = snap
            else:
                _, endpoints = snap
                kinds = None
            fig = render_navigation_frame(
                ground_truth, belief_snapshots[i], traj_xy, trajectory_yaw,
                goal_pos, start_pos, endpoints, i, n_steps, metrics, episode,
                experiment_name=experiment_name, cumulative_rewards=cumulative_rewards,
                lidar_kinds=kinds, checkpoint_ep=checkpoint_ep, map_name=map_name,
                action_snapshots=action_snapshots,
                traj_roll=traj_roll, traj_pitch=traj_pitch,
                trajectory=trajectory,
            )
            fig.set_dpi(72)
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            w, h = fig.canvas.get_width_height()
            img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
            gif_writer.append_data(img)
            plt.close(fig)

    # Final frame as PNG
    last_snap = lidar_snapshots[-1]
    if len(last_snap) == 3:
        _, last_endpoints, last_kinds = last_snap
    else:
        _, last_endpoints = last_snap
        last_kinds = None
    fig_final = render_navigation_frame(
        ground_truth, belief_snapshots[-1], traj_xy, trajectory_yaw,
        goal_pos, start_pos, last_endpoints, n_steps - 1, n_steps, metrics, episode,
        experiment_name=experiment_name, cumulative_rewards=cumulative_rewards,
        lidar_kinds=last_kinds, checkpoint_ep=checkpoint_ep, map_name=map_name,
        action_snapshots=action_snapshots,
        traj_roll=traj_roll, traj_pitch=traj_pitch,
        trajectory=trajectory,
    )
    png_path = os.path.join(plots_dir, f'{stem}.png')
    fig_final.savefig(png_path, dpi=150)
    plt.close(fig_final)

    print(f'  Saved: {gif_path}')
    print(f'  Saved: {png_path}')
