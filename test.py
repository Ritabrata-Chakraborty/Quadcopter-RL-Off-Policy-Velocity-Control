#!/usr/bin/env python3
"""Evaluate a trained quadcopter point-nav checkpoint (TD3 / DDPG / SAC)."""

import csv
import json
import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ray
import torch

matplotlib.use('Agg')

from agent import Agent
from env import QuadNavEnv
from model import Actor, SACActor
from parameter import (
    ACTION_SIZE,
    CHECKPOINT_DIR,
    EVAL_DIR,
    EVAL_GOALS_DIR,
    EVAL_MAPS_DIR,
    EXPERIMENT_NAME,
    EXPERIMENT_TYPE,
    GOAL_THRESHOLD,
    HIDDEN_SIZE,
    MAX_EPISODE_STEP,
    NUM_META_AGENT,
    STATE_SIZE,
    TENSORBOARD_DIR,
)
from utils import compute_lowess, extract_tensorboard_metrics, save_navigation_episode_outputs

# ------------------------------------------------------------------
# Global configuration
# ------------------------------------------------------------------

SEED = 42

# ------------------------------------------------------------------
# Checkpoint loading
# ------------------------------------------------------------------

def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """Load actor from checkpoint. Returns (actor, train_episode)."""
    if EXPERIMENT_TYPE == 'SAC':
        actor = SACActor(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
    else:
        actor = Actor(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    actor.load_state_dict(checkpoint['actor_model'])
    actor.eval()
    episode = checkpoint.get('episode', '?')
    print(f"Loaded checkpoint: {checkpoint_path} (episode {episode})")
    return actor, episode


# ------------------------------------------------------------------
# Evaluation episode (module-level so EvalRunner can call it)
# ------------------------------------------------------------------

def run_eval_episode(agent: Agent, env: QuadNavEnv, record: bool = True) -> tuple:
    """Run one evaluation episode (no noise). Returns (metrics, viz_data)."""
    state = env.reset()
    done = False
    total_reward = 0.0

    viz_data = None
    if record:
        viz_data = {
            'trajectory': [(env.quad.pos[0], env.quad.pos[1], env.quad.pos[2])],
            'trajectory_yaw': [float(env.quad.euler[2])],
            'trajectory_roll': [float(env.quad.euler[0])],
            'trajectory_pitch': [float(env.quad.euler[1])],
            'cumulative_rewards': [0.0],
            'belief_snapshots': [env.robot_belief.copy()],
            'lidar_snapshots': [env.get_lidar()],
            'action_snapshots': [],
            'ground_truth': env.ground_truth.copy(),
            'map_name': env.map_name,
        }

    for step in range(MAX_EPISODE_STEP):
        action = agent.get_action(state, is_training=False)
        state, reward, done, info_dict = env.step(action)
        # Handle both single-critic (scalar) and multi-critic (array) rewards
        reward_scalar = float(np.sum(reward)) if isinstance(reward, np.ndarray) else float(reward)
        total_reward += reward_scalar
        if record:
            viz_data['trajectory'].append(
                (env.quad.pos[0], env.quad.pos[1], env.quad.pos[2])
            )
            viz_data['trajectory_yaw'].append(float(env.quad.euler[2]))
            viz_data['trajectory_roll'].append(float(env.quad.euler[0]))
            viz_data['trajectory_pitch'].append(float(env.quad.euler[1]))
            viz_data['cumulative_rewards'].append(
                viz_data['cumulative_rewards'][-1] + reward_scalar
            )
            viz_data['belief_snapshots'].append(env.robot_belief.copy())
            viz_data['lidar_snapshots'].append(env.get_lidar())
            viz_data['action_snapshots'].append(action.copy())
        if done:
            break

    goal_dist = float(np.linalg.norm(env.goal_pos - env.quad.pos[0:2]))
    success = bool(info_dict.get('success', False))
    crash = bool(info_dict.get('crash', False))
    metrics = {
        'reward': total_reward,
        'steps': step + 1,
        'success': success,
        'crash': crash,
        'timeout': not success and not crash,
        'travel_dist': float(env.travel_dist),
        'goal_distance': goal_dist,
        'goal_pos': env.goal_pos.tolist(),
        'start_pos': env.start_pos.tolist(),
        'start_corner_pixels': getattr(env, 'start_corner_pixels', None),
        'goal_corner_pixels': getattr(env, 'active_goal_corner_pixels', None),
        'initial_goal_dist': float(env.initial_goal_dist),
        'map_name': env.map_name,
    }
    return metrics, viz_data


# ------------------------------------------------------------------
# Ray remote eval worker
# ------------------------------------------------------------------

@ray.remote(num_cpus=1)
class EvalRunner:
    """Runs evaluation episodes in a Ray worker process."""

    def __init__(self) -> None:
        self.device = torch.device('cpu')
        if EXPERIMENT_TYPE == 'SAC':
            self.network = SACActor(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
        else:
            self.network = Actor(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE)
        self.network.eval()
        self.agent = None

    def set_weights(self, weights: dict) -> None:
        self.network.load_state_dict(weights)
        self.agent = Agent(self.network, self.device)

    def run(self, episode_idx: int) -> tuple:
        env = QuadNavEnv(episode_idx, maps_dir=EVAL_MAPS_DIR, goals_dir=EVAL_GOALS_DIR)
        metrics, viz_data = run_eval_episode(self.agent, env, record=True)
        return episode_idx, metrics, viz_data


# ------------------------------------------------------------------
# Training metrics extraction from tensorboard
# ------------------------------------------------------------------

def save_training_plots(metrics: dict, eval_dir: str) -> None:
    """Create and save training plots from tensorboard metrics."""
    if not metrics:
        print("No tensorboard metrics found, skipping training plots")
        return

    # Organize metrics by category
    losses = {}
    perf_metrics = {}

    for metric_name, values in metrics.items():
        if not values:
            continue
        steps, vals = zip(*values)
        steps, vals = np.array(steps), np.array(vals)

        if 'Losses' in metric_name:
            losses[metric_name.replace('Losses/', '')] = (steps, vals)
        elif 'Perf' in metric_name:
            perf_metrics[metric_name.replace('Perf/', '')] = (steps, vals)

    # Create losses plot (dynamic grid based on number of losses)
    if losses:
        n_losses = len(losses)
        # For multi-critic: 5 losses (Critic Loss + Critic_0/1/2_Loss + Actor Loss)
        # For single-critic: 2 losses (Critic Loss + Actor Loss)
        cols = min(3, n_losses)  # Max 3 columns
        rows = (n_losses + cols - 1) // cols  # Calculate needed rows
        figsize = (5*cols, 4*rows)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        fig.suptitle(f'{EXPERIMENT_NAME} - Training Losses', fontsize=14, fontweight='bold')

        # Handle case where axes is 1D (only 1 row) or scalar (only 1 subplot)
        if n_losses == 1:
            axes = np.array([axes])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(rows, cols) if axes.ndim == 1 else axes

        ax_idx = 0
        for metric_name, (steps, vals) in losses.items():
            row, col = ax_idx // cols, ax_idx % cols
            ax = axes[row, col] if axes.ndim > 1 else axes[ax_idx]
            lowess_vals = compute_lowess(steps, vals, frac=0.05)
            ax.plot(steps, vals, linewidth=0.8, color='#CCCCCC', alpha=0.6, label='Raw')
            ax.plot(steps, lowess_vals, linewidth=2.5, color='#1565C0', label='LOWESS')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title(metric_name, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper right')
            ax_idx += 1

        # Hide unused subplots
        for i in range(ax_idx, rows * cols):
            row, col = i // cols, i % cols
            ax = axes[row, col] if axes.ndim > 1 else axes[i]
            ax.set_visible(False)

        fig.tight_layout()
        losses_path = os.path.join(eval_dir, 'plots', 'Training_Losses.png')
        fig.savefig(losses_path, dpi=150)
        plt.close(fig)
        print(f"Training losses plot saved: {losses_path} ({n_losses} loss curves)")

    # Create performance metrics plot (dynamic grid based on number of metrics)
    if perf_metrics:
        n_perf = len(perf_metrics)
        cols = min(3, n_perf)
        rows = (n_perf + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_perf == 1:
            axes = np.array([axes])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        else:
            axes = axes.reshape(rows, cols) if axes.ndim == 1 else axes

        fig.suptitle(f'{EXPERIMENT_NAME} - Training Performance', fontsize=14, fontweight='bold')

        colors = ['#4CAF50', '#FF9800', '#2196F3', '#F44336', '#E91E63', '#9C27B0']
        ax_idx = 0
        for metric_name, (steps, vals) in perf_metrics.items():
            row, col = ax_idx // cols, ax_idx % cols
            ax = axes[row, col] if axes.ndim > 1 else axes[ax_idx]
            lowess_vals = compute_lowess(steps, vals, frac=0.1)
            color = colors[ax_idx % len(colors)]
            ax.plot(steps, vals, linewidth=0.8, color='#CCCCCC', alpha=0.6, label='Raw')
            ax.plot(steps, lowess_vals, linewidth=2.5, color=color, label='LOWESS')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Value')
            ax.set_title(metric_name, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper right')
            ax_idx += 1

        # Hide unused subplots
        for i in range(ax_idx, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].set_visible(False) if axes.ndim > 1 else axes[i].set_visible(False)

        fig.tight_layout()
        perf_path = os.path.join(eval_dir, 'plots', 'Training_Performance.png')
        fig.savefig(perf_path, dpi=150)
        plt.close(fig)
        print(f"Training performance plot saved: {perf_path}")


# ------------------------------------------------------------------
# Summary plot
# ------------------------------------------------------------------

def save_summary_plot(all_metrics: list, eval_dir: str, checkpoint_ep=None) -> None:
    """Save a 4-panel evaluation summary plot."""
    n = len(all_metrics)
    episodes = list(range(n))
    rewards = [m['reward'] for m in all_metrics]
    goal_dists = [m['goal_distance'] for m in all_metrics]
    travel_dists = [m['travel_dist'] for m in all_metrics]
    colors = [
        '#4CAF50' if m['success'] else ('#F44336' if m['crash'] else '#FF9800')
        for m in all_metrics
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    if checkpoint_ep is not None:
        suptitle = f'{EXPERIMENT_NAME}  |  Ckpt ep {checkpoint_ep}  |  Evaluation ({n} episodes)'
    else:
        suptitle = f'{EXPERIMENT_NAME}  |  Evaluation ({n} episodes)'
    fig.suptitle(suptitle, fontsize=13, fontweight='bold')

    axes[0].bar(episodes, rewards, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Cumulative Reward')
    axes[0].set_title('Reward per Episode', fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(episodes, goal_dists, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[1].axhline(y=GOAL_THRESHOLD, color='#FF1744', linestyle='--', linewidth=2, alpha=0.7,
                    label=f'Success ({GOAL_THRESHOLD}m)')
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Goal Distance (m)')
    axes[1].set_title('Final Goal Distance', fontweight='bold')
    axes[1].legend(fontsize=9, loc='upper right')
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(episodes, travel_dists, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    axes[2].set_xlabel('Episode')
    axes[2].set_ylabel('Travel Distance (m)')
    axes[2].set_title('Travel Distance', fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')

    successes = sum(1 for m in all_metrics if m['success'])
    crashes = sum(1 for m in all_metrics if m['crash'])
    timeouts = sum(1 for m in all_metrics if m['timeout'])
    labels, sizes, pie_colors = [], [], []
    if successes:
        labels.append(f'Success\n({successes})')
        sizes.append(successes)
        pie_colors.append('#4CAF50')
    if crashes:
        labels.append(f'Crash\n({crashes})')
        sizes.append(crashes)
        pie_colors.append('#F44336')
    if timeouts:
        labels.append(f'Timeout\n({timeouts})')
        sizes.append(timeouts)
        pie_colors.append('#FF9800')
    axes[3].pie(sizes, labels=labels, colors=pie_colors, autopct='%1.0f%%', startangle=90,
                textprops={'fontsize': 10})
    axes[3].set_title('Outcome Distribution', fontweight='bold')

    fig.tight_layout()
    plot_path = os.path.join(eval_dir, 'plots', 'Summary.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Summary plot saved: {plot_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ckpt_path = os.path.join(CHECKPOINT_DIR, 'latest.pth')
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        print("Train first with: python3 driver.py")
        return

    actor, train_episode = load_checkpoint(ckpt_path, torch.device('cpu'))
    weights = actor.state_dict()

    eval_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    eval_run_dir = os.path.join(EVAL_DIR, eval_name)
    os.makedirs(os.path.join(eval_run_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(eval_run_dir, 'gifs'), exist_ok=True)

    steps_dir = os.path.join(eval_run_dir, 'steps')
    os.makedirs(steps_dir, exist_ok=True)

    map_files = sorted(f for f in os.listdir(EVAL_MAPS_DIR) if f.lower().endswith('.png'))
    if not map_files:
        print(f"No PNG maps found in {EVAL_MAPS_DIR}")
        return
    n_ep = len(map_files)

    print(f"\nEvaluating {n_ep} map(s) from {EVAL_MAPS_DIR}")
    print(f"Goals:   {EVAL_GOALS_DIR}")
    print(f"Workers: {min(n_ep, NUM_META_AGENT)}")
    print(f"Output:  {eval_run_dir}/\n")

    # ---- Extract and plot training metrics from tensorboard ----
    print("Extracting training metrics from tensorboard...")
    if os.path.exists(TENSORBOARD_DIR):
        metrics = extract_tensorboard_metrics(TENSORBOARD_DIR)
        save_training_plots(metrics, eval_run_dir)
    else:
        print(f"Tensorboard directory not found: {TENSORBOARD_DIR}")

    # ---- Parallel evaluation with Ray ----
    print("\nStarting evaluation on maps...")
    ray.init(ignore_reinit_error=True)

    n_workers = min(n_ep, NUM_META_AGENT)
    eval_runners = [EvalRunner.remote() for _ in range(n_workers)]
    ray.get([r.set_weights.remote(weights) for r in eval_runners])

    # Seed the job queue and track ref -> runner for replenishment
    ep_queue = list(range(n_ep))
    ref_to_runner: dict = {}
    for runner in eval_runners:
        ep = ep_queue.pop(0)
        ref = runner.run.remote(ep)
        ref_to_runner[ref] = runner

    # Rolling collect loop
    all_results: dict = {}
    while ref_to_runner:
        [done_ref], _ = ray.wait(list(ref_to_runner), num_returns=1)
        ep_idx, metrics, viz_data = ray.get(done_ref)
        runner = ref_to_runner.pop(done_ref)
        all_results[ep_idx] = (metrics, viz_data)

        outcome = 'SUCCESS' if metrics['success'] else ('CRASH' if metrics['crash'] else 'TIMEOUT')
        mn = metrics.get('map_name', '?')
        print(f"Episode {ep_idx:3d} ({mn}): {outcome:7s} | reward={metrics['reward']:8.1f} | "
              f"steps={metrics['steps']:3d} | goal_dist={metrics['goal_distance']:.2f}m | "
              f"travel={metrics['travel_dist']:.1f}m")

        if ep_queue:
            ep = ep_queue.pop(0)
            ref = runner.run.remote(ep)
            ref_to_runner[ref] = runner

    ray.shutdown()

    # ---- Process results in deterministic episode order ----
    csv_path = os.path.join(eval_run_dir, 'summary.csv')
    csv_header = ['episode', 'map_name', 'outcome', 'reward', 'steps',
                  'goal_distance', 'travel_dist', 'initial_goal_dist',
                  'start_pos', 'goal_pos']

    all_metrics = []
    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)

        for ep in range(n_ep):
            metrics, viz_data = all_results[ep]
            all_metrics.append(metrics)

            outcome = 'SUCCESS' if metrics['success'] else ('CRASH' if metrics['crash'] else 'TIMEOUT')
            mn = metrics.get('map_name', str(ep))
            sp = metrics['start_pos']
            gp = metrics['goal_pos']
            csv_writer.writerow([
                ep, mn, outcome, f"{metrics['reward']:.2f}", metrics['steps'],
                f"{metrics['goal_distance']:.4f}", f"{metrics['travel_dist']:.2f}",
                f"{metrics['initial_goal_dist']:.2f}",
                f"{sp[0]:.4f} {sp[1]:.4f}", f"{gp[0]:.4f} {gp[1]:.4f}",
            ])

            if viz_data is not None:
                steps_csv_path = os.path.join(steps_dir, f"{ep:03d}_{mn}.csv")
                with open(steps_csv_path, 'w', newline='') as steps_file:
                    steps_writer = csv.writer(steps_file)
                    action_names = ['linear_velocity_x', 'lateral_velocity_y', 'angular_velocity']
                    headers = ['step'] + action_names[:ACTION_SIZE]
                    steps_writer.writerow(headers)
                    for step_idx, action in enumerate(viz_data['action_snapshots']):
                        action_vals = [f"{a:.4f}" for a in action]
                        steps_writer.writerow([step_idx] + action_vals)

                save_navigation_episode_outputs(
                    viz_data, metrics, ep,
                    os.path.join(eval_run_dir, 'plots'),
                    os.path.join(eval_run_dir, 'gifs'),
                    experiment_name=EXPERIMENT_NAME,
                    checkpoint_ep=train_episode,
                )

    print(f"\nStep data saved to: {steps_dir}/")

    # Summary
    n = len(all_metrics)
    successes = sum(1 for m in all_metrics if m['success'])
    crashes = sum(1 for m in all_metrics if m['crash'])
    timeouts = sum(1 for m in all_metrics if m['timeout'])
    avg_reward = float(np.mean([m['reward'] for m in all_metrics]))
    avg_steps = float(np.mean([m['steps'] for m in all_metrics]))
    avg_goal_dist = float(np.mean([m['goal_distance'] for m in all_metrics]))
    avg_travel = float(np.mean([m['travel_dist'] for m in all_metrics]))

    summary = {
        'experiment': EXPERIMENT_NAME,
        'eval_name': eval_name,
        'checkpoint': ckpt_path,
        'train_episode': train_episode,
        'seed': SEED,
        'eval_maps_dir': EVAL_MAPS_DIR,
        'eval_goals_dir': EVAL_GOALS_DIR,
        'num_episodes': n,
        'num_workers': n_workers,
        'successes': successes,
        'crashes': crashes,
        'timeouts': timeouts,
        'success_rate': successes / n,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'avg_goal_distance': avg_goal_dist,
        'avg_travel_dist': avg_travel,
        'episodes': all_metrics,
    }

    with open(os.path.join(eval_run_dir, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    save_summary_plot(all_metrics, eval_run_dir, checkpoint_ep=train_episode)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Episodes:     {n}")
    print(f"Success Rate: {successes}/{n} ({100*successes/n:.1f}%)")
    print(f"Crash Rate:   {crashes}/{n} ({100*crashes/n:.1f}%)")
    print(f"Timeout Rate: {timeouts}/{n} ({100*timeouts/n:.1f}%)")
    print(f"Avg Reward:   {avg_reward:.1f}")
    print(f"Avg Steps:    {avg_steps:.1f}")
    print(f"Avg Goal Distance (Final): {avg_goal_dist:.2f}m")
    print(f"Avg Travel Distance:       {avg_travel:.1f}m")
    print(f"\nResults saved to: {eval_run_dir}/")
    print("  metrics.json | summary.csv | plots/ | gifs/ | steps/")


if __name__ == "__main__":
    main()
