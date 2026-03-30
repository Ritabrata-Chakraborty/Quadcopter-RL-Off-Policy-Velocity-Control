"""Episode runner for quadcopter point navigation on 2D maps."""

from typing import Union

import numpy as np
import torch

from agent import Agent
from env import QuadNavEnv
from parameter import MAX_EPISODE_STEP


class Worker:
    """Runs a single episode, collecting transitions and optional visualization data."""

    def __init__(
        self,
        meta_agent_id: int,
        actor_net: torch.nn.Module,
        episode_number: int,
        use_policy: bool,
        policy_episode: int,
        device: Union[str, torch.device] = 'cpu',
        save_image: bool = False,
    ):
        self.meta_agent_id = meta_agent_id
        self.episode_number = episode_number
        self.use_policy = use_policy
        self.policy_episode = policy_episode
        self.save_image = save_image
        self.device = device

        self.env = QuadNavEnv(episode_number)
        self.agent = Agent(actor_net, device)

        self.episode_buffer = []
        self.perf_metrics = {}
        self.trajectory = None
        self.trajectory_yaw = None
        self.trajectory_roll = None
        self.trajectory_pitch = None
        self.cumulative_rewards = None
        self.belief_snapshots = None
        self.lidar_snapshots = None
        self.action_snapshots = None
        self.ground_truth = None
        self.map_name = None

    def run_episode(self) -> None:
        """Run one full episode, storing transitions and metrics."""
        state = self.env.reset()
        done = False
        total_reward = 0.0
        info = {}

        if self.save_image:
            self.trajectory = [(self.env.quad.pos[0], self.env.quad.pos[1], self.env.quad.pos[2])]
            self.trajectory_yaw = [float(self.env.quad.euler[2])]
            self.trajectory_roll = [float(self.env.quad.euler[0])]
            self.trajectory_pitch = [float(self.env.quad.euler[1])]
            self.cumulative_rewards = [0.0]
            self.ground_truth = self.env.ground_truth.copy()
            self.map_name = self.env.map_name
            self.belief_snapshots = [self.env.robot_belief.copy()]
            self.lidar_snapshots = [self.env.get_lidar()]
            self.action_snapshots = []

        for step in range(MAX_EPISODE_STEP):
            if not self.use_policy:
                action = self.agent.get_action_random()
            else:
                action = self.agent.get_action(state, is_training=True, step=self.policy_episode)

            next_state, reward, done, info = self.env.step(action)

            self.episode_buffer.append((
                state.copy(),
                action.copy(),
                np.array([reward], dtype=np.float32),
                next_state.copy(),
                np.array([float(done)], dtype=np.float32),
            ))

            total_reward += reward
            state = next_state

            if self.save_image:
                self.trajectory.append((self.env.quad.pos[0], self.env.quad.pos[1], self.env.quad.pos[2]))
                self.trajectory_yaw.append(float(self.env.quad.euler[2]))
                self.trajectory_roll.append(float(self.env.quad.euler[0]))
                self.trajectory_pitch.append(float(self.env.quad.euler[1]))
                self.cumulative_rewards.append(self.cumulative_rewards[-1] + float(reward))
                self.belief_snapshots.append(self.env.robot_belief.copy())
                self.lidar_snapshots.append(self.env.get_lidar())
                self.action_snapshots.append(action.copy())

            if done:
                break

        self.perf_metrics = {
            'travel_dist': self.env.travel_dist,
            'total_reward': total_reward,
            'success_rate': float(info.get('success', False)),
            'goal_distance': float(np.linalg.norm(self.env.goal_pos - self.env.quad.pos[0:2])),
            'goal_pos': self.env.goal_pos.copy(),
            'start_pos': self.env.start_pos.copy(),
            'start_corner_pixels': getattr(self.env, 'start_corner_pixels', None),
            'goal_corner_pixels': getattr(self.env, 'active_goal_corner_pixels', None),
            'success': bool(info.get('success', False)),
            'crash': bool(info.get('crash', False)),
            'timeout': bool(info.get('timeout', False)),
            'map_name': self.env.map_name,
        }
        del self.env
