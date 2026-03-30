"""Ray-based remote episode runner for distributed training."""

from typing import Optional

import ray
import torch

from model import Actor, SACActor
from parameter import (
    ACTION_SIZE,
    EXPERIMENT_TYPE,
    HIDDEN_SIZE,
    NUM_GPU,
    NUM_META_AGENT,
    SAVE_IMG_GAP,
    STATE_SIZE,
    USE_GPU,
)
from worker import Worker


class Runner:
    """Local episode runner wrapping Worker with network weight management."""

    def __init__(self, meta_agent_id: int):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        if EXPERIMENT_TYPE == 'SAC':
            self.network = SACActor(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(self.device)
        else:
            self.network = Actor(STATE_SIZE, ACTION_SIZE, HIDDEN_SIZE).to(self.device)

    def set_policy_net_weights(self, weights: dict) -> None:
        self.network.load_state_dict(weights)

    def do_job(self, episode_number: int, use_policy: bool, policy_episode: int) -> tuple[list, dict, Optional[dict]]:
        save_img = episode_number % SAVE_IMG_GAP == 0
        worker = Worker(
            self.meta_agent_id, self.network, episode_number,
            use_policy=use_policy, policy_episode=policy_episode,
            device=self.device, save_image=save_img,
        )
        worker.run_episode()

        viz_data = None
        if worker.trajectory is not None:
            viz_data = {
                'trajectory': worker.trajectory,
                'trajectory_yaw': worker.trajectory_yaw,
                'trajectory_roll': worker.trajectory_roll,
                'trajectory_pitch': worker.trajectory_pitch,
                'cumulative_rewards': worker.cumulative_rewards,
                'belief_snapshots': worker.belief_snapshots,
                'lidar_snapshots': worker.lidar_snapshots,
                'action_snapshots': worker.action_snapshots,
                'ground_truth': worker.ground_truth,
                'map_name': worker.map_name,
            }
        episode_buffer = worker.episode_buffer
        perf_metrics = worker.perf_metrics
        del worker
        return episode_buffer, perf_metrics, viz_data

    def job(self, weights_set: list, episode_number: int, use_policy: bool = False, policy_episode: int = 0) -> tuple[list, dict, dict, Optional[dict]]:
        print(f"Starting episode {episode_number} on metaAgent {self.meta_agent_id}")
        self.set_policy_net_weights(weights_set[0])
        job_results, metrics, viz_data = self.do_job(episode_number, use_policy, policy_episode)
        info = {"id": self.meta_agent_id, "episode_number": episode_number}
        return job_results, metrics, info, viz_data


@ray.remote(num_cpus=1, num_gpus=NUM_GPU / NUM_META_AGENT)
class RLRunner(Runner):
    """Ray remote wrapper for distributed training."""

    def __init__(self, meta_agent_id: int):
        super().__init__(meta_agent_id)
