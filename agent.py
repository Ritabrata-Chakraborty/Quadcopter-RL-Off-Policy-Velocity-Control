"""Action selector with algorithm-aware exploration noise."""

import numpy as np
import torch

from parameter import ACTION_SIZE, EXPERIMENT_TYPE, OU_NOISE_DECAY_EPISODES, OU_NOISE_MAX_SIGMA, OU_NOISE_MIN_SIGMA
from utils import OUNoise


class Agent:
    """Wraps an Actor network with algorithm-appropriate exploration.

    TD3/DDPG: deterministic actor + Ornstein-Uhlenbeck noise during training.
    SAC: stochastic actor (entropy provides exploration, no OU noise).
    """

    def __init__(self, actor_net: torch.nn.Module, device: str = 'cpu'):
        self.device = device
        self.actor_net = actor_net
        self.algorithm = EXPERIMENT_TYPE

        if self.algorithm in ('TD3', 'DDPG'):
            self.noise = OUNoise(
                action_space=ACTION_SIZE,
                max_sigma=OU_NOISE_MAX_SIGMA,
                min_sigma=OU_NOISE_MIN_SIGMA,
                decay_period=OU_NOISE_DECAY_EPISODES,
            )

    def get_action(
        self,
        state: np.ndarray,
        is_training: bool = True,
        step: int = 0,
    ) -> np.ndarray:
        """Forward pass through actor, with appropriate exploration."""
        state_tensor = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device)

        if self.algorithm == 'SAC' and is_training:
            action, _ = self.actor_net.sample(state_tensor.unsqueeze(0))
            action = action.squeeze(0)
        else:
            # TD3/DDPG: deterministic + OU noise; SAC eval: deterministic mean
            action = self.actor_net(state_tensor)
            if is_training and self.algorithm in ('TD3', 'DDPG'):
                noise = torch.from_numpy(self.noise.get_noise(step)).float().to(self.device)
                action = torch.clamp(action + noise, -1.0, 1.0)

        return action.detach().cpu().numpy()

    def get_action_random(self) -> np.ndarray:
        """Uniform random action for initial exploration phase."""
        return np.random.uniform(-1.0, 1.0, ACTION_SIZE).astype(np.float32)
