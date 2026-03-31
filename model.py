"""Actor and Critic networks for quadcopter point navigation (TD3, DDPG, SAC)."""

import torch
import torch.nn as nn

from parameter import ACTION_SIZE, HIDDEN_SIZE, LOG_STD_MAX, LOG_STD_MIN, NUM_CRITICS, STATE_SIZE


# ------------------------------------------------------------------
# Weight initialisation
# ------------------------------------------------------------------

def init_linear_xavier(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)


# ------------------------------------------------------------------
# Actor
# ------------------------------------------------------------------

class Actor(nn.Module):
    """3-layer MLP: state -> tanh action in [-1, 1]."""

    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
        hidden_size: int = HIDDEN_SIZE,
    ):
        super().__init__()
        self.fa1 = nn.Linear(state_size, hidden_size)
        self.fa2 = nn.Linear(hidden_size, hidden_size)
        self.fa3 = nn.Linear(hidden_size, action_size)
        self.apply(init_linear_xavier)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fa1(states))
        x = torch.relu(self.fa2(x))
        return torch.tanh(self.fa3(x))


# ------------------------------------------------------------------
# Critic
# ------------------------------------------------------------------

class Critic(nn.Module):
    """Twin Q-networks: (state, action) -> (Q1, Q2)."""

    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
        hidden_size: int = HIDDEN_SIZE,
    ):
        super().__init__()
        half = hidden_size // 2

        # Q1
        self.l1 = nn.Linear(state_size, half)
        self.l2 = nn.Linear(action_size, half)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, 1)

        # Q2
        self.l5 = nn.Linear(state_size, half)
        self.l6 = nn.Linear(action_size, half)
        self.l7 = nn.Linear(hidden_size, hidden_size)
        self.l8 = nn.Linear(hidden_size, 1)

        self.apply(init_linear_xavier)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        xs = torch.relu(self.l1(states))
        xa = torch.relu(self.l2(actions))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.l3(x))
        q1 = self.l4(x)

        xs = torch.relu(self.l5(states))
        xa = torch.relu(self.l6(actions))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.l7(x))
        q2 = self.l8(x)

        return q1, q2

    def Q1_forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward pass through Q1 only (used for actor loss)."""
        xs = torch.relu(self.l1(states))
        xa = torch.relu(self.l2(actions))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.l3(x))
        return self.l4(x)


# ------------------------------------------------------------------
# DDPG Critic (single Q-network)
# ------------------------------------------------------------------

class DDPGCritic(nn.Module):
    """Single Q-network for DDPG: (state, action) -> Q."""

    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
        hidden_size: int = HIDDEN_SIZE,
    ):
        super().__init__()
        half = hidden_size // 2
        self.l1 = nn.Linear(state_size, half)
        self.l2 = nn.Linear(action_size, half)
        self.l3 = nn.Linear(hidden_size, hidden_size)
        self.l4 = nn.Linear(hidden_size, 1)
        self.apply(init_linear_xavier)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        xs = torch.relu(self.l1(states))
        xa = torch.relu(self.l2(actions))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.l3(x))
        return self.l4(x)

    def Q1_forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Alias for forward (unified interface with multi-critic)."""
        return self.forward(states, actions)


# ------------------------------------------------------------------
# SAC Actor (stochastic with reparameterization trick)
# ------------------------------------------------------------------

class SACActor(nn.Module):
    """Stochastic actor for SAC: state -> (action, log_prob).

    Uses reparameterization trick with tanh squashing.
    """

    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
        hidden_size: int = HIDDEN_SIZE,
    ):
        super().__init__()
        self.fa1 = nn.Linear(state_size, hidden_size)
        self.fa2 = nn.Linear(hidden_size, hidden_size)
        self.mean_head = nn.Linear(hidden_size, action_size)
        self.log_std_head = nn.Linear(hidden_size, action_size)
        self.apply(init_linear_xavier)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Deterministic forward: returns tanh(mean). Used for evaluation."""
        x = torch.relu(self.fa1(states))
        x = torch.relu(self.fa2(x))
        return torch.tanh(self.mean_head(x))

    def sample(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Stochastic forward for training. Returns (action, log_prob).

        Uses reparameterization trick: action = tanh(mean + std * eps).
        Log probability includes the tanh squashing correction.
        """
        x = torch.relu(self.fa1(states))
        x = torch.relu(self.fa2(x))
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()                         # reparameterization trick
        action = torch.tanh(u)

        # log pi(a|s) = log N(u; mean, std) - sum log(1 - tanh(u)^2)
        log_prob = normal.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


# ------------------------------------------------------------------
# Multi-Critic wrappers
# ------------------------------------------------------------------

class MultiCritic(nn.Module):
    """N independent twin-Q critics for TD3/SAC multi-critic decomposition.

    forward()    -> list of N (Q1_i, Q2_i) tuples
    Q1_forward() -> sum of Q1 across all critics (used for actor loss)
    """

    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_critics: int = NUM_CRITICS,
    ):
        super().__init__()
        self.critics = nn.ModuleList(
            [Critic(state_size, action_size, hidden_size) for _ in range(num_critics)]
        )

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> list:
        return [c(states, actions) for c in self.critics]

    def Q1_forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return sum(c.Q1_forward(states, actions) for c in self.critics)


class MultiDDPGCritic(nn.Module):
    """N independent single-Q critics for DDPG multi-critic decomposition.

    forward()    -> list of N Q tensors
    Q1_forward() -> sum of Q across all critics (used for actor loss)
    """

    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
        hidden_size: int = HIDDEN_SIZE,
        num_critics: int = NUM_CRITICS,
    ):
        super().__init__()
        self.critics = nn.ModuleList(
            [DDPGCritic(state_size, action_size, hidden_size) for _ in range(num_critics)]
        )

    def forward(
        self, states: torch.Tensor, actions: torch.Tensor
    ) -> list:
        return [c(states, actions) for c in self.critics]

    def Q1_forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return sum(c(states, actions) for c in self.critics)
