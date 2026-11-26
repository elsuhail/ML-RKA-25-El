"""Double DQN agent, Replay Buffer, and baseline agents for Stealth Heist."""

from collections import deque
import random
from typing import Deque, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .stealth_env import ACTION_DELTAS, ACTION_STAY, StealthHeistEnv


class QNetwork(nn.Module):
    """3-layer MLP predicting Q(s, a) for each of the 5 discrete actions."""

    def __init__(self, state_dim: int = 22, action_dim: int = 5, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Fixed-size experience replay buffer storing terminated separately from truncated."""

    def __init__(self, capacity: int = 50_000, seed: int = 42) -> None:
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store a single transition; only terminated cuts off Bellman bootstrapping."""
        self.buffer.append(
            (
                np.array(state, dtype=np.float32, copy=True),
                int(action),
                float(reward),
                np.array(next_state, dtype=np.float32, copy=True),
                bool(terminated),
            )
        )

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a mini-batch and convert efficiently via NumPy to PyTorch tensors."""
        batch = self.rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminateds = zip(*batch)

        return (
            torch.from_numpy(np.asarray(states, dtype=np.float32)),
            torch.from_numpy(np.asarray(actions, dtype=np.int64)).unsqueeze(1),
            torch.from_numpy(np.asarray(rewards, dtype=np.float32)).unsqueeze(1),
            torch.from_numpy(np.asarray(next_states, dtype=np.float32)),
            torch.from_numpy(np.asarray(terminateds, dtype=np.float32)).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DoubleDQNAgent:
    """Double DQN agent with online/target networks, Huber loss, and epsilon-greedy policy."""

    def __init__(
        self,
        state_dim: int = 22,
        action_dim: int = 5,
        hidden_dim: int = 128,
        gamma: float = 0.98,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 150_000,
        learning_starts: int = 1_000,
        train_freq: int = 2,
        target_update_freq: int = 200,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        seed: int = 42,
        device: Optional[str] = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.target_update_freq = target_update_freq
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.epsilon = eps_start

        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.criterion = nn.SmoothL1Loss()  # Huber loss for stable TD errors
        self.buffer = ReplayBuffer(capacity=buffer_size, seed=seed)

        self.env_steps = 0
        self.grad_steps = 0

    def update_epsilon(self, episode: int, total_episodes: int) -> float:
        """Linearly decay epsilon from eps_start to eps_end over first 50% of episodes."""
        decay_episodes = max(1, int(0.50 * total_episodes))
        progress = min(1.0, episode / decay_episodes)
        self.epsilon = self.eps_start - progress * (self.eps_start - self.eps_end)
        return self.epsilon

    def get_q_values(self, obs: np.ndarray) -> np.ndarray:
        """Return predicted Q-values for all 5 actions as a 1D NumPy array."""
        self.online_net.eval()
        with torch.no_grad():
            state_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(self.device)
            q_vals = self.online_net(state_t).squeeze(0).cpu().numpy()
        return q_vals

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> int:
        """Select action using greedy policy (deterministic=True) or epsilon-greedy policy."""
        if not deterministic and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.action_dim)
        q_vals = self.get_q_values(obs)
        return int(np.argmax(q_vals))

    def step_update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> Optional[float]:
        """Record transition and perform Double DQN update when schedule permits."""
        self.buffer.push(state, action, reward, next_state, terminated)
        self.env_steps += 1

        if len(self.buffer) < max(self.learning_starts, self.batch_size):
            return None
        if self.env_steps % self.train_freq != 0:
            return None

        return self._learn_batch()

    def _learn_batch(self) -> float:
        """Perform one Double DQN gradient step and periodically sync target network."""
        states, actions, rewards, next_states, terminateds = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        terminateds = terminateds.to(self.device)

        self.online_net.train()
        q_current = self.online_net(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN: online network selects best next action, target network evaluates it
            next_best_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states).gather(1, next_best_actions)
            td_target = rewards + self.gamma * next_q_target * (1.0 - terminateds)

        loss = self.criterion(q_current, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.grad_steps += 1
        # Soft Polyak target update (tau=0.01) for smooth, responsive Q-targets
        tau = 0.01
        for tp, op in zip(self.target_net.parameters(), self.online_net.parameters()):
            tp.data.copy_(tau * op.data + (1.0 - tau) * tp.data)

        return float(loss.item())

    def save(self, path: str) -> None:
        """Save online network weights to disk."""
        torch.save(self.online_net.state_dict(), path)

    def load(self, path: str) -> None:
        """Load weights into both online and target networks."""
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.online_net.eval()
        self.target_net.eval()


class RandomAgent:
    """Baseline agent selecting actions uniformly at random."""

    def __init__(self, action_dim: int = 5, seed: int = 42) -> None:
        self.action_dim = action_dim
        self.rng = random.Random(seed)

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        return self.rng.randrange(self.action_dim)


class BFSHeuristicAgent:
    """Baseline agent that follows shortest walkable BFS path in O(1) ignoring guards."""

    def __init__(self, env: StealthHeistEnv) -> None:
        self.env = env

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        dist_map = self.env.dist_to_exit if self.env.has_loot else self.env.dist_to_loot
        ar, ac = self.env.agent_pos
        best_action = ACTION_STAY
        best_dist = dist_map[(ar, ac)]

        for action in range(4):
            dr, dc = ACTION_DELTAS[action]
            nr, nc = ar + dr, ac + dc
            if (nr, nc) in dist_map and dist_map[(nr, nc)] < best_dist:
                best_dist = dist_map[(nr, nc)]
                best_action = action

        return best_action
