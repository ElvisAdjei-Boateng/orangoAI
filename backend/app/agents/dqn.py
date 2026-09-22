from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.items: list[Transition] = []

    def add(self, transition: Transition) -> None:
        if len(self.items) >= self.capacity:
            self.items.pop(0)
        self.items.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        indices = self.rng.choice(len(self.items), size=batch_size, replace=False)
        return [self.items[int(index)] for index in indices]

    def __len__(self) -> int:
        return len(self.items)


class MLP:
    """Small NumPy MLP with tanh hidden layers and linear Q outputs."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, rng: np.random.Generator):
        self.weights = [
            rng.normal(0.0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim)).astype(np.float32),
            rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, output_dim)).astype(np.float32),
        ]
        self.biases = [np.zeros(hidden_dim, dtype=np.float32), np.zeros(output_dim, dtype=np.float32)]

    def predict(self, states: np.ndarray) -> np.ndarray:
        hidden = np.tanh(states @ self.weights[0] + self.biases[0])
        return hidden @ self.weights[1] + self.biases[1]

    def train_batch(self, states: np.ndarray, actions: np.ndarray, targets: np.ndarray, learning_rate: float) -> float:
        hidden = np.tanh(states @ self.weights[0] + self.biases[0])
        predictions = hidden @ self.weights[1] + self.biases[1]
        errors = predictions[np.arange(len(states)), actions] - targets
        loss = float(np.mean(errors**2))

        output_gradient = np.zeros_like(predictions)
        output_gradient[np.arange(len(states)), actions] = (2.0 / len(states)) * errors
        weight_1_gradient = hidden.T @ output_gradient
        bias_1_gradient = output_gradient.sum(axis=0)
        hidden_gradient = (output_gradient @ self.weights[1].T) * (1.0 - hidden**2)
        weight_0_gradient = states.T @ hidden_gradient
        bias_0_gradient = hidden_gradient.sum(axis=0)

        self.weights[1] -= learning_rate * weight_1_gradient
        self.biases[1] -= learning_rate * bias_1_gradient
        self.weights[0] -= learning_rate * weight_0_gradient
        self.biases[0] -= learning_rate * bias_0_gradient
        return loss

    def copy(self) -> "MLP":
        clone = object.__new__(MLP)
        clone.weights = [weight.copy() for weight in self.weights]
        clone.biases = [bias.copy() for bias in self.biases]
        return clone

    def soft_update(self, source: "MLP", tau: float) -> None:
        for target, source_weight in zip(self.weights, source.weights):
            target *= 1.0 - tau
            target += tau * source_weight
        for target, source_bias in zip(self.biases, source.biases):
            target *= 1.0 - tau
            target += tau * source_bias


class DQNAdvisor:
    """A dependency-light DQN advisor using a NumPy neural network."""

    def __init__(
        self,
        state_dim: int = 7,
        action_size: int = 6,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        discount: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 10_000,
        batch_size: int = 32,
        target_tau: float = 0.02,
        seed: int | None = None,
    ):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_tau = target_tau
        self.rng = np.random.default_rng(seed)
        self.online = MLP(state_dim, hidden_dim, action_size, self.rng)
        self.target = self.online.copy()
        self.replay = ReplayBuffer(replay_capacity, self.rng)

    def choose_action(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))
        return int(np.argmax(self.online.predict(np.asarray(state, dtype=np.float32)[None, :])[0]))

    def update(self, state, action, reward, next_state, done) -> float | None:
        self.replay.add(Transition(np.asarray(state, dtype=np.float32), int(action), float(reward), np.asarray(next_state, dtype=np.float32), bool(done)))
        if len(self.replay) < self.batch_size:
            return None
        batch = self.replay.sample(self.batch_size)
        states = np.stack([item.state for item in batch])
        actions = np.array([item.action for item in batch], dtype=np.int64)
        rewards = np.array([item.reward for item in batch], dtype=np.float32)
        next_states = np.stack([item.next_state for item in batch])
        dones = np.array([item.done for item in batch], dtype=np.float32)
        next_values = np.max(self.target.predict(next_states), axis=1)
        targets = rewards + (1.0 - dones) * self.discount * next_values
        loss = self.online.train_batch(states, actions, targets, self.learning_rate)
        self.target.soft_update(self.online, self.target_tau)
        return loss

    def train(self, env, episodes: int = 500) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0.0
            losses: list[float] = []
            for step in range(env.max_steps):
                action = self.choose_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                loss = self.update(state, action, reward, next_state, terminated or truncated)
                if loss is not None:
                    losses.append(loss)
                state = next_state
                total_reward += reward
                if terminated or truncated:
                    break
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            history.append({
                "episode": episode + 1,
                "reward": float(total_reward),
                "steps": step + 1,
                "loss": float(np.mean(losses)) if losses else None,
                "epsilon": float(self.epsilon),
            })
        return history
