from __future__ import annotations

import numpy as np


class QLearningAdvisor:
    """A minimal tabular Q-learning advisor for the Orango environment."""

    def __init__(self, state_dim: int = 7, action_size: int = 6, bins: int = 5, learning_rate: float = 0.25, discount: float = 0.95, epsilon: float = 1.0):
        self.state_dim = state_dim
        self.action_size = action_size
        self.bins = bins
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.rng = np.random.default_rng()
        self.q_table = np.zeros((bins,) * state_dim + (action_size,), dtype=np.float32)

    def _discretize_state(self, state):
        clipped = np.clip(np.asarray(state, dtype=np.float32), 0.0, 1.0)
        bin_edges = np.linspace(0.0, 1.0, self.bins + 1)
        indices = np.digitize(clipped, bin_edges[1:-1], right=False)
        return tuple(int(i) for i in indices)

    def choose_action(self, state, explore: bool = True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_size))

        state_key = self._discretize_state(state)
        q_values = self.q_table[state_key]
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, terminated):
        current_key = self._discretize_state(state)
        next_key = self._discretize_state(next_state)

        current_q = self.q_table[current_key + (action,)]
        future_q = 0.0 if terminated else np.max(self.q_table[next_key])
        target = reward + self.discount * future_q
        self.q_table[current_key + (action,)] = current_q + self.learning_rate * (target - current_q)

    def train(self, env, episodes: int = 250, epsilon_min: float = 0.05):
        scores = []
        for episode in range(episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            for _ in range(env.max_steps):
                action = self.choose_action(state, explore=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                self.update(state, action, reward, next_state, terminated)
                state = next_state
                episode_reward += reward
                if terminated or truncated:
                    break
            scores.append(episode_reward)
            self.epsilon = max(epsilon_min, self.epsilon * 0.995)
        return scores

    def best_action(self, state):
        return self.choose_action(state, explore=False)
