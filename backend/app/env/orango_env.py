from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.simulator.user_simulator import UserSimulator, create_user_profile


class OrangoEnv(gym.Env):
    """Environment for a conversational AI advisor that helps a user progress."""

    metadata = {"render_modes": []}

    def __init__(self, user_type: str = "neutral"):
        super().__init__()
        self.user_type = user_type
        self.user_profile = create_user_profile(user_type)
        self.simulator = UserSimulator(profile=self.user_profile)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

        # 0 Ask, 1 Inform, 2 Plan, 3 Small action, 4 Reflect, 5 Summarize
        self.action_space = spaces.Discrete(6)
        self.max_steps = 10
        self.current_step = 0
        self.state = np.zeros(7, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        base = self.np_random.uniform(low=0.15, high=0.85, size=7).astype(np.float32)
        base[3] = np.clip(base[3] + 0.25, 0.0, 1.0)
        base[6] = 0.1
        self.state = base
        return self.state.copy(), {"user_type": self.user_type}

    def step(self, action):
        if not self.action_space.contains(int(action)):
            raise ValueError(f"Unsupported action {action}")

        old_state = self.state.copy()
        self.current_step += 1
        self.state = self.simulator.step(self.state, int(action))

        reward = self.calculate_reward(old_state, self.state, int(action))
        terminated = self.state[6] >= 0.9
        truncated = self.current_step >= self.max_steps
        info = {
            "step": self.current_step,
            "action": int(action),
            "user_type": self.user_type,
        }

        return self.state.copy(), float(reward), terminated, truncated, info

    def calculate_reward(self, old_state, new_state, action):
        goal_progress_change = new_state[6] - old_state[6]
        clarity_change = new_state[1] - old_state[1]
        stress_change = old_state[3] - new_state[3]
        engagement_change = new_state[5] - old_state[5]

        reward = (
            1.6 * goal_progress_change
            + 1.0 * clarity_change
            + 0.8 * engagement_change
            + 1.2 * stress_change
        )

        if action in (0, 1, 4, 5):
            reward += 0.05
        if action in (2, 3):
            reward += 0.08
        return float(np.clip(reward, -2.0, 3.0))

    def render(self):
        return self.state.copy()
