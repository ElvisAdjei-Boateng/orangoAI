from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class UserProfile:
    """Hidden characteristics of a simulated user."""

    receptiveness: float
    follow_through: float
    learning_rate: float
    stress_sensitivity: float


class UserSimulator:
    """Simulate how a user reacts to different advisor actions."""

    def __init__(self, profile: UserProfile | None = None, rng: np.random.Generator | None = None):
        self.profile = profile or UserProfile(
            receptiveness=0.6,
            follow_through=0.6,
            learning_rate=0.6,
            stress_sensitivity=0.5,
        )
        self.rng = rng or np.random.default_rng()

    def step(self, state: np.ndarray, action: int) -> np.ndarray:
        """Apply one advisor action to the user state and return the next state."""
        if action not in range(6):
            raise ValueError(f"Unsupported action {action}; expected 0..5")

        next_state = state.copy().astype(np.float32)
        noise = self.rng.normal(0.0, 0.015, size=7)

        severity, clarity, goal_clarity, stress, motivation, engagement, progress = next_state

        if action == 0:  # Ask
            clarity += 0.08 * self.profile.learning_rate * self.profile.receptiveness
            engagement += 0.03 * self.profile.receptiveness
            motivation += 0.01
        elif action == 1:  # Inform
            clarity += 0.06 * self.profile.learning_rate * self.profile.receptiveness
            stress -= 0.02 * self.profile.learning_rate
            goal_clarity += 0.02
        elif action == 2:  # Plan
            progress += 0.10 * self.profile.follow_through * self.profile.receptiveness
            goal_clarity += 0.06 * self.profile.receptiveness
            engagement += 0.03 * self.profile.receptiveness
            motivation += 0.02
        elif action == 3:  # Small action
            progress += 0.08 * self.profile.follow_through
            engagement += 0.05 * self.profile.receptiveness
            stress -= 0.03 * self.profile.follow_through
            motivation += 0.03
        elif action == 4:  # Reflect
            clarity += 0.05 * self.profile.learning_rate
            stress -= 0.05 * self.profile.stress_sensitivity
            goal_clarity += 0.04 * self.profile.receptiveness
            motivation += 0.02
        elif action == 5:  # Summarize
            clarity += 0.04 * self.profile.learning_rate
            engagement += 0.02 * self.profile.receptiveness
            motivation += 0.03

        next_state = np.array(
            [
                severity,
                clarity,
                goal_clarity,
                stress,
                motivation,
                engagement,
                progress,
            ],
            dtype=np.float32,
        )
        next_state += noise
        next_state = np.clip(next_state, 0.0, 1.0)
        return next_state


def create_user_profile(user_type: str, rng: np.random.Generator | None = None) -> UserProfile:
    rng = rng or np.random.default_rng()

    if user_type == "motivated":
        return UserProfile(
            receptiveness=0.85,
            follow_through=0.85,
            learning_rate=0.80,
            stress_sensitivity=0.60,
        )
    if user_type == "neutral":
        return UserProfile(
            receptiveness=0.55,
            follow_through=0.50,
            learning_rate=0.50,
            stress_sensitivity=0.50,
        )
    if user_type == "resistant":
        return UserProfile(
            receptiveness=0.30,
            follow_through=0.25,
            learning_rate=0.30,
            stress_sensitivity=0.30,
        )

    raise ValueError(f"Unknown user type: {user_type}")