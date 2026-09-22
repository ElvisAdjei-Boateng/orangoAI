from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.q_learning import QLearningAdvisor
from app.env.orango_env import OrangoEnv


def run_demo(episodes: int = 120):
    env = OrangoEnv(user_type="neutral")
    advisor = QLearningAdvisor()
    scores = advisor.train(env, episodes=episodes)

    print(f"Training complete: average reward = {sum(scores) / len(scores):.3f}")
    state, _ = env.reset()
    for step in range(env.max_steps):
        action = advisor.best_action(state)
        state, reward, terminated, truncated, info = env.step(action)
        print(f"Step {step + 1}: action={action}, reward={reward:.3f}, progress={state[6]:.3f}, clarity={state[1]:.3f}")
        if terminated or truncated:
            break


if __name__ == "__main__":
    run_demo()
