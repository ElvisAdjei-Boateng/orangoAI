from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from app.agents.dqn import DQNAdvisor
from app.config import ExperimentConfig
from app.env.orango_env import OrangoEnv


def run_experiment(config: ExperimentConfig) -> list[dict]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    env = OrangoEnv(user_type=config.user_type)
    agent = DQNAdvisor(
        hidden_dim=config.hidden_dim,
        learning_rate=config.learning_rate,
        discount=config.discount,
        epsilon_decay=config.epsilon_decay,
        seed=config.seed,
    )
    history = agent.train(env, episodes=config.episodes)
    (output_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    (output_dir / "episode_stats.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with (output_dir / "episode_stats.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    rewards = np.array([row["reward"] for row in history], dtype=np.float32)
    heatmap = np.zeros((7, agent.action_size), dtype=np.float32)
    for feature in range(7):
        for action in range(agent.action_size):
            heatmap[feature, action] = float(np.mean(agent.online.weights[1][:, action]))
    np.save(output_dir / "policy_heatmap.npy", heatmap)
    (output_dir / "summary.json").write_text(json.dumps({
        "episodes": config.episodes,
        "mean_reward": float(np.mean(rewards)),
        "last_50_mean_reward": float(np.mean(rewards[-50:])),
        "best_reward": float(np.max(rewards)),
    }, indent=2), encoding="utf-8")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Orango DQN advisor.")
    parser.add_argument("--config", default="config/experiment.yaml")
    args = parser.parse_args()
    history = run_experiment(ExperimentConfig.from_file(args.config))
    print(f"Completed {len(history)} episodes; final reward={history[-1]['reward']:.3f}")


if __name__ == "__main__":
    main()
