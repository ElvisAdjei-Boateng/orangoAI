# Orango AI Logical Advisor

This project implements an RL-based conversational advisor loop with a NumPy DQN,
an HTTP API, and a small React client.

## Architecture
1. Simple frontend shell for future chat UI
2. FastAPI-ready backend structure with RL environment and simulator
3. User state engine and conversation context modeling
4. DQN advisor policy with replay memory and a target network
5. Template-first natural-language response layer
6. FastAPI service and React/Vite client
7. Experiment runner with YAML configuration, CSV/JSON metrics, and policy heatmap data

## Current status
The environment, user simulator, Q-learning baseline, DQN agent, response layer,
API scaffold, frontend scaffold, and experiment runner are implemented.

## Run
From the backend directory:

python -m app.main

# DQN experiment
python -m app.experiments --config ..\config\experiment.yaml

# API
uvicorn app.api:app --reload

Or for tests:

python -m pytest -q tests/test_env.py

From the frontend directory:

npm install
npm run dev

## What the environment models
- User clarity
- Goal progress
- Stress level
- Motivation and engagement
- Advice strategies: Ask, Inform, Plan, Small action, Reflect, Summarize

This creates a learnable simulation where an advisor can improve by selecting actions that improve user progress while reducing stress.

The experiment runner writes `episode_stats.csv`, `episode_stats.json`, `summary.json`,
and `policy_heatmap.npy` to the configured output directory.