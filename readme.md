# Orango AI Logical Advisor

This project implements a minimal RL-based conversational advisor loop.

## Architecture
1. Simple frontend shell for future chat UI
2. FastAPI-ready backend structure with RL environment and simulator
3. User state engine and conversation context modeling
4. Q-learning advisor policy for action selection
5. LLM response generation layer to be added on top of the learned policy
6. User feedback and reward shaping for future RL tuning

## Current status
The environment, user simulator, and Q-learning policy are implemented and validated.

## Run
From the backend directory:

python -m app.main

Or for tests:

python -m pytest -q tests/test_env.py

## What the environment models
- User clarity
- Goal progress
- Stress level
- Motivation and engagement
- Advice strategies: Ask, Inform, Plan, Small action, Reflect, Summarize

This creates a learnable simulation where an advisor can improve by selecting actions that improve user progress while reducing stress.