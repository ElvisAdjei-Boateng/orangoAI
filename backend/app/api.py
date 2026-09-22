from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.agents.dqn import DQNAdvisor
from app.env.orango_env import OrangoEnv
from app.llm.responses import ACTION_NAMES, ResponseGenerator


class AdviceRequest(BaseModel):
    state: list[float] | None = Field(default=None, min_length=7, max_length=7)
    user_type: str = "neutral"


class AdviceResponse(BaseModel):
    action: int
    action_name: str
    response: str
    state: list[float]


app = FastAPI(title="Orango AI Logical Advisor", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)
agent = DQNAdvisor(seed=7)
responses = ResponseGenerator()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/advice", response_model=AdviceResponse)
def advice(request: AdviceRequest) -> AdviceResponse:
    if request.user_type not in {"motivated", "neutral", "resistant"}:
        raise HTTPException(status_code=400, detail="user_type must be motivated, neutral, or resistant")
    env = OrangoEnv(user_type=request.user_type)
    state = np.asarray(request.state, dtype=np.float32) if request.state is not None else env.reset()[0]
    if not env.observation_space.contains(state):
        raise HTTPException(status_code=400, detail="state values must be between 0 and 1")
    action = agent.choose_action(state, explore=False)
    return AdviceResponse(
        action=action,
        action_name=ACTION_NAMES[action],
        response=responses.generate(action, state.tolist()),
        state=state.tolist(),
    )
