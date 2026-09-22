from __future__ import annotations

from typing import Callable


ACTION_NAMES = ("Ask", "Inform", "Plan", "Small action", "Reflect", "Summarize")


def template_response(action: int, state: list[float] | tuple[float, ...]) -> str:
    """Convert an RL action into a safe, deterministic conversational reply."""
    if action not in range(len(ACTION_NAMES)):
        raise ValueError(f"Unsupported advisor action: {action}")
    clarity, goal_clarity, stress, progress = state[1], state[2], state[3], state[6]
    responses = {
        0: "What feels like the most important part of this situation to clarify first?",
        1: "Here is a concise way to think about this: separate what you can control today from what needs more information.",
        2: "Let’s turn this into a small plan with one clear next step and a way to check progress.",
        3: "What is one small action you could realistically take in the next few minutes?",
        4: "Before deciding, pause and notice what outcome matters most and what concern is creating the most pressure.",
        5: "So far, we have clarified the situation and identified a next step. Does that summary fit your understanding?",
    }
    suffix = ""
    if stress > 0.8:
        suffix = " We can slow down and make the next step smaller."
    elif progress > 0.7 and clarity > 0.7 and goal_clarity > 0.7:
        suffix = " You appear to have a clear path forward."
    return responses[action] + suffix


class ResponseGenerator:
    """Template-first interface that can later be backed by an external LLM."""

    def __init__(self, provider: Callable[[int, list[float]], str] | None = None):
        self.provider = provider

    def generate(self, action: int, state: list[float] | tuple[float, ...]) -> str:
        if self.provider is not None:
            return self.provider(action, state)
        return template_response(action, state)
