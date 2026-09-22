import numpy as np

from app.env.orango_env import OrangoEnv
from app.simulator.user_simulator import UserSimulator, create_user_profile
from app.agents.dqn import DQNAdvisor
from app.llm.responses import template_response


def test_environment_reset_and_action_space():
    env = OrangoEnv(user_type="neutral")
    state, info = env.reset(seed=3)

    assert state.shape == (7,)
    assert env.action_space.n == 6
    assert np.all((0.0 <= state) & (state <= 1.0))
    assert info["user_type"] == "neutral"


def test_step_updates_state_and_reward():
    env = OrangoEnv(user_type="motivated")
    env.reset(seed=4)

    state, reward, terminated, truncated, info = env.step(2)

    assert state.shape == (7,)
    assert np.all((0.0 <= state) & (state <= 1.0))
    assert np.isfinite(reward)
    assert info["action"] == 2
    assert not terminated or state[6] >= 0.9


def test_user_simulator_profile_matches_expected_behaviour():
    profile = create_user_profile("motivated")
    simulator = UserSimulator(profile=profile, rng=np.random.default_rng(5))
    state = np.array([0.4, 0.5, 0.3, 0.6, 0.7, 0.5, 0.2], dtype=np.float32)

    next_state = simulator.step(state, 2)

    assert next_state.shape == state.shape
    assert np.all((0.0 <= next_state) & (next_state <= 1.0))
    assert next_state[6] >= state[6]


def test_q_learning_agent_can_train():
    env = OrangoEnv(user_type="neutral")
    from app.agents.q_learning import QLearningAdvisor

    advisor = QLearningAdvisor(learning_rate=0.3, epsilon=1.0)
    scores = advisor.train(env, episodes=30)

    assert len(scores) == 30
    assert np.mean(scores) > -1.0


def test_dqn_can_select_actions_and_train():
    env = OrangoEnv(user_type="neutral")
    advisor = DQNAdvisor(batch_size=4, hidden_dim=16, seed=3)
    history = advisor.train(env, episodes=8)

    assert len(history) == 8
    assert all(0 <= advisor.choose_action(env.reset(seed=9)[0], explore=False) < 6 for _ in range(3))
    assert history[-1]["epsilon"] < 1.0


def test_template_response_maps_action_to_text():
    response = template_response(2, [0.4, 0.5, 0.4, 0.6, 0.5, 0.5, 0.1])

    assert "plan" in response.lower()
