from app.env.orango_env import OrangoEnv


env = OrangoEnv()

state, info = env.reset()

print("Initial state:")
print(state)

print("\nAction space:")
print(env.action_space)

print("\nObservation space:")
print(env.observation_space)

for step in range(10):

    action = env.action_space.sample()

    state, reward, terminated, truncated, info = env.step(action)

    print(
        f"Step {step + 1}: "
        f"Action={action}, "
        f"Reward={reward:.3f}"
    )

