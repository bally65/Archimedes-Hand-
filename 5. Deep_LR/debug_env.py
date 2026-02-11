import numpy as np
from whole_body_env import WholeBodyEnv

env = WholeBodyEnv()
obs, _ = env.reset()
print(f"Initial Distance: {np.linalg.norm(obs[:3])}")
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
print(f"Reward: {reward}, Done: {done}")
