import gymnasium as gym
from stable_baselines3 import TD3
from whole_body_env import WholeBodyEnv

print("Starting foreground test...")
env = WholeBodyEnv()
model = TD3("MlpPolicy", env, verbose=1, learning_starts=10)
print("Calling learn...")
model.learn(total_timesteps=50)
print("Success!")
