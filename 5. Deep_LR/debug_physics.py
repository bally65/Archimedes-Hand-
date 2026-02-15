import gymnasium as gym
import numpy as np
from whole_body_env import WholeBodyEnv

def debug_actions():
    env = WholeBodyEnv()
    obs, _ = env.reset()
    print(f"Initial Obs (first 3): {obs[:3]}")
    
    # Try a few steps with random actions
    for i in range(10):
        action = np.ones(10) * 0.5 # Constant forward/move action
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: Pos={env.data.qpos[0]:.4f}, Vel={env.data.qvel[0]:.4f}, Reward={reward:.4f}")

if __name__ == "__main__":
    debug_actions()
