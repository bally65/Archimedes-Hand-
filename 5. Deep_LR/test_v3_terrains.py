import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from whole_body_env import WholeBodyEnv
import os

def test_v3():
    env_func = lambda: WholeBodyEnv()
    env = DummyVecEnv([env_func])
    
    stats_path = "models/whole_body_v3/vec_normalize_v3.pkl"
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    
    model_path = "models/whole_body_v3/td3_terrain_aware_final.zip"
    if not os.path.exists(model_path):
        print(f"❌ Model {model_path} not found.")
        return
        
    model = TD3.load(model_path, env=env)
    
    terrains = ["SOLID", "SAND", "MUD", "WATER"]
    print("\n🤖 Archimedes v3.0 Multi-Terrain Mission Report:")
    
    for idx, terrain_name in enumerate(terrains):
        obs = env.reset()
        # Overwrite terrain idx after reset
        env.envs[0].unwrapped.terrain_idx = idx
        
        target = env.envs[0].unwrapped.target_pos
        steps = 0
        success = False
        
        while steps < 300:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps += 1
            
            ee_site_id = 0
            current_pos = env.envs[0].unwrapped.data.site_xpos[ee_site_id]
            dist = np.linalg.norm(current_pos - target)
            
            if dist < 0.05: # Using 5cm as success for whole-body distance
                success = True
                break
        
        status = "✅ SUCCESS" if success else "❌ TIMEOUT"
        print(f"   - Terrain: {terrain_name:6} | Status: {status} | Steps: {steps:3} | Dist: {dist:.4f}m")

if __name__ == "__main__":
    test_v3()
