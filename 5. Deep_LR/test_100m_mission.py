import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from whole_body_env import WholeBodyEnv
import os
import time

def run_100m_mission():
    print("🏁 啟動 Archimedes' Hand 4.0：百米超長跑實測任務...")
    
    env_func = lambda: WholeBodyEnv()
    env = DummyVecEnv([env_func])
    
    # Load normalization stats
    stats_path = "models/whole_body_v3/vec_normalize_v3.pkl"
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    
    # Load the champion model (v4.0)
    model_path = "models/whole_body_v3/td3_terrain_aware_final.zip"
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案: {model_path}")
        return
    
    model = TD3.load(model_path, env=env)
    
    # Reset first to get initial setup
    obs = env.reset()
    
    # Force SOLID terrain for baseline move test
    env.envs[0].unwrapped.terrain_idx = 0 
    floor_id = env.envs[0].unwrapped.model.geom("floor").id
    env.envs[0].unwrapped.model.geom_friction[floor_id] = [1.0, 0.005, 0.0001]
    env.envs[0].unwrapped.model.opt.density = 0.0
    env.envs[0].unwrapped.model.opt.viscosity = 0.0
    
    # Force target to exactly 100m away
    env.envs[0].unwrapped.target_pos = np.array([100.0, 0.0, 0.2])
    
    print(f"📍 目標：硬地環境 (SOLID)，距離 100.0 公尺。")
    
    steps = 0
    while steps < 5000:
        action, _ = model.predict(obs, deterministic=True)
        # Manually verify action range
        # print(f"Action sample: {action[0][:4]}") 
        
        obs, reward, done, info = env.step(action)
        steps += 1
        
        current_pos = env.envs[0].unwrapped.data.qpos[:3]
        
        if steps % 500 == 0:
            print(f"   🚩 步數: {steps:5} | 位置 X: {current_pos[0]:.2f}m | 速度: {env.envs[0].unwrapped.data.qvel[0]:.2f}m/s")
        
        if np.linalg.norm(current_pos - env.envs[0].unwrapped.target_pos) < 0.2:
            print(f"\n✨ 任務達成！")
            break
            
    print(f"📊 測試總結：")
    print(f"   - 總步數: {steps}")
    print(f"   - 最終 X 座標: {current_pos[0]:.4f}m")

if __name__ == "__main__":
    run_100m_mission()
