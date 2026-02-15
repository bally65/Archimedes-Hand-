import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from whole_body_env import WholeBodyEnv
import os

def test_phase1():
    print("🎯 正在測試第一階段（20公尺固定地面）的訓練成果...")
    env_func = lambda: WholeBodyEnv()
    env = DummyVecEnv([env_func])
    
    # Load v3-based whole body normalization
    stats_path = "models/whole_body_v3/vec_normalize_v3.pkl"
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    
    model_path = "models/whole_body_v3/td3_terrain_aware_final.zip"
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型: {model_path}")
        return
        
    model = TD3.load(model_path, env=env)
    
    successes = 0
    total_tests = 5
    
    for i in range(total_tests):
        obs = env.reset()
        # Ensure we are testing the fixed condition
        env.envs[0].unwrapped.terrain_idx = 0
        env.envs[0].unwrapped.target_pos = np.array([20.0, 0.0, 0.2])
        
        steps = 0
        while steps < 2000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps += 1
            
            current_pos = env.envs[0].unwrapped.data.qpos[:3]
            dist = np.linalg.norm(current_pos - env.envs[0].unwrapped.target_pos)
            
            if dist < 0.2: # 20cm threshold
                print(f"   ✅ 測試 {i+1} 成功！耗時 {steps} 步")
                successes += 1
                break
        
        if steps >= 2000:
            print(f"   ❌ 測試 {i+1} 超時，最終距離目標 {dist:.2f}m")

    print(f"\n📊 第一階段總結：成功率 {(successes/total_tests)*100:.1f}%")

if __name__ == "__main__":
    test_phase1()
