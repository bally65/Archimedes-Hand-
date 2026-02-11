import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from robot_arm_env import RobotArmEnv
import os

def test():
    # 創建環境
    env_func = lambda: RobotArmEnv()
    env = DummyVecEnv([env_func])
    
    # 加載歸一化參數
    stats_path = "models/vec_normalize.pkl"
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    
    # 加載模型
    model_path = "models/td3_robot_arm_final.zip"
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found.")
        return
        
    model = TD3.load(model_path, env=env)
    
    print("🚀 啟動模型測試（連續 5 個任務）...")
    
    for i in range(5):
        obs = env.reset()
        done = False
        steps = 0
        
        # 獲取目標位置
        target = env.envs[0].unwrapped.target_pos
        print(f"\n任務 {i+1}: 目標座標 -> {target}")
        
        while steps < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps += 1
            
            # 獲取當前末端位置
            ee_site_id = 0 # site 0 is ee_site in this model
            current_pos = env.envs[0].unwrapped.data.site_xpos[ee_site_id]
            dist = np.linalg.norm(current_pos - target)
            
            if steps % 20 == 0:
                print(f"   步數: {steps} | 距離目標: {dist:.4f}m")
            
            if dist < 0.01: # 1cm 判定成功
                print(f"✅ 成功命中目標！耗時 {steps} 步。")
                break
        
        if steps >= 200:
            print("❌ 任務未在限時內完成。")

if __name__ == "__main__":
    test()
