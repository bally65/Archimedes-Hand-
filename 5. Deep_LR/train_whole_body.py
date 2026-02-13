import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import os
from whole_body_env import WholeBodyEnv

def train_whole_body_v3():
    print("🚀 [Version 3.0] 啟動具備地形預測能力的全系統協同訓練...")
    
    # 創建環境
    env = make_vec_env(lambda: WholeBodyEnv(), n_envs=1)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    n_actions = env.action_space.shape[-1]
    # 使用更有層次的動作噪聲，促進探索
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.15 * np.ones(n_actions))
    
    # 提升模型規模以處理高度圖數據
    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        device="auto",
        learning_rate=1e-4, # Lower LR for better convergence with complex input
        batch_size=128,
        buffer_size=1000000,
        learning_starts=2000,
        policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256]))
    )
    
    os.makedirs("./models/whole_body_v3", exist_ok=True)
    
    print("⏳ 正在進行 1,500,000 步的深度強化學習訓練...")
    model.learn(total_timesteps=1500000, log_interval=100)
    
    model.save("./models/whole_body_v3/td3_terrain_aware_final")
    env.save("./models/whole_body_v3/vec_normalize_v3.pkl")
    print("✅ 地形預測模型訓練完成並存檔！")

if __name__ == "__main__":
    train_whole_body_v3()
