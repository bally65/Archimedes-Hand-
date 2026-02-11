import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import os
from whole_body_env import WholeBodyEnv

def train_whole_body():
    print("🚀 啟動全系統協同 (Whole-Body) 複合任務訓練...")
    
    # 創建環境
    env = make_vec_env(lambda: WholeBodyEnv(), n_envs=1)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions))
    
    # 使用與之前相似的 TD3 結構
    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        device="auto",
        learning_rate=3e-4,
        batch_size=128, # Reduced from 256 for stability
        buffer_size=500000, # Reduced from 1M
        learning_starts=1000,
        policy_kwargs=dict(net_arch=dict(pi=[400, 300], qf=[400, 300]))
    )
    
    os.makedirs("./models/whole_body", exist_ok=True)
    
    # 開始訓練 (增加步數至 1,000,000 以適應多環境探索)
    model.learn(total_timesteps=1000000, log_interval=100)
    
    model.save("./models/whole_body/td3_whole_body_initial")
    env.save("./models/whole_body/vec_normalize_whole_body.pkl")
    print("✅ 全系統初步訓練完成！")

if __name__ == "__main__":
    train_whole_body()
