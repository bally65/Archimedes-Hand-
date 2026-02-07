#!/usr/bin/env python3
"""
使用Stable-Baselines3的TD3算法训练机械臂进行位置跟踪
"""

import numpy as np
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import os
import time
from robot_arm_env import RobotArmEnv
import argparse
import torch.nn as nn
import signal
import sys


class SaveVecNormalizeCallback(BaseCallback):
    """
    在保存最佳模型时同时保存VecNormalize参数的回调函数
    """
    def __init__(self, eval_callback, verbose=0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.eval_callback = eval_callback
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # 检查是否有新的最佳模型
        if self.eval_callback.best_mean_reward > self.best_mean_reward:
            self.best_mean_reward = self.eval_callback.best_mean_reward
            
            # 保存VecNormalize参数到logs/best_model目录
            if self.verbose > 0:
                print("保存与最佳模型对应的VecNormalize参数")
            vec_normalize_path = "./logs/best_model"
            os.makedirs(vec_normalize_path, exist_ok=True)
            self.model.get_vec_normalize_env().save(os.path.join(vec_normalize_path, "vec_normalize.pkl"))
        
        return True


class ManualInterruptCallback(BaseCallback):
    """
    允许手动中断训练并保存模型的回调函数
    """
    def __init__(self, verbose=0):
        super(ManualInterruptCallback, self).__init__(verbose)
        self.interrupted = False
        # 设置信号处理器来捕获Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, sig, frame):
        print('\n接收到中断信号，正在保存模型...')
        self.interrupted = True
        # 保存当前模型
        self.save_model()
        print('模型已保存，退出程序')
        sys.exit(0)
        
    def save_model(self):
        """
        保存当前模型和环境归一化参数
        """
        if self.model is not None:
            # 创建保存目录
            os.makedirs("./models/interrupted", exist_ok=True)
            
            # 保存模型
            self.model.save("./models/interrupted/td3_robot_arm_interrupted")
            
            # 保存VecNormalize参数
            env = self.model.get_vec_normalize_env()
            if env is not None:
                env.save("./models/interrupted/vec_normalize.pkl")
                
            print("已保存中断时的模型和参数到 ./models/interrupted/")
        
    def _on_step(self) -> bool:
        # 如果收到中断信号，停止训练
        if self.interrupted:
            return False
        return True


def train_robot_arm():
    """
    训练机械臂进行位置跟踪
    """
    print("创建机械臂环境...")
    
    # 创建环境并使用VecNormalize进行归一化
    env = make_vec_env(lambda: RobotArmEnv(), n_envs=1)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # 设置动作噪声
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=2.5 * np.ones(n_actions))
    
    # 自定义TD3神经网络结构
    # 这里我们定义一个更复杂的网络结构：
    # Actor网络: [512, 512, 256]
    # Critic网络: [512, 512, 256] (每个Q网络)
    policy_kwargs = dict(
        net_arch=dict(
            pi=[512, 512, 256],  # Actor网络结构
            qf=[512, 512, 256]   # Critic网络结构 (每个Q网络)
        ),
        activation_fn=nn.ReLU  # 使用ReLU激活函数
    )
    
    # 创建TD3模型
    model = TD3(
        "MlpPolicy",
        env,
        action_noise=action_noise,
        verbose=1,
        device="auto",  # 自动选择设备(CUDA/CPU)
        learning_rate=3e-4,
        buffer_size=3000000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_delay=4,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        policy_kwargs=policy_kwargs  # 使用自定义网络结构
    )
    
    # 创建评估环境和回调函数
    eval_env = make_vec_env(lambda: RobotArmEnv(), n_envs=1)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    # 加载训练环境的归一化参数到评估环境中
    eval_env.obs_rms = env.obs_rms
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/",
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    # 创建保存VecNormalize参数的回调函数
    save_vec_normalize_callback = SaveVecNormalizeCallback(eval_callback, verbose=1)
    
    # 创建手动中断回调函数
    manual_interrupt_callback = ManualInterruptCallback(verbose=1)
    
    # 创建日志目录
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    
    print("开始训练...")
    print("提示: 按 Ctrl+C 可以中途停止训练并保存最后一次模型数据")
    start_time = time.time()
    
    # 训练模型，同时使用三个回调函数
    model.learn(
        total_timesteps=5000000,
        callback=[eval_callback, save_vec_normalize_callback, manual_interrupt_callback],
        log_interval=1000
    )
    
    # 保存归一化环境和最终模型
    env.save("./models/vec_normalize.pkl")
    model.save("./models/td3_robot_arm_final")
    
    end_time = time.time()
    print(f"训练完成，耗时: {end_time - start_time:.2f}秒")
    
    return model, env


def test_robot_arm(model_path="./models/td3_robot_arm_final", 
                   normalize_path="./models/vec_normalize.pkl",
                   num_episodes=10):
    """
    测试训练好的模型
    """
    print("加载模型并测试...")
    
    # 创建环境
    env = make_vec_env(lambda: RobotArmEnv(render_mode="human"), n_envs=1)
    
    # 加载归一化环境
    if os.path.exists(normalize_path):
        env = VecNormalize.load(normalize_path, env)
        env.training = False
        env.norm_reward = False
    
    # 加载模型
    model = TD3.load(model_path, env=env)
    
    episode_rewards = []
    
    # 运行指定数量的回合
    for episode in range(num_episodes):
        obs = env.reset()
        total_reward = 0
        
        # 打印当前回合的目标位置
        # 正确获取多层包装环境中的目标位置
        target_pos = env.venv.envs[0].env.unwrapped.target_pos                    
        print(f"Episode {episode+1} target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        
        # 运行一个episode
        for i in range(5000):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0] if isinstance(reward, np.ndarray) else reward
            time.sleep(0.01)
            env.render()
            
            if done:
                print(f"Episode {episode+1} finished after {i+1} timesteps")
                print(f"Episode reward: {total_reward}")
                episode_rewards.append(total_reward)
                break    
    
    env.close()
    print(f"Average reward over {num_episodes} episodes: {np.mean(episode_rewards)}")
    return episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test robot arm with TD3")
    parser.add_argument("--test", action="store_true", help="Test the trained model")
    parser.add_argument("--model-path", type=str, default="./models/td3_robot_arm_final", 
                        help="Path to the model for testing")
    parser.add_argument("--normalize-path", type=str, default="./models/vec_normalize.pkl",
                        help="Path to the normalization parameters")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to test")
    
    args = parser.parse_args()
    
    if args.test:
        test_robot_arm(args.model_path, args.normalize_path, args.episodes)
    else:
        train_robot_arm()