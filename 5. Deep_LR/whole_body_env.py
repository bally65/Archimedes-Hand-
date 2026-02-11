import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

class WholeBodyEnv(gym.Env):
    """
    全系統協同環境：同時控制螺桿移動與手臂抓取
    支援多種地形探索 (Domain Randomization)
    """
    def __init__(self):
        super().__init__()
        import os
        xml_path = os.path.join(os.path.dirname(__file__), "archimedes_hand_mujoco.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 動作空間：6個手臂關節 + 4個螺桿
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        
        # 狀態空間：手臂狀態 + 底座位置 + 目標距離 + 地形編碼 (4) = 50維
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(50,), dtype=np.float32)
        
        self.target_pos = np.array([1.0, 0.0, 0.2]) 
        self.terrain_idx = 0

    def _get_obs(self):
        ee_pos = self.data.site_xpos[0] # ee_site
        rel_pos = self.target_pos - ee_pos
        
        # 地形 One-hot 編碼
        terrain_one_hot = np.zeros(4)
        terrain_one_hot[self.terrain_idx] = 1.0
        
        obs = np.concatenate([
            rel_pos, 
            self.data.qpos[:self.model.nq], 
            self.data.qvel[:self.model.nv],
            self.data.ctrl,
            terrain_one_hot
        ], axis=0).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # --- Domain Randomization: 多種環境探索 ---
        terrains = ["solid", "sand", "mud", "water"]
        self.terrain_idx = self.np_random.integers(0, len(terrains))
        terrain_type = terrains[self.terrain_idx]
        
        # 重置物理參數
        self.model.opt.density = 0.0
        self.model.opt.viscosity = 0.0
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        
        if terrain_type == "solid":
            self.model.geom_friction[floor_id] = [1.0, 0.005, 0.0001]
        elif terrain_type == "sand":
            self.model.geom_friction[floor_id] = [1.5, 0.5, 0.01]
            self.model.geom_solref[floor_id] = [0.02, 1.0] # 較軟的地面
        elif terrain_type == "mud":
            self.model.geom_friction[floor_id] = [2.0, 1.0, 0.05]
            self.model.opt.viscosity = 0.1 # 模擬粘滯阻力
        elif terrain_type == "water":
            self.model.geom_friction[floor_id] = [0.3, 0.005, 0.0001] # 濕滑
            self.model.opt.density = 1000.0 # 水的密度
            self.model.opt.viscosity = 0.01
            
        # 隨機化目標位置 (距離底座 0.5m ~ 2.0m)
        self.target_pos = np.array([
            self.np_random.uniform(0.5, 2.0),
            self.np_random.uniform(-0.5, 0.5),
            self.np_random.uniform(0.1, 0.4)
        ])
        
        return self._get_obs(), {}

    def step(self, action):
        # 映射動作值到實際物理量
        self.data.ctrl[:6] = action[:6] * 15.0 # 手臂扭矩
        self.data.ctrl[6:] = action[6:] * 10.0 # 螺桿推力
        
        mujoco.mj_step(self.model, self.data)
        
        ee_pos = self.data.site_xpos[0]
        dist = np.linalg.norm(ee_pos - self.target_pos)
        
        # 複合獎勵函數
        reward = -dist * 2.0 # 距離懲罰
        
        # 加上生存/完成獎勵
        if dist < 0.05: reward += 5.0
        if dist < 0.01: reward += 50.0 
        
        # 能量消耗懲罰 (鼓勵高效動作)
        reward -= 0.01 * np.sum(np.square(action))
        
        # 判斷結束：成功命中或超時 (15秒)
        done = dist < 0.01 or self.data.time > 15.0
        
        return self._get_obs(), reward, done, False, {}

if __name__ == "__main__":
    env = WholeBodyEnv()
    obs, _ = env.reset()
    print(f"🛠️ 全系統協同環境 (Whole-Body) 已就緒。")
    print(f"   觀測維度: {len(obs)} | 當前地形索引: {env.terrain_idx}")
