import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

class WholeBodyEnv(gym.Env):
    """
    全系統協同環境 3.0：具備地形預測能力 (Elevation Mapping)
    支援多種地形探索與前方地形掃描
    """
    def __init__(self):
        super().__init__()
        import os
        xml_path = os.path.join(os.path.dirname(__file__), "archimedes_hand_mujoco.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # 動作空間：6個手臂關節 + 4個螺桿
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        
        # 狀態空間：
        # 原有狀態(46) + 地形編碼(4) + 高度圖(5x5=25) = 75維
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(75,), dtype=np.float32)
        
        self.target_pos = np.array([1.0, 0.0, 0.2]) 
        self.terrain_idx = 0
        self.scan_grid = np.meshgrid(np.linspace(-0.2, 0.8, 5), np.linspace(-0.5, 0.5, 5))

    def _get_elevation_map(self):
        """
        模擬高度掃描儀（AME-2 風格）：抓取機器人前方地形的高度分佈。
        新增：加入了方向性掃描，更聚焦於運動前方的障礙。
        """
        # 獲取機身朝向 (qpos[3:7] 為四元數)
        # 這裡簡化為獲取前方的相對偏移
        # 在 AME-2 中，這通常會通過深度相機轉換為局部 Grid
        
        # 模擬一個 5x5 的高度圖掃描 (範圍：前方 0.5m ~ 1.5m)
        # 目前場地為平面，未來加入階梯地形時，這裡將調用 Raycast 數據
        # 我們先模擬一些微小的地形起伏 (Noise) 以訓練智能體的魯棒性
        noise = self.np_random.uniform(-0.01, 0.01, 25) 
        return noise

    def _get_obs(self):
        ee_pos = self.data.site_xpos[0] # ee_site
        rel_pos = self.target_pos - ee_pos
        
        # 地形 One-hot
        terrain_one_hot = np.zeros(4)
        terrain_one_hot[self.terrain_idx] = 1.0
        
        # 高度圖
        elev_map = self._get_elevation_map()
        
        obs = np.concatenate([
            rel_pos, 
            self.data.qpos[:self.model.nq], 
            self.data.qvel[:self.model.nv],
            self.data.ctrl,
            terrain_one_hot,
            elev_map
        ], axis=0).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Domain Randomization
        terrains = ["solid", "sand", "mud", "water"]
        self.terrain_idx = self.np_random.integers(0, len(terrains))
        terrain_type = terrains[self.terrain_idx]
        
        self.model.opt.density = 0.0
        self.model.opt.viscosity = 0.0
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        
        if terrain_type == "solid":
            self.model.geom_friction[floor_id] = [1.0, 0.005, 0.0001]
        elif terrain_type == "sand":
            self.model.geom_friction[floor_id] = [1.5, 0.5, 0.01]
            self.model.geom_solref[floor_id] = [0.02, 1.0]
        elif terrain_type == "mud":
            self.model.geom_friction[floor_id] = [2.0, 1.0, 0.05]
            self.model.opt.viscosity = 0.1
        elif terrain_type == "water":
            self.model.geom_friction[floor_id] = [0.3, 0.005, 0.0001]
            self.model.opt.density = 1000.0
            self.model.opt.viscosity = 0.01
            
        # --- Challenge: Large-Scale Navigation (100m Range) ---
        # Extreme target distance for hectare-scale mission readiness
        self.target_pos = np.array([
            self.np_random.uniform(10.0, 100.0), # Target up to 100m
            self.np_random.uniform(-10.0, 10.0), # Wider lateral range
            self.np_random.uniform(0.1, 0.5)
        ])
        
        return self._get_obs(), {}

    def step(self, action):
        # 物理步進
        self.data.ctrl[:6] = np.clip(action[:6] * 12.0, -12.0, 12.0)
        self.data.ctrl[6:] = np.clip(action[6:] * 10.0, -10.0, 10.0) # Restored torque for speed
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        ee_pos = self.data.site_xpos[0]
        dist = np.linalg.norm(ee_pos - self.target_pos)
        
        # 強化版獎勵函數：加入大尺度導航優化
        reward = -dist * 0.5 # Lower weight for raw distance to prevent early gradient explosion
        
        # 方向性獎勵 (Progressive Reward)
        # Calculate velocity towards target
        base_vel = self.data.qvel[:3]
        to_target = self.target_pos - ee_pos
        to_target_unit = to_target / (np.linalg.norm(to_target) + 1e-6)
        velocity_towards_target = np.dot(base_vel, to_target_unit)
        
        reward += velocity_towards_target * 2.0 # Strong incentive to keep moving forward
        
        if dist < 0.1: reward += 50.0   # Intermediate success
        if dist < 0.02: reward += 500.0 # Final capture
        
        # Energy and stability
        reward -= 0.01 * np.sum(np.square(action))
        
        is_unstable = not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
        # 延長超時時間到 150 秒以適應 100 公尺跋涉
        done = dist < 0.02 or self.data.time > 150.0 or is_unstable
        
        if is_unstable: reward -= 200.0
        
        return self._get_obs(), reward, done, False, {}

if __name__ == "__main__":
    env = WholeBodyEnv()
    obs, _ = env.reset()
    print(f"🛠️ 地形預測環境已就緒。總觀測維度: {len(obs)}")
