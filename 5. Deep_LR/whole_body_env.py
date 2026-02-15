import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

class WholeBodyEnv(gym.Env):
    """
    全系統協同環境 3.5：固定條件循序漸進學習 (Step-by-Step Learning)
    第一階段：固定 SOLID 地面 + 固定 20 公尺目標
    """
    def __init__(self):
        super().__init__()
        import os
        xml_path = os.path.join(os.path.dirname(__file__), "archimedes_hand_mujoco.xml")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(10,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(75,), dtype=np.float32)
        self.target_pos = np.array([20.0, 0.0, 0.2]) 
        self.terrain_idx = 0

    def _get_elevation_map(self):
        return self.np_random.uniform(-0.005, 0.005, 25)

    def _get_obs(self):
        ee_pos = self.data.site_xpos[0]
        rel_pos = self.target_pos - ee_pos
        terrain_one_hot = np.zeros(4)
        terrain_one_hot[self.terrain_idx] = 1.0
        elev_map = self._get_elevation_map()
        return np.concatenate([
            rel_pos, 
            self.data.qpos[:self.model.nq], 
            self.data.qvel[:self.model.nv],
            self.data.ctrl,
            terrain_one_hot,
            elev_map
        ], axis=0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # --- EXTREME CHALLENGE: Super Sticky Mud ---
        self.terrain_idx = 1 # Marking as Mud
        self.model.opt.density = 50.0   # High fluid density
        self.model.opt.viscosity = 0.8  # EXTREME Viscosity (Sticky!)
        
        floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        # Increase friction to simulate sticking to the ground
        self.model.geom_friction[floor_id] = [5.0, 0.1, 0.05] 
        
        self.target_pos = np.array([20.0, 0.0, 0.2])
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl[:6] = np.clip(action[:6] * 12.0, -12.0, 12.0)
        self.data.ctrl[6:] = np.clip(action[6:] * 15.0, -15.0, 15.0) # Boosted power
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        ee_pos = self.data.site_xpos[0]
        dist = np.linalg.norm(ee_pos - self.target_pos)
        
        # --- Speed & Direction Reward ---
        v_vec = self.data.qvel[:3]
        target_dir = (self.target_pos - ee_pos) / (dist + 1e-6)
        v_target = np.dot(v_vec, target_dir)
        
        # 1. Target Progress Reward (Strong incentive for moving forward)
        reward = v_target * 50.0 
        
        # 2. Heading Penalty (Stay in a straight line, penalize Y-axis deviation)
        reward -= abs(ee_pos[1]) * 10.0 
        
        # 3. Spin Penalty (Stop rotating/spinning)
        # qvel[3:6] are angular velocities
        reward -= abs(self.data.qvel[5]) * 5.0 
        
        # 4. Energy Penalty
        reward -= 0.05 * np.sum(np.square(action))
        
        if dist < 0.1: reward += 500.0   
        if dist < 0.02: reward += 5000.0 
        is_unstable = not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
        done = dist < 0.02 or self.data.time > 60.0 or is_unstable
        
        return self._get_obs(), reward, done, False, {}

if __name__ == "__main__":
    env = WholeBodyEnv()
    print("🛠️ 已優化速度獎勵：第一階段 (20m) 訓練環境就緒。")
