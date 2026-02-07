import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer

class RobotArmEnv(gym.Env):
    """
    机械臂末端跟踪环境，使用MuJoCo作为物理引擎
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None):
        """
        初始化机械臂环境
        
        参数:
        - render_mode: 渲染模式
        """
        # 加载MuJoCo模型
        self.model = mujoco.MjModel.from_xml_path("robot_arm_mujoco.xml")
        self.data = mujoco.MjData(self.model)
        
        # 获取关节数量
        self.nu = self.model.nu
        self.nq = self.model.nq
        
        # 设置动作空间 (扭矩)
        self.action_space = spaces.Box(
            low=-15.0, high=15.0, shape=(self.nu,), dtype=np.float32
        )
        
        # 设置状态空间
        # 状态维度: 相对位置向量(3) + 关节角度(6) + 关节速度(6) + 上一时刻关节扭矩(6) + 末端速度(3) = 24维
        state_dim = 3 + 6 + 6 + 6 + 3  # 24维状态
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        
        # 设置目标位置 (末端执行器)
        self.target_pos = np.array([0.2, 0.2, 0.2])  # 初始目标位置
        self.previous_distance = None
        
        # 保存上一时刻的扭矩
        self.previous_torque = np.zeros(self.nu)
        
        # 保存上一时刻的关节速度
        self.previous_joint_velocities = np.zeros(self.nu)
        
        # 初始化可视化器为None
        self.render_mode = render_mode
        self.viewer = None
                
        # 重置环境
        self.reset()

    def _get_state(self):
        """
        获取当前状态
        """
        # 获取末端执行器位置（通过名称查找）
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        
        # 获取末端执行器位置
        ee_pos = self.data.site_xpos[ee_site_id].copy()
        
        # 计算相对位置向量 (目标位置 - 当前末端位置)
        relative_pos = self.target_pos - ee_pos
        
        # 获取关节角度
        joint_angles = self.data.qpos[:self.nu].copy()
        
        # 获取关节速度
        joint_velocities = self.data.qvel[:self.nu].copy()
        
        # 获取上一时刻的关节扭矩
        previous_torques = self.previous_torque.copy()
        
        # 使用body速度获取末端执行器速度
        ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        ee_vel = self.data.cvel[ee_body_id][:3].copy()
        
        # 拼接状态向量
        state = np.concatenate([
            relative_pos,          # 相对位置向量 (3,)
            joint_angles,          # 关节角度 (6,)
            joint_velocities,      # 关节角速度 (6,)
            previous_torques,      # 上一时刻关节扭矩 (6,)
            ee_vel                 # 末端速度 (3,)
        ])
        
        return state.astype(np.float32)

    def reset(self, seed=None, options=None):
        """
        重置环境
        """
        super().reset(seed=seed)
        
        # 重置MuJoCo数据
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置随机目标位置
        # x轴: [-0.3, 0.3]
        # y轴: [-0.47, -0.17] 
        # z轴: [0.2, 0.5]
        self.target_pos = np.array([
            self.np_random.uniform(-0.2, 0.2),    # x轴
            self.np_random.uniform(-0.37, -0.17),   # y轴
            self.np_random.uniform(0.2, 0.4)        # z轴
        ])
        
        self.success_threshold = 0.01  # 1mm阈值
        
        # 重置步数计数器
        self.step_count = 0
        self.previous_distance = None
        self.min_distance = None
        self.previous_torque = np.zeros(self.nu)
        self.previous_joint_velocities = np.zeros(self.nu)
        
        # 重置阶段性奖励标记
        self._phase_rewards_given = set()
        
        # 获取初始状态
        state = self._get_state()
        
        return state, {}

    def step(self, action):
        """
        执行动作并返回结果
        """
        # 保存当前扭矩作为上一时刻的扭矩
        self.previous_torque = self.data.ctrl[:self.nu].copy()
        
        # 保存当前关节速度作为上一时刻的关节速度
        self.previous_joint_velocities = self.data.qvel[:self.nu].copy()
        
        # 应用动作（扭矩）
        action = np.clip(action, -15.0, 15.0)
        self.data.ctrl[:self.nu] = action
        
        # 仿真一步
        mujoco.mj_step(self.model, self.data)
        
        # 更新步数计数器
        self.step_count += 1

        # 最大步数
        max_steps = 3000

        # 获取新状态
        state = self._get_state()
        
        # 计算奖励
        ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
        ee_pos = self.data.site_xpos[ee_site_id].copy()
        distance = np.linalg.norm(ee_pos - self.target_pos)
        reward = 0

        # 时间奖励 - 鼓励更快完成任务
        # 步数惩罚系数
        setp_penalty = 0.1  # 每步扣除0.1奖励，鼓励快速完成任务
        reward -= setp_penalty

        # 基础距离奖励
        if self.min_distance is None:
            self.min_distance = distance
            improvement_reward = 0
        elif distance < self.min_distance: # 距离变更近时给予奖励
            improvement_reward = 1 * (self.min_distance - distance)
            self.min_distance = distance
        elif distance > self.previous_distance: # 距离变更远时给予惩罚
            improvement_reward = -0.8 * (distance - self.previous_distance)
        else:
            improvement_reward = 0
        self.previous_distance = distance
        # 基础距离惩罚
        # base_distance_penalty = -distance * 0.5
        base_distance_penalty = -distance ** 0.5 * 0.8
        
        # 阶段性距离奖励（一次性）
        phase_distance_reward = 0.0
        phase_thresholds = [0.5, 0.3, 0.1, 0.05, 0.01, 0.005, 0.002]
        phase_rewards = [100.0, 200.0, 300.0, 500.0, 1000.0, 1500.0, 2000.0]
        
        for thresh, phase_reward in zip(phase_thresholds, phase_rewards):
            if distance < thresh and thresh not in self._phase_rewards_given:
                phase_distance_reward += phase_reward
                self._phase_rewards_given.add(thresh)

        # 近距离持续奖励
        close_range_bonus = 0.0
        # if distance < 0.01:
        #     # 距离越近奖励越高
        #     close_range_bonus = 3 * (0.01 - distance)/0.01
        
        # 综合距离奖励
        reward += improvement_reward + base_distance_penalty + phase_distance_reward + close_range_bonus
        
        # 速度奖励 - 鼓励在整个过程中保持适中的速度
        ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ee_link")
        ee_vel = self.data.cvel[ee_body_id][:3].copy()
        ee_speed = np.linalg.norm(ee_vel)
        
        # 速度惩罚 -  discourage high speeds
        if ee_speed > 0.5:
            reward -= 0.2

        # 计算当前运动方向与目标方向的夹角
        to_target = self.target_pos - ee_pos
        to_target /= (np.linalg.norm(to_target) + 1e-6)
        movement_dir = ee_vel / (np.linalg.norm(ee_vel) + 1e-6)
        # 夹角越小奖励越高
        direction_cos = np.dot(to_target, movement_dir)
        direction_reward = max(0, direction_cos)**2 * 1.0  # 只奖励正向运动，平方以增加奖励差异
        reward += direction_reward
                 
        # 碰撞惩罚 - 检查是否有接触
        collision_penalty = 0.0
        collision_detected = False
        for i in range(self.data.ncon):
            # 如果检测到碰撞，给予惩罚
            collision_penalty -= 5000.0
            collision_detected = True
            
        reward += collision_penalty
        
        # 添加关节速度变化惩罚项，抑制机械臂失控d
        current_joint_velocities = self.data.qvel[:self.nu].copy()
        joint_velocity_change = np.abs(current_joint_velocities - self.previous_joint_velocities)
        # 对关节速度变化进行惩罚，系数可调整
        # if distance < 0.05:
        #     # 距离越小, 希望越稳定
        #     joint_velocity_change_penalty = -0.1 * np.sum(joint_velocity_change)
        # else:
        joint_velocity_change_penalty = -0.03 * np.sum(joint_velocity_change)
        reward += joint_velocity_change_penalty
                
        # 检查是否完成
        distance_to_target = np.linalg.norm(ee_pos - self.target_pos)
        
        done = False
        # 新的成功条件：只要碰到目标位置就算成功（距离小于1mm）
        if distance_to_target <= self.success_threshold:
            done = True
            # 成功奖励
            success_reward = 10000.0
            reward += success_reward
            reward += (4*(max_steps - self.step_count))
            # 速度奖励：成功时速度越慢奖励越高
            if ee_speed < 0.01:
                speed_reward_on_success = 2000.0
            elif ee_speed < 0.05:
                speed_reward_on_success = 1000.0
            elif ee_speed < 0.1:
                speed_reward_on_success = 500.0
            else:
                speed_reward_on_success = 0.0
                
            reward += speed_reward_on_success
        
        # 如果发生碰撞，也结束episode
        if collision_detected:
            reward += -setp_penalty*(max_steps - self.step_count) # 扣除剩余的步数惩罚，防止自杀
            done = True
            
        truncated = self.step_count >= max_steps  # 最大步数限制
        
        return state, reward, done, truncated, {}

    def render(self):
        """
        渲染环境
        """
        if self.render_mode is None:
            return
            
        # 如果可视化器尚未创建，则创建一个
        if self.viewer is None:
            try:
                # 尝试创建被动式可视化器
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                # 初始化目标可视化添加标记
                self._target_viz_added = False
            except Exception as e:
                print(f"无法启动可视化器: {e}")
                return
        
        # 如果可视化器已经创建，同步更新场景
        if self.viewer:
            try:
                # 添加目标位置的可视化（绿色小球）
                self._add_target_visualization()
                self.viewer.sync()
            except Exception as e:
                print(f"可视化器同步失败: {e}")
                self.viewer = None

    def _add_target_visualization(self):
        """
        在目标位置添加一个绿色小球用于可视化
        """
        if self.viewer:
            # 获取用户场景对象
            scn = self.viewer.user_scn
            
            # 检查是否已经添加了目标可视化
            if not getattr(self, '_target_viz_added', False):
                # 增加一个几何体（小球）
                if scn.ngeom < scn.maxgeom:
                    # 获取新增几何体的引用
                    self._target_geom = scn.ngeom
                    geom = scn.geoms[scn.ngeom]
                    scn.ngeom += 1
                    # 初始化几何体为球体
                    mujoco.mjv_initGeom(
                        geom,
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.02, 0.02, 0.02]),  # 球体半径
                        self.target_pos,  # 球体位置（目标位置）
                        np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),  # 单位矩阵
                        np.array([0.0, 1.0, 0.0, 0.8])  # RGBA: 绿色，半透明
                    )
                    # 设置几何体类别为仅可视化，不参与物理碰撞
                    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
                    # 标记已添加目标可视化
                    self._target_viz_added = True
            else:
                # 更新已存在的几何体位置
                geom = scn.geoms[self._target_geom]
                geom.pos = self.target_pos
    def close(self):
        """
        关闭可视化器
        """
        if self.viewer:
            self.viewer.close()
            self.viewer = None