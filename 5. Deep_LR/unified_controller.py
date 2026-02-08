import mujoco
import mujoco.viewer
import numpy as np
import time
from arm_ik import ZeroArmIK

class ArchimedesController:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.ik_solver = ZeroArmIK()
        
        # Target states
        self.chassis_v_target = np.zeros(3) # [vx, vy, wz]
        self.arm_ee_target = np.array([0.2, 0.0, 0.3]) # Target relative to arm base

    def set_chassis_velocity(self, vx, vy, wz):
        """
        vx: forward velocity
        vy: strafe velocity (requires quad-screw)
        wz: yaw velocity
        """
        self.chassis_v_target = np.array([vx, vy, wz])

    def set_arm_target(self, x, y, z):
        self.arm_ee_target = np.array([x, y, z])

    def update_actuators(self):
        # 1. Chassis Control Logic (Screw mixing)
        # Standard differential for now (vx, wz)
        vx = self.chassis_v_target[0]
        wz = self.chassis_v_target[2]
        
        # Power distribution (Simplified)
        left_pwr = vx - wz
        right_pwr = vx + wz
        
        # Map to 4 screws
        self.data.actuator('screw_fl_ctrl').ctrl = left_pwr * 10
        self.data.actuator('screw_rl_ctrl').ctrl = left_pwr * 10
        self.data.actuator('screw_fr_ctrl').ctrl = -right_pwr * 10 # Sign depends on handedness
        self.data.actuator('screw_rr_ctrl').ctrl = -right_pwr * 10

        # 2. Arm Control Logic (IK)
        angles = self.ik_solver.solve_ik(self.arm_ee_target)
        if angles:
            self.data.actuator('arm_j1').ctrl = angles[0]
            self.data.actuator('arm_j2').ctrl = angles[1]
            self.data.actuator('arm_j3').ctrl = angles[2]
            # ... and so on

    def run_sim(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                
                # Interactive Control logic here (e.g. key handling)
                self.update_actuators()
                
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
                
                # Sleep to maintain 1x real-time
                elapsed = time.time() - step_start
                if self.model.opt.timestep > elapsed:
                    time.sleep(self.model.opt.timestep - elapsed)

if __name__ == "__main__":
    ctrl = ArchimedesController('projects/robotics/5. Deep_LR/archimedes_hand_mujoco.xml')
    # Test: Move forward and reach out
    ctrl.set_chassis_velocity(2.0, 0, 0)
    ctrl.set_arm_target(0.25, 0.05, 0.2)
    ctrl.run_sim()
